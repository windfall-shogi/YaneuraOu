#pragma once
#ifndef _NNUE_LAYERS_BINARIZATION_H_
#define _NNUE_LAYERS_BINARIZATION_H_

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"

#include <limits>

namespace Eval {

namespace NNUE {

namespace Layers {

template<typename PreviousLayer>
class Binarization {
public:
  // 入出力の型
  using InputType = typename PreviousLayer::OutputType;
  using OutputType = std::uint8_t;
  static_assert(std::is_same<InputType, std::int32_t>::value, "");

  // 入出力の次元数
  static constexpr IndexType kInputDimensions =
    PreviousLayer::kOutputDimensions;
  static constexpr IndexType kOutputDimensions = kInputDimensions;

  // この層で使用する順伝播用バッファのサイズ
  static constexpr std::size_t kSelfBufferSize = CeilToMultiple(
      kOutputDimensions * sizeof(OutputType) / 8, kCacheLineSize);

  // 入力層からこの層までで使用する順伝播用バッファのサイズ
  static constexpr std::size_t kBufferSize =
      PreviousLayer::kBufferSize + kSelfBufferSize;

  //! スケール用バッファのインデックス
  /*! 値を書き込む位置 */
  static constexpr IndexType kScaleIndex = PreviousLayer::kScaleIndex + 1;

private:
  //! この層で打ち消すパラメータの倍率
  /*! ビット数は学習結果に基づいて良い感じに決定された */
  static constexpr IndexType kSelfShiftScaleBits = kFeatureScaleBits - 1;

 public:
  //! 累積のパラメータの倍率
  static constexpr IndexType kShiftScaleBits =
      PreviousLayer::kShiftScaleBits - kSelfShiftScaleBits;

  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr std::uint32_t GetHashValue() {
    std::uint32_t hash_value = 0xB5A1F2EFu;
    hash_value += PreviousLayer::GetHashValue();
    hash_value += 1 << kShiftScaleBits;
    return hash_value;
  }

  // 入力層からこの層までの構造を表す文字列
  static std::string GetStructureString() {
    return "Binarization[" + std::to_string(kOutputDimensions) + "](" +
           PreviousLayer::GetStructureString() + ")";
  }

  // パラメータを読み込む
  bool ReadParameters(std::istream& stream) {
    return previous_layer_.ReadParameters(stream);
  }

  // パラメータを書き込む
  bool WriteParameters(std::ostream& stream) const {
    return previous_layer_.WriteParameters(stream);
  }

  // 順伝播
  const OutputType* Propagate(
      const TransformedFeatureType* transformed_features, char* buffer,
      std::uint32_t* scale_buffer) const {
    const auto input =
        reinterpret_cast<const __m256i*>(previous_layer_.Propagate(
            transformed_features, buffer + kSelfBufferSize, scale_buffer));
    const auto output = reinterpret_cast<std::int32_t*>(buffer);
#if !defined(USE_AVX2)
#error Not Implemented.
#endif  // !defined(USE_AVX2)

    constexpr IndexType kNumInputChunks =
        kInputDimensions / kSimdWidth * sizeof(InputType);
    constexpr IndexType kNumInputSize = sizeof(InputType);

    // 値をクリップ
    // embedding層のスケールを打ち消したので、直前の層のスケールのみになっている
    const __m256i max_value = _mm256_set1_epi16(1 << kShiftScaleBits);

    //sync_cout << "--- Inputs: " << "binarization" << " ---" << std::endl;
    //for (int i = 0; i < kInputDimensions; ++i) {
    //  std::cout << (reinterpret_cast<const int32_t*>(input)[i] >> 11) << ' ';
    //  if (i % 8 == 7) {
    //    std::cout << std::endl;
    //  }
    //}
    //std::cout << std::endl << "-----" << sync_endl;


    __m256i sum = _mm256_setzero_si256();
    for (IndexType i = 0; i < kNumInputChunks / kNumInputSize; ++i) {
      const IndexType offset = i * kNumInputSize;
      // embedding層の分のスケールを打ち消す
      const __m256i in0 = _mm256_srai_epi32(
          _mm256_stream_load_si256(&input[offset + 0]), kSelfShiftScaleBits);
      const __m256i in1 = _mm256_srai_epi32(
          _mm256_stream_load_si256(&input[offset + 1]), kSelfShiftScaleBits);
      const __m256i in2 = _mm256_srai_epi32(
          _mm256_stream_load_si256(&input[offset + 2]), kSelfShiftScaleBits);
      const __m256i in3 = _mm256_srai_epi32(
          _mm256_stream_load_si256(&input[offset + 3]), kSelfShiftScaleBits);

      //
      // 2値化
      //
      // 64bitずつin0の前半、in1の前半、in0の後半、in1の後半
      const __m256i a = _mm256_packs_epi32(in0, in1);
      // 同様に64bitずつin2の前半、in3の前半、in2の後半、in3の後半
      const __m256i b = _mm256_packs_epi32(in2, in3);
      // 32bitずつin0の前半、in1の前半、in2の前半、in3の前半、
      //          in0の後半、in1の後半、in2の後半、in3の後半
      const __m256i c = _mm256_packs_epi16(a, b);
      // 各int8_tの最上位ビットを集める
      // そのため正なら0、負なら1で2値化される
      output[i] = _mm256_movemask_epi8(c);

      //
      // 絶対値の合計
      //
      const __m256i clipped_a =
          _mm256_min_epu16(_mm256_abs_epi16(a), max_value);
      const __m256i clipped_b =
          _mm256_min_epu16(_mm256_abs_epi16(b), max_value);
      // そのまま足しても桁あふれはない
      static_assert(kInputDimensions / 16 * (1 << kShiftScaleBits) <
                        std::numeric_limits<std::uint16_t>::max(),
                    "");
      sum = _mm256_adds_epi16(sum, _mm256_adds_epi16(clipped_a, clipped_b));
    }

    const __m256i mask = _mm256_set1_epi32(0x0000FFFF);
    // 隣接する16bitの組を足して32bitにする
    const __m256i sum32 =
        _mm256_add_epi32(_mm256_and_si256(sum, mask),
                         _mm256_and_si256(_mm256_srai_epi32(sum, 16), mask));

    // sum2, sum3, sum0, sum1, sum6, sum7 sum4, sum5
    const __m256i shuffle1 = _mm256_shuffle_epi32(sum32, _MM_SHUFFLE(1, 0, 3, 2));
    // sum0+sum2, sum1+sum3, _, _, sum4+sum6, sum5+sum7, _, _
    const __m256i total1 = _mm256_add_epi32(sum32, shuffle1);

    // sum1+sum3, sum0+sum2, _, _,  sum5+sum7, sum4+sum6, _, _
    const __m256i shuffle2 =
      _mm256_shuffle_epi32(total1, _MM_SHUFFLE(2, 3, 0, 1));
    // sum0+sum1+sum2+sum3, _, _, _, sum4+sum5+sum6+sum7, _, _, _
    const __m256i total2 = _mm256_add_epi32(total1, shuffle2);

    const __m128i hi = _mm256_extractf128_si256(total2, 1);
    const __m128i lo = _mm256_extractf128_si256(total2, 0);

    const int32_t v = _mm_cvtsi128_si32(lo) + _mm_cvtsi128_si32(hi);
    scale_buffer[kScaleIndex] = v;

    return reinterpret_cast<OutputType*>(buffer);
  }

private:
  // 学習用クラスをfriendにする
  friend class Trainer<Binarization>;

  // この層の直前の層
  PreviousLayer previous_layer_;
};

}  // namespace Layers

}  // namespace NNUE

}  // namespace Eval

#endif // defined(EVAL_NNUE)

#endif // !_NNUE_LAYERS_BINARIZATION_H_

