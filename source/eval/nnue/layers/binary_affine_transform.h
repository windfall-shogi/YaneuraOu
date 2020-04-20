#pragma once
#ifndef _NNUE_LAYERS_BINARY_AFFINE_TRANSFORM_H_
#define _NNUE_LAYERS_BINARY_AFFINE_TRANSFORM_H_

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"

#include <array>

namespace Eval {

namespace NNUE {

namespace Layers {

template<typename PreviousLayer, IndexType OutputDimensions>
class BinaryAffineTransform {
public:
  // 入出力の型
  using InputType = typename PreviousLayer::OutputType;
  using OutputType = std::int32_t;
  static_assert(std::is_same<InputType, std::uint8_t>::value, "");

  // 入出力の次元数
  static constexpr IndexType kInputDimensions =
    PreviousLayer::kOutputDimensions;
  static constexpr IndexType kOutputDimensions = OutputDimensions;
  static constexpr IndexType kPaddedInputDimensions =
    CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);

  // 水平方向に合計できる範囲
  using BlockType = uint64_t;
  // 64bitの倍数
  static_assert(kInputDimensions % (sizeof(BlockType) * 8) == 0, "");
  static_assert(kPaddedInputDimensions % (sizeof(BlockType) * 8) == 0, "");

  // この層で使用する順伝播用バッファのサイズ
  static constexpr std::size_t kSelfBufferSize =
    CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

  // 入力層からこの層までで使用する順伝播用バッファのサイズ
  static constexpr std::size_t kBufferSize =
    PreviousLayer::kBufferSize + kSelfBufferSize;

  //! スケール用バッファのインデックス
  /*! 値を読み取る位置 */
  static constexpr IndexType kScaleIndex = PreviousLayer::kScaleIndex;


  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr std::uint32_t GetHashValue() {
    std::uint32_t hash_value = 0x6C19E9D8u;
    hash_value += kOutputDimensions;
    hash_value ^= PreviousLayer::GetHashValue() >> 1;
    hash_value ^= PreviousLayer::GetHashValue() << 31;
    return hash_value;
  }

  // 入力層からこの層までの構造を表す文字列
  static std::string GetStructureString() {
    return "BinaryAffineTransform[" +
      std::to_string(kOutputDimensions) + "<-" +
      std::to_string(kInputDimensions) + "](" +
      PreviousLayer::GetStructureString() + ")";
  }

  // パラメータを読み込む
  bool ReadParameters(std::istream& stream) {
    if (!previous_layer_.ReadParameters(stream)) return false;
    stream.read(reinterpret_cast<char*>(biases_),
                kOutputDimensions * sizeof(BiasType));
    stream.read(
        reinterpret_cast<char*>(weights_),
        kOutputDimensions * kPaddedInputDimensions * sizeof(WeightType) / 8);
    stream.read(reinterpret_cast<char*>(scales_),
                kOutputDimensions * sizeof(ScaleType));
    return !stream.fail();
  }

  // パラメータを書き込む
  bool WriteParameters(std::ostream& stream) const {
    if (!previous_layer_.WriteParameters(stream)) return false;
    stream.write(reinterpret_cast<const char*>(biases_),
                 kOutputDimensions * sizeof(BiasType));
    stream.write(
        reinterpret_cast<const char*>(weights_),
        kOutputDimensions * kPaddedInputDimensions * sizeof(WeightType) / 8);
    stream.write(reinterpret_cast<const char*>(scales_),
                 kOutputDimensions * sizeof(ScaleType));
    return !stream.fail();
  }

  // 順伝播
  const OutputType* Propagate(
      const TransformedFeatureType* transformed_features, char* buffer,
      std::int32_t* scale_buffer) const {
    const auto input = previous_layer_.Propagate(
      transformed_features, buffer + kSelfBufferSize, scale_buffer);
    const auto output = reinterpret_cast<__m256i*>(buffer);
#if !defined(USE_AVX2)
#error Not Implemented.
#endif // !defined(USE_AVX2)

    // 64bitずつを同時にいくつ処理できるか
    constexpr IndexType kNumblocks = kSimdWidth / sizeof(BlockType);
    static_assert(kNumblocks == 4, "");
    // 入力ベクトルを何回に分けて積を計算するか
    // ブロック数
    // 512bitなら8
    constexpr IndexType kNumInputChucks =
        kInputDimensions / sizeof(BlockType) / 8;
    // 出力ベクトルを何回に分けて計算するか
    // 1回の処理で4個計算できる
    // 256次元なら64
    constexpr IndexType kNumOutputChunks = kOutputDimensions / kNumblocks;

    // 1blockあたりに32bitの要素がいくつ含まれるか
    constexpr IndexType kNumElements = sizeof(BlockType) / sizeof(OutputType);
    static_assert(kNumElements == 2, "");
    // 配列の外側にアクセスするのを防ぐ
    static_assert(kOutputDimensions % (kNumblocks * kNumElements) == 0, "");

    const __m256i vmask = _mm256_set1_epi8(0xF);
    const __m256i vpop1 =
        _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1,
                         1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i vpop2 =
        _mm256_setr_epi8(8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 8, 7,
                         7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4);

    std::array<__m256i, kNumInputChucks> input_list;
    for (IndexType i = 0; i < kNumInputChucks; ++i) {
      input_list[i] =
          _mm256_set1_epi64x(reinterpret_cast<const std::int64_t*>(input)[i]);
    }

    const auto weights = reinterpret_cast<const __m256i*>(weights_);
    const auto weight_scales = reinterpret_cast<const __m256i*>(scales_);
    const auto biases = reinterpret_cast<const __m256i*>(biases_);
    const auto input_scale = _mm256_set1_epi32(scale_buffer[kScaleIndex]);

    for (IndexType i = 0; i < kNumOutputChunks; ++i) {
      __m256i result = _mm256_setzero_si256();
      for (IndexType j = 0; j < kNumElements; ++j) {
        __m256i sum_a = _mm256_setzero_si256();
        __m256i sum_b = _mm256_setzero_si256();
        const IndexType offset = i * kNumElements + j;
        for (IndexType k = 0; k < kNumInputChucks; ++k) {
          const __m256i& in = input_list[k];
          const auto w = _mm256_load_si256(&weights[offset * kNumInputChucks + k]);

          // xor
          const __m256i xor = _mm256_xor_si256(in, w);

          // 0の個数を数える
          // https://qiita.com/Seizh/items/26eef63af739ba48e36b
          __m256i a = _mm256_and_si256(xor, vmask);
          __m256i b = _mm256_and_si256(_mm256_srli_epi64(xor, 4), vmask);

          a = _mm256_shuffle_epi8(vpop1, a);
          b = _mm256_shuffle_epi8(vpop2, b);

          sum_a = _mm256_add_epi8(sum_a, a);
          sum_b = _mm256_add_epi8(sum_b, b);
        }
        const __m256i sum = _mm256_sad_epu8(sum_a, smu_b);
        result = _mm256_or_si256(
          result, _mm256_slli_epi64(sum, j * sizeof(ScaleType) * 8));
      }
      result = _mm256_madd_epi16(result, _mm256_load_si256(&weight_scales[i]));
      result = _mm256_mullo_epi32(result, input_scale);

      _mm256_store_si256(
          &outputs[i], _mm256_add_epi32(result, _mm256_load_si256(&biases[i])));
    }

    return buffer;
  }

private:
  // パラメータの型
  using BiasType = OutputType;
  using WeightType = std::int8_t;
  using ScaleType = OutputType;

  // 学習用クラスをfriendにする
  friend class Trainer<AffineTransform>;

  // この層の直前の層
  PreviousLayer previous_layer_;

  // パラメータ
  alignas(kCacheLineSize) BiasType biases_[kOutputDimensions];
  alignas(kCacheLineSize)
    WeightType weights_[kOutputDimensions * kPaddedInputDimensions / 8];
  alignas(kCacheLineSize) ScaleType scales_[kOutputDimensions];
};

}  // namespace Layers

}  // namespace NNUE

}  // namespace Eval

#endif // defined(EVAL_NNUE)

#endif // !_NNUE_LAYERS_BINARY_AFFINE_TRANSFORM_H_

