#pragma once
#ifndef _NNUE_LAYERS_BINARY_INNER_PRODUCT_H_
#define _NNUE_LAYERS_BINARY_INNER_PRODUCT_H_

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"

namespace Eval {

namespace NNUE {

namespace Layers {

template <typename PreviousLayer, IndexType OutputDimensions = 1>
class BinaryInnerProduct {
  // 入出力の型
  using InputType = typename PreviousLayer::OutputType;
  using OutputType = std::int32_t;
  static_assert(std::is_same<InputType, std::int32_t>::value, "");

  // 入出力の次元数
  static constexpr IndexType kInputDimensions =
    PreviousLayer::kOutputDimensions;
  static constexpr IndexType kOutputDimensions = OutputDimensions;
  static constexpr IndexType kPaddedInputDimensions =
    CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);

  // この層で使用する順伝播用バッファのサイズ
  static constexpr std::size_t kSelfBufferSize =
    CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

  // 入力層からこの層までで使用する順伝播用バッファのサイズ
  static constexpr std::size_t kBufferSize =
    PreviousLayer::kBufferSize + kSelfBufferSize;

  //! スケール用バッファのインデックス
  /*! 使わない */
  static constexpr IndexType kScaleIndex = PreviousLayer::kScaleIndex;

  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr std::uint32_t GetHashValue() {
    std::uint32_t hash_value = 0xBA3BA60Cu;
    hash_value += kOutputDimensions;
    hash_value ^= PreviousLayer::GetHashValue() >> 1;
    hash_value ^= PreviousLayer::GetHashValue() << 31;
    return hash_value;
  }

  // 入力層からこの層までの構造を表す文字列
  static std::string GetStructureString() {
    return "BinaryInnerProduct[" +
      std::to_string(kOutputDimensions) + "<-" +
      std::to_string(kInputDimensions) + "](" +
      PreviousLayer::GetStructureString() + ")";
  }

  // パラメータを読み込む
  bool ReadParameters(std::istream& stream) {
    if (!previous_layer_.ReadParameters(stream)) return false;
    stream.read(reinterpret_cast<char*>(&biases_),
      kOutputDimensions * sizeof(BiasType));
    stream.read(
      reinterpret_cast<char*>(weights_),
      kOutputDimensions * kPaddedInputDimensions * sizeof(WeightType));
    stream.read(reinterpret_cast<char*>(&scales_),
      kOutputDimensions * sizeof(ScaleType));
    return !stream.fail();
  }

  // パラメータを書き込む
  bool WriteParameters(std::ostream& stream) const {
    if (!previous_layer_.WriteParameters(stream)) return false;
    stream.write(reinterpret_cast<const char*>(&biases_),
      kOutputDimensions * sizeof(BiasType));
    stream.write(
      reinterpret_cast<const char*>(weights_),
      kOutputDimensions * kPaddedInputDimensions * sizeof(WeightType));
    stream.write(reinterpret_cast<const char*>(&scales_),
      kOutputDimensions * sizeof(ScaleType));
    return !stream.fail();
  }

  // 順伝播
  const OutputType* Propagate(
      const TransformedFeatureType* transformed_features, char* buffer,
      std::int32_t* scale_buffer) const {
    const auto input = previous_layer_.Propagate(
        transformed_features, buffer + kSelfBufferSize, scale_buffer);
    const auto output = reinterpret_cast<OutputType*>(buffer);
#if !defined(USE_AVX2)
#error Not Implemented.
#endif  // !defined(USE_AVX2)

    constexpr IndexType kNumChunks =
        kInputDimensions / kSimdWidth * sizeof(InputType);
    __m256i sum = _mm256_setzero_si256();
    for (IndexType i = 0; i < kNumChunks; ++i) {
      const auto x = _mm256_stream_load_si256(&input[i]);

      const IndexType index = kSimdWidth / sizeof(WeightType) * i;
      const auto w = _mm256_load_si256(&weights_[index]);

      // xとwを要素ごとに掛け算
      // 補数で1を足す処理はバイアス項にまとめた
      const auto y = _mm256_xor_si256(x, w);

      sum = _mm256_add_epi32(sum, y);
    }
    // 水平方向に合計

    // sum2, sum3, sum0, sum1, sum6, sum7 sum4, sum5
    const __m256i shuffle1 = _mm256_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
    // sum0+sum2, sum1+sum3, _, _, sum4+sum6, sum5+sum7, _, _
    const __m256i total1 = _mm256_add_epi32(sum, shuffle1);

    // sum1+sum3, sum0+sum2, _, _,  sum5+sum7, sum4+sum6, _, _
    const _m256i shuffle2 =
        _mm256_shuffle_epi32(total1, _MM_SHUFFLE(2, 3, 0, 1));
    // sum0+sum1+sum2+sum3, _, _, _, sum4+sum5+sum6+sum7, _, _, _
    const _m256i total2 = _mm256_add_epi32(total1, shuffle2);

    const __m128i hi = _mm256_extractf128_si256(total2, 1);
    const __m128i lo = _mm256_extractf128_si256(total2, 0);

    const int32_t v = _mm_cvtsi128_si32(lo) + _mm_cvtsi128_si32(hi);

    // まだ前のbinary denseの入力と重みについて平均にしていない(合計のまま)
    constexpr IndexType shift = Log2<InputDimensions>::value * 2;

    // まだこの層での重みについての平均が残っている
    // この後でFV_SCALEで割る処理があるので、そこで一緒に行う
    output[0] = (v >> shift) * scale_ + bias_;

    return buffer;
  }

 private:
  // パラメータの型
  using ScaleType = OutputType;
  using BiasType = OutputType;
  using WeightType = InputType;

  // 学習用クラスをfriendにする
  friend class Trainer<AffineTransform>;

  // この層の直前の層
  PreviousLayer previous_layer_;

  // パラメータ
  ScaleType scale_;
  //! バイアス項
  /*! 2の補数で1を加える分はここに含まれている */
  BiasType bias_;
  alignas(kCacheLineSize)
    WeightType weights_[kOutputDimensions];
};

}  // namespace Layers

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif // !_NNUE_LAYERS_BINARY_INNER_PRODUCT_H_

