#pragma once
#ifndef _NNUE_LAYERS_LEAKY_RELU_H_
#define _NNUE_LAYERS_LEAKY_RELU_H_

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"

namespace Eval {

namespace NNUE {

namespace Layers {

template<typename PreviousLayer>
class LeakyReLU {
public:
  // 入出力の型
  using InputType = typename PreviousLayer::OutputType;
  using OutputType = std::int32_t;
  static_assert(std::is_same<InputType, std::int32_t>::value, "");
  // int16_tの方が望ましいが入力の値の範囲が良くわからないので、
  // 安全のためにint32_tにする
  static_assert(std::is_same_v<InputType, OutputType>, "");

  // 入出力の次元数
  static constexpr IndexType kInputDimensions =
    PreviousLayer::kOutputDimensions;
  static constexpr IndexType kOutputDimensions = kInputDimensions;

  // この層で使用する順伝播用バッファのサイズ
  static constexpr std::size_t kSelfBufferSize =
    CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

  // 入力層からこの層までで使用する順伝播用バッファのサイズ
  static constexpr std::size_t kBufferSize =
    PreviousLayer::kBufferSize + kSelfBufferSize;

  //! スケール用バッファのインデックス
  /*! 使わない */
  static constexpr IndexType kScaleIndex = PreviousLayer::kScaleIndex;

 private:
  //! この層で打ち消すパラメータの倍率
  /*! 何かいい感じの値 */
  static constexpr IndexType kSelfShiftScaleBits = 16;

 public:
  //! 累積のパラメータの倍率
  static constexpr IndexType kShiftScaleBits =
          PreviousLayer::kShiftScaleBits - kSelfShiftScaleBits;

  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr std::uint32_t GetHashValue() {
    std::uint32_t hash_value = 0xBB7BA525u;
    hash_value += PreviousLayer::GetHashValue();
    hash_value += 1 << kShiftScaleBits;
    return hash_value;
  }

  // 入力層からこの層までの構造を表す文字列
  static std::string GetStructureString() {
    return "LeakyReLU[" +
      std::to_string(kOutputDimensions) + "](" +
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
    const auto output = reinterpret_cast<__m256i*>(buffer);
#if !defined(USE_AVX2)
#error Not Implemented.
#endif  // !defined(USE_AVX2)

    constexpr IndexType kNumInputChunks =
        kInputDimensions / kSimdWidth * sizeof(InputType);
    for (IndexType i = 0; i < kNumInputChunks; ++i) {
      // 正の値を1.5倍、負の値を0.5倍する
      // 頑張れば正の値をそのままで負の値を2^p倍(pは負の整数)にできるが、
      // 計算コストをかけるメリットがあるのかは不明
      const __m256i x = _mm256_load_si256(&input[i]);
      const __m256i y = _mm256_srli_epi32(_mm256_abs_epi32(x), 1);
      const __m256i z = _mm256_add_epi32(x, y);
      // 桁数を調整
      _mm256_store_si256(&output[i], _mm256_srai_epi32(z, kSelfShiftScaleBits));
    }
    return reinterpret_cast<OutputType*>(buffer);
  }

private:
  // 学習用クラスをfriendにする
  friend class Trainer<LeakyReLU>;

  // この層の直前の層
  PreviousLayer previous_layer_;
};

}  // namespace Layers

}  // namespace NNUE

}  // namespace Eval

#endif // defined(EVAL_NNUE)

#endif // !_NNUE_LAYERS_LEAKY_RELU_H_

