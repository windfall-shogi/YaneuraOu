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
  // ���o�͂̌^
  using InputType = typename PreviousLayer::OutputType;
  using OutputType = std::int32_t;
  static_assert(std::is_same<InputType, std::int32_t>::value, "");

  // ���o�͂̎�����
  static constexpr IndexType kInputDimensions =
    PreviousLayer::kOutputDimensions;
  static constexpr IndexType kOutputDimensions = OutputDimensions;
  static constexpr IndexType kPaddedInputDimensions =
    CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);

  // ���̑w�Ŏg�p���鏇�`�d�p�o�b�t�@�̃T�C�Y
  static constexpr std::size_t kSelfBufferSize =
    CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

  // ���͑w���炱�̑w�܂łŎg�p���鏇�`�d�p�o�b�t�@�̃T�C�Y
  static constexpr std::size_t kBufferSize =
    PreviousLayer::kBufferSize + kSelfBufferSize;

  //! �X�P�[���p�o�b�t�@�̃C���f�b�N�X
  /*! �g��Ȃ� */
  static constexpr IndexType kScaleIndex = PreviousLayer::kScaleIndex;

  // �]���֐��t�@�C���ɖ��ߍ��ރn�b�V���l
  static constexpr std::uint32_t GetHashValue() {
    std::uint32_t hash_value = 0xBA3BA60Cu;
    hash_value += kOutputDimensions;
    hash_value ^= PreviousLayer::GetHashValue() >> 1;
    hash_value ^= PreviousLayer::GetHashValue() << 31;
    return hash_value;
  }

  // ���͑w���炱�̑w�܂ł̍\����\��������
  static std::string GetStructureString() {
    return "BinaryInnerProduct[" +
      std::to_string(kOutputDimensions) + "<-" +
      std::to_string(kInputDimensions) + "](" +
      PreviousLayer::GetStructureString() + ")";
  }

  // �p�����[�^��ǂݍ���
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

  // �p�����[�^����������
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

  // ���`�d
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

      // x��w��v�f���ƂɊ|���Z
      // �␔��1�𑫂������̓o�C�A�X���ɂ܂Ƃ߂�
      const auto y = _mm256_xor_si256(x, w);

      sum = _mm256_add_epi32(sum, y);
    }
    // ���������ɍ��v

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

    // �܂��O��binary dense�̓��͂Əd�݂ɂ��ĕ��ςɂ��Ă��Ȃ�(���v�̂܂�)
    constexpr IndexType shift = Log2<InputDimensions>::value * 2;

    // �܂����̑w�ł̏d�݂ɂ��Ă̕��ς��c���Ă���
    // ���̌��FV_SCALE�Ŋ��鏈��������̂ŁA�����ňꏏ�ɍs��
    output[0] = (v >> shift) * scale_ + bias_;

    return buffer;
  }

 private:
  // �p�����[�^�̌^
  using ScaleType = OutputType;
  using BiasType = OutputType;
  using WeightType = InputType;

  // �w�K�p�N���X��friend�ɂ���
  friend class Trainer<AffineTransform>;

  // ���̑w�̒��O�̑w
  PreviousLayer previous_layer_;

  // �p�����[�^
  ScaleType scale_;
  //! �o�C�A�X��
  /*! 2�̕␔��1�������镪�͂����Ɋ܂܂�Ă��� */
  BiasType bias_;
  alignas(kCacheLineSize)
    WeightType weights_[kOutputDimensions];
};

}  // namespace Layers

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif // !_NNUE_LAYERS_BINARY_INNER_PRODUCT_H_

