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
  // ���o�͂̌^
  using InputType = typename PreviousLayer::OutputType;
  using OutputType = std::int32_t;
  static_assert(std::is_same<InputType, std::int32_t>::value, "");
  // int16_t�̕����]�܂��������͂̒l�͈̔͂��ǂ��킩��Ȃ��̂ŁA
  // ���S�̂��߂�int32_t�ɂ���
  static_assert(std::is_same_v<InputType, OutputType>, "");

  // ���o�͂̎�����
  static constexpr IndexType kInputDimensions =
    PreviousLayer::kOutputDimensions;
  static constexpr IndexType kOutputDimensions = kInputDimensions;

  // ���̑w�Ŏg�p���鏇�`�d�p�o�b�t�@�̃T�C�Y
  static constexpr std::size_t kSelfBufferSize =
    CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

  // ���͑w���炱�̑w�܂łŎg�p���鏇�`�d�p�o�b�t�@�̃T�C�Y
  static constexpr std::size_t kBufferSize =
    PreviousLayer::kBufferSize + kSelfBufferSize;

  //! �X�P�[���p�o�b�t�@�̃C���f�b�N�X
  /*! �g��Ȃ� */
  static constexpr IndexType kScaleIndex = PreviousLayer::kScaleIndex;

 private:
  //! ���̑w�őł������p�����[�^�̔{��
  /*! �������������̒l */
  static constexpr IndexType kSelfShiftScaleBits = 16;

 public:
  //! �ݐς̃p�����[�^�̔{��
  static constexpr IndexType kShiftScaleBits =
          PreviousLayer::kShiftScaleBits - kSelfShiftScaleBits;

  // �]���֐��t�@�C���ɖ��ߍ��ރn�b�V���l
  static constexpr std::uint32_t GetHashValue() {
    std::uint32_t hash_value = 0xBB7BA525u;
    hash_value += PreviousLayer::GetHashValue();
    hash_value += 1 << kShiftScaleBits;
    return hash_value;
  }

  // ���͑w���炱�̑w�܂ł̍\����\��������
  static std::string GetStructureString() {
    return "LeakyReLU[" +
      std::to_string(kOutputDimensions) + "](" +
      PreviousLayer::GetStructureString() + ")";
  }

  // �p�����[�^��ǂݍ���
  bool ReadParameters(std::istream& stream) {
    return previous_layer_.ReadParameters(stream);
  }

  // �p�����[�^����������
  bool WriteParameters(std::ostream& stream) const {
    return previous_layer_.WriteParameters(stream);
  }

  // ���`�d
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
      // ���̒l��1.5�{�A���̒l��0.5�{����
      // �撣��ΐ��̒l�����̂܂܂ŕ��̒l��2^p�{(p�͕��̐���)�ɂł��邪�A
      // �v�Z�R�X�g�������郁���b�g������̂��͕s��
      const __m256i x = _mm256_load_si256(&input[i]);
      const __m256i y = _mm256_srli_epi32(_mm256_abs_epi32(x), 1);
      const __m256i z = _mm256_add_epi32(x, y);
      // �����𒲐�
      _mm256_store_si256(&output[i], _mm256_srai_epi32(z, kSelfShiftScaleBits));
    }
    return reinterpret_cast<OutputType*>(buffer);
  }

private:
  // �w�K�p�N���X��friend�ɂ���
  friend class Trainer<LeakyReLU>;

  // ���̑w�̒��O�̑w
  PreviousLayer previous_layer_;
};

}  // namespace Layers

}  // namespace NNUE

}  // namespace Eval

#endif // defined(EVAL_NNUE)

#endif // !_NNUE_LAYERS_LEAKY_RELU_H_

