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
  // ���o�͂̌^
  using InputType = typename PreviousLayer::OutputType;
  using OutputType = std::uint8_t;
  static_assert(std::is_same<InputType, std::int32_t>::value, "");

  // ���o�͂̎�����
  static constexpr IndexType kInputDimensions =
    PreviousLayer::kOutputDimensions;
  static constexpr IndexType kOutputDimensions = kInputDimensions;

  // ���̑w�Ŏg�p���鏇�`�d�p�o�b�t�@�̃T�C�Y
  static constexpr std::size_t kSelfBufferSize = CeilToMultiple(
      kOutputDimensions * sizeof(OutputType) / 8, kCacheLineSize);

  // ���͑w���炱�̑w�܂łŎg�p���鏇�`�d�p�o�b�t�@�̃T�C�Y
  static constexpr std::size_t kBufferSize =
      PreviousLayer::kBufferSize + kSelfBufferSize;

  //! �X�P�[���p�o�b�t�@�̃C���f�b�N�X
  /*! �l���������ވʒu */
  static constexpr IndexType kScaleIndex = PreviousLayer::kScaleIndex + 1;

private:
  //! ���̑w�őł������p�����[�^�̔{��
  /*! �r�b�g���͊w�K���ʂɊ�Â��ėǂ������Ɍ��肳�ꂽ */
  static constexpr IndexType kSelfShiftScaleBits = kFeatureScaleBits - 1;

 public:
  //! �ݐς̃p�����[�^�̔{��
  static constexpr IndexType kShiftScaleBits =
      PreviousLayer::kShiftScaleBits - kSelfShiftScaleBits;

  // �]���֐��t�@�C���ɖ��ߍ��ރn�b�V���l
  static constexpr std::uint32_t GetHashValue() {
    std::uint32_t hash_value = 0xB5A1F2EFu;
    hash_value += PreviousLayer::GetHashValue();
    hash_value += 1 << kShiftScaleBits;
    return hash_value;
  }

  // ���͑w���炱�̑w�܂ł̍\����\��������
  static std::string GetStructureString() {
    return "Binarization[" + std::to_string(kOutputDimensions) + "](" +
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
    const auto output = reinterpret_cast<std::int32_t*>(buffer);
#if !defined(USE_AVX2)
#error Not Implemented.
#endif  // !defined(USE_AVX2)

    constexpr IndexType kNumInputChunks =
        kInputDimensions / kSimdWidth * sizeof(InputType);
    constexpr IndexType kNumInputSize = sizeof(InputType);

    // �l���N���b�v
    // embedding�w�̃X�P�[����ł��������̂ŁA���O�̑w�̃X�P�[���݂̂ɂȂ��Ă���
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
      // embedding�w�̕��̃X�P�[����ł�����
      const __m256i in0 = _mm256_srai_epi32(
          _mm256_stream_load_si256(&input[offset + 0]), kSelfShiftScaleBits);
      const __m256i in1 = _mm256_srai_epi32(
          _mm256_stream_load_si256(&input[offset + 1]), kSelfShiftScaleBits);
      const __m256i in2 = _mm256_srai_epi32(
          _mm256_stream_load_si256(&input[offset + 2]), kSelfShiftScaleBits);
      const __m256i in3 = _mm256_srai_epi32(
          _mm256_stream_load_si256(&input[offset + 3]), kSelfShiftScaleBits);

      //
      // 2�l��
      //
      // 64bit����in0�̑O���Ain1�̑O���Ain0�̌㔼�Ain1�̌㔼
      const __m256i a = _mm256_packs_epi32(in0, in1);
      // ���l��64bit����in2�̑O���Ain3�̑O���Ain2�̌㔼�Ain3�̌㔼
      const __m256i b = _mm256_packs_epi32(in2, in3);
      // 32bit����in0�̑O���Ain1�̑O���Ain2�̑O���Ain3�̑O���A
      //          in0�̌㔼�Ain1�̌㔼�Ain2�̌㔼�Ain3�̌㔼
      const __m256i c = _mm256_packs_epi16(a, b);
      // �eint8_t�̍ŏ�ʃr�b�g���W�߂�
      // ���̂��ߐ��Ȃ�0�A���Ȃ�1��2�l�������
      output[i] = _mm256_movemask_epi8(c);

      //
      // ��Βl�̍��v
      //
      const __m256i clipped_a =
          _mm256_min_epu16(_mm256_abs_epi16(a), max_value);
      const __m256i clipped_b =
          _mm256_min_epu16(_mm256_abs_epi16(b), max_value);
      // ���̂܂ܑ����Ă������ӂ�͂Ȃ�
      static_assert(kInputDimensions / 16 * (1 << kShiftScaleBits) <
                        std::numeric_limits<std::uint16_t>::max(),
                    "");
      sum = _mm256_adds_epi16(sum, _mm256_adds_epi16(clipped_a, clipped_b));
    }

    const __m256i mask = _mm256_set1_epi32(0x0000FFFF);
    // �אڂ���16bit�̑g�𑫂���32bit�ɂ���
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
  // �w�K�p�N���X��friend�ɂ���
  friend class Trainer<Binarization>;

  // ���̑w�̒��O�̑w
  PreviousLayer previous_layer_;
};

}  // namespace Layers

}  // namespace NNUE

}  // namespace Eval

#endif // defined(EVAL_NNUE)

#endif // !_NNUE_LAYERS_BINARIZATION_H_

