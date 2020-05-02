// NNUE評価関数の入力特徴量の変換を行うクラス

#ifndef _NNUE_FEATURE_TRANSFORMER_H_
#define _NNUE_FEATURE_TRANSFORMER_H_

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <cstring> // std::memset()

namespace Eval {

namespace NNUE {

// 入力特徴量変換器
class FeatureTransformer {
 private:
  // 片側分の出力の次元数
  static constexpr IndexType kHalfDimensions = kTransformedFeatureDimensions;

 public:
  // 出力の型
  using OutputType = TransformedFeatureType;
  //! スケールの型
  using ScaleType = uint32_t;


  // 入出力の次元数
  static constexpr IndexType kInputDimensions = RawFeatures::kDimensions;
  static constexpr IndexType kOutputDimensions = kHalfDimensions * 2;

  // 順伝播用バッファのサイズ
  static constexpr std::size_t kBufferSize =
      kOutputDimensions * sizeof(OutputType) / 8;

  //! スケール用バッファのインデックス
  /*! 値を読み取る位置 */
  static constexpr IndexType kScaleIndex = 0;

  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr std::uint32_t GetHashValue() {
    return RawFeatures::kHashValue ^ kOutputDimensions + kFeatureScaleBits;
  }

  // 構造を表す文字列
  static std::string GetStructureString() {
    return RawFeatures::GetName() + "[" +
        std::to_string(kInputDimensions) + "->" +
        std::to_string(kHalfDimensions) + "x2]";
  }

  // パラメータを読み込む
  bool ReadParameters(std::istream& stream) {
    stream.read(reinterpret_cast<char*>(biases_),
                kHalfDimensions * sizeof(BiasType));
    stream.read(reinterpret_cast<char*>(weights_),
                kHalfDimensions * kInputDimensions * sizeof(WeightType));
    return !stream.fail();
  }

  // パラメータを書き込む
  bool WriteParameters(std::ostream& stream) const {
    stream.write(reinterpret_cast<const char*>(biases_),
                 kHalfDimensions * sizeof(BiasType));
    stream.write(reinterpret_cast<const char*>(weights_),
                 kHalfDimensions * kInputDimensions * sizeof(WeightType));
    return !stream.fail();
  }

  // 可能なら差分計算を進める
  bool UpdateAccumulatorIfPossible(const Position& pos) const {
    const auto now = pos.state();
    if (now->accumulator.computed_accumulation) {
      return true;
    }
    const auto prev = now->previous;
    if (prev && prev->accumulator.computed_accumulation) {
      UpdateAccumulator(pos);
      return true;
    }
    return false;
  }

  // 入力特徴量を変換する
  void Transform(const Position& pos, OutputType* output, ScaleType* scale_buffer,
                 bool refresh) const {
    if (refresh || !UpdateAccumulatorIfPossible(pos)) {
      RefreshAccumulator(pos);
    }

    constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
    // スケール用の合計
    // 32bitのうちの下位11bit
    __m256i sum_lo = _mm256_setzero_si256();
    // スケール用の合計
    // sum_loの担当より上のbit
    __m256i sum_hi = _mm256_setzero_si256();
    // 上位5bitのマスク
    constexpr uint16_t mask_value = 0xF800;
    constexpr IndexType mask_size = kFeatureScaleBits;
    static_assert((mask_value & ((1 << mask_size) - 1)) == 0, "");
    static_assert(mask_value == static_cast<uint16_t>(~((1 << mask_size) - 1)),
      "");
    const __m256i sum_mask = _mm256_set1_epi16(mask_value);

    // 値をクリップ
    const __m256i max_value = _mm256_set1_epi16(1 << mask_size);

    const auto& accumulation = pos.state()->accumulator.accumulation;
    const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
    for (IndexType p = 0; p < 2; ++p) {
      const IndexType offset = kHalfDimensions * p / 8;

      auto out = reinterpret_cast<int32_t*>(&output[offset]);

      const auto ptr = accumulation[perspectives[p]];
      __m256i mask = _mm256_set1_epi16(1);
      for (IndexType j = 0; j < kNumChunks; ++j) {
        const auto tmp = reinterpret_cast<const __m256i*>(ptr[0]);
        __m256i v1 = _mm256_load_si256(&tmp[2 * j + 0]);
        __m256i v2 = _mm256_load_si256(&tmp[2 * j + 1]);
        for (IndexType k = 1; k < kRefreshTriggers.size(); ++k) {
          const auto tmp2 = reinterpret_cast<const __m256i*>(ptr[k]);
          v1 = _mm256_add_epi16(v1, _mm256_load_si256(&tmp2[2 * j + 0]));
          v2 = _mm256_add_epi16(v2, _mm256_load_si256(&tmp2[2 * j + 1]));
        }
        //
        // 2値化
        //
        // 64bitずつ v1の前半, v2の前半, v1の後半, v2の後半となる
        const __m256i packed = _mm256_packs_epi16(v1, v2);
        // 各int8_tの最上位ビットを集める
        // そのため正なら0、負なら1で2値化される
        out[j] = _mm256_movemask_epi8(packed);

        //
        // 入力ベクトルのスケールを計算
        //
        // 絶対値の合計
        const auto clipped1 = _mm256_min_epi16(_mm256_abs_epi16(v1), max_value);
        const auto clipped2 = _mm256_min_epi16(_mm256_abs_epi16(v2), max_value);
        sum_lo = _mm256_adds_epi16(sum_lo, clipped1);
        sum_lo = _mm256_adds_epi16(sum_lo, clipped2);
      }
      // オーバーフローしない
      static_assert((1 << mask_size) * kNumChunks * 2 + ((1 << mask_size) - 1) <
                        std::numeric_limits<uint16_t>::max(),
                    "");
      sum_hi = _mm256_adds_epi16(sum_hi, _mm256_srai_epi16(sum_lo, mask_size));
      sum_lo = _mm256_andnot_si256(sum_mask, sum_lo);
    }
    //
    // 水平方向に合計
    //
    // loの前半とhiの前半
    const __m256i a = _mm256_permute2x128_si256(sum_lo, sum_hi, 0x20);
    // loの後半とhiの後半
    const __m256i b = _mm256_permute2x128_si256(sum_lo, sum_hi, 0x31);
    const __m256i c = _mm256_adds_epi16(a, b);  // 16bitの値が16個

    // clang-format off
    const __m256i order =
      _mm256_set_epi8(0xF, 0x7, 0xD, 0x5, 0xB, 0x3, 0x9, 0x1,
                      0xE, 0x6, 0xC, 0x4, 0xA, 0x2, 0x8, 0x0,
                      0xF, 0x7, 0xD, 0x5, 0xB, 0x3, 0x9, 0x1,
                      0xE, 0x6, 0xC, 0x4, 0xA, 0x2, 0x8, 0x0);
    // clang-format on
    // 64bitずつでloの下位8bit、loの上位8bit、hiの下位8bit、hiの上位8bitに並び替えた
    const __m256i shuffled1 = _mm256_shuffle_epi8(c, order);

    // 64bitの範囲ずつ水平方向に合計
    const __m256i zeros = _mm256_setzero_si256();
    const __m256i s = _mm256_sad_epu8(shuffled1, zeros);
    // 32bitの範囲で正しい位置にそれぞれシフト
    // loの13bitより上をhiに含めた
    const __m256i shift = _mm256_set_epi64x(mask_size + 8, mask_size, 8, 0);
    // t0, _, t2, _, t4, _ t6, _
    const __m256i t = _mm256_sllv_epi64(s, shift);

    // t2, _, t0, _, t6, _, t4, _
    const __m256i shuffled2 = _mm256_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2));
    // t0+t2, _, _, _, t4+t6, _, _, _
    const __m256i u = _mm256_add_epi64(shuffled2, t);

    const __m128i hi = _mm256_extractf128_si256(u, 1);
    const __m128i lo = _mm256_extractf128_si256(u, 0);

    /*uint32_t tmp = 0;
    for (int i = 0; i < 4; ++i) {
      tmp += static_cast<uint32_t>(shuffled2.m256i_u64[i]);
    }*/
    // それぞれから下位32bitを取り出す
    scale_buffer[kScaleIndex] = _mm_cvtsi128_si32(lo) + _mm_cvtsi128_si32(hi);
    //scale_buffer[kScaleIndex] = tmp;
  }

 private:
  // 差分計算を用いずに累積値を計算する
  void RefreshAccumulator(const Position& pos) const {
    auto& accumulator = pos.state()->accumulator;
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList active_indices[2];
      RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i],
                                       active_indices);
      for (const auto perspective : COLOR) {
        if (i == 0) {
          std::memcpy(accumulator.accumulation[perspective][i], biases_,
                      kHalfDimensions * sizeof(BiasType));
        } else {
          std::memset(accumulator.accumulation[perspective][i], 0,
                      kHalfDimensions * sizeof(BiasType));
        }
        for (const auto index : active_indices[perspective]) {
          const IndexType offset = kHalfDimensions * index;
#if defined(USE_AVX2)
          auto accumulation = reinterpret_cast<__m256i*>(
              &accumulator.accumulation[perspective][i][0]);
          auto column = reinterpret_cast<const __m256i*>(&weights_[offset]);
          constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
          for (IndexType j = 0; j < kNumChunks; ++j) {
            accumulation[j] = _mm256_add_epi16(accumulation[j], column[j]);
          }
#elif defined(USE_SSE2)
          auto accumulation = reinterpret_cast<__m128i*>(
              &accumulator.accumulation[perspective][i][0]);
          auto column = reinterpret_cast<const __m128i*>(&weights_[offset]);
          constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
          for (IndexType j = 0; j < kNumChunks; ++j) {
            accumulation[j] = _mm_add_epi16(accumulation[j], column[j]);
          }
#elif defined(IS_ARM)
          auto accumulation = reinterpret_cast<int16x8_t*>(
              &accumulator.accumulation[perspective][i][0]);
          auto column = reinterpret_cast<const int16x8_t*>(&weights_[offset]);
          constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
          for (IndexType j = 0; j < kNumChunks; ++j) {
            accumulation[j] = vaddq_s16(accumulation[j], column[j]);
          }
#else
          for (IndexType j = 0; j < kHalfDimensions; ++j) {
            accumulator.accumulation[perspective][i][j] += weights_[offset + j];
          }
#endif
        }
      }
    }

    accumulator.computed_accumulation = true;
    accumulator.computed_score = false;
  }

  // 差分計算を用いて累積値を計算する
  void UpdateAccumulator(const Position& pos) const {
    const auto prev_accumulator = pos.state()->previous->accumulator;
    auto& accumulator = pos.state()->accumulator;
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList removed_indices[2], added_indices[2];
      bool reset[2];
      RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i],
                                        removed_indices, added_indices, reset);
      for (const auto perspective : COLOR) {
#if defined(USE_AVX2)
        constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
        auto accumulation = reinterpret_cast<__m256i*>(
            &accumulator.accumulation[perspective][i][0]);
#elif defined(USE_SSE2)
        constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
        auto accumulation = reinterpret_cast<__m128i*>(
            &accumulator.accumulation[perspective][i][0]);
#elif defined(IS_ARM)
        constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
        auto accumulation = reinterpret_cast<int16x8_t*>(
            &accumulator.accumulation[perspective][i][0]);
#endif
        if (reset[perspective]) {
          if (i == 0) {
            std::memcpy(accumulator.accumulation[perspective][i], biases_,
                        kHalfDimensions * sizeof(BiasType));
          } else {
            std::memset(accumulator.accumulation[perspective][i], 0,
                        kHalfDimensions * sizeof(BiasType));
          }
        } else {  // 1から0に変化した特徴量に関する差分計算
          std::memcpy(accumulator.accumulation[perspective][i],
                      prev_accumulator.accumulation[perspective][i],
                      kHalfDimensions * sizeof(BiasType));
          for (const auto index : removed_indices[perspective]) {
            const IndexType offset = kHalfDimensions * index;
#if defined(USE_AVX2)
            auto column = reinterpret_cast<const __m256i*>(&weights_[offset]);
            for (IndexType j = 0; j < kNumChunks; ++j) {
              accumulation[j] = _mm256_sub_epi16(accumulation[j], column[j]);
            }
#elif defined(USE_SSE2)
            auto column = reinterpret_cast<const __m128i*>(&weights_[offset]);
            for (IndexType j = 0; j < kNumChunks; ++j) {
              accumulation[j] = _mm_sub_epi16(accumulation[j], column[j]);
            }
#elif defined(IS_ARM)
            auto column = reinterpret_cast<const int16x8_t*>(&weights_[offset]);
            for (IndexType j = 0; j < kNumChunks; ++j) {
              accumulation[j] = vsubq_s16(accumulation[j], column[j]);
            }
#else
            for (IndexType j = 0; j < kHalfDimensions; ++j) {
              accumulator.accumulation[perspective][i][j] -=
                  weights_[offset + j];
            }
#endif
          }
        }
        {  // 0から1に変化した特徴量に関する差分計算
          for (const auto index : added_indices[perspective]) {
            const IndexType offset = kHalfDimensions * index;
#if defined(USE_AVX2)
            auto column = reinterpret_cast<const __m256i*>(&weights_[offset]);
            for (IndexType j = 0; j < kNumChunks; ++j) {
              accumulation[j] = _mm256_add_epi16(accumulation[j], column[j]);
            }
#elif defined(USE_SSE2)
            auto column = reinterpret_cast<const __m128i*>(&weights_[offset]);
            for (IndexType j = 0; j < kNumChunks; ++j) {
              accumulation[j] = _mm_add_epi16(accumulation[j], column[j]);
            }
#elif defined(IS_ARM)
            auto column = reinterpret_cast<const int16x8_t*>(&weights_[offset]);
            for (IndexType j = 0; j < kNumChunks; ++j) {
              accumulation[j] = vaddq_s16(accumulation[j], column[j]);
            }
#else
            for (IndexType j = 0; j < kHalfDimensions; ++j) {
              accumulator.accumulation[perspective][i][j] +=
                  weights_[offset + j];
            }
#endif
          }
        }
      }
    }

    accumulator.computed_accumulation = true;
    accumulator.computed_score = false;
  }

  // パラメータの型
  using BiasType = std::int16_t;
  using WeightType = std::int16_t;
  //using ScaleType = int16_t;

  // 学習用クラスをfriendにする
  friend class Trainer<FeatureTransformer>;

  // パラメータ
  alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
  alignas(kCacheLineSize)
      WeightType weights_[kHalfDimensions * kInputDimensions];
};

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
