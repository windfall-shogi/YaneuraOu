// NNUE評価関数で用いる定数など

#ifndef _NNUE_COMMON_H_
#define _NNUE_COMMON_H_

#include "../../config.h"

#if defined(EVAL_NNUE)

#include <cstdint>
#include <cstddef>

#if defined(USE_AVX512)
// immintrin.h から AVX512 関連の intrinsic は読み込まれる
// intel: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX_512
// gcc: https://github.com/gcc-mirror/gcc/blob/master/gcc/config/i386/immintrin.h
// clang: https://github.com/llvm-mirror/clang/blob/master/lib/Headers/immintrin.h
#include <immintrin.h>
#elif defined(USE_AVX2)
#include <immintrin.h>
#elif defined(USE_SSE42)
#include <nmmintrin.h>
#elif defined(USE_SSE41)
#include <smmintrin.h>
#elif defined(USE_SSE2)
#include <emmintrin.h>
#elif defined(IS_ARM)
#include <arm_neon.h>
#include <mm_malloc.h> // for _mm_alloc()
#else
#if defined (__GNUC__)
#include <mm_malloc.h> // for _mm_alloc()
#endif
#endif


namespace Eval {

namespace NNUE {

// 評価関数ファイルのバージョンを表す定数
constexpr std::uint32_t kVersion = 0x7AF32F16u;

// 評価値の計算で利用する定数
constexpr int FV_SCALE = 16;
constexpr int kWeightScaleBits = 6;

//! embedding層でのパラメータの倍率
constexpr int kFeatureScaleBits = 11;

// キャッシュラインのサイズ（バイト単位）
constexpr std::size_t kCacheLineSize = 64;

// SIMD幅（バイト単位）
#if defined(USE_AVX2)
constexpr std::size_t kSimdWidth = 32;
#elif defined(USE_SSE2)
constexpr std::size_t kSimdWidth = 16;
#elif defined(IS_ARM)
constexpr std::size_t kSimdWidth = 16;
#endif
constexpr std::size_t kMaxSimdWidth = 32;

// 変換後の入力特徴量の型
using TransformedFeatureType = std::uint8_t;

// インデックスの型
using IndexType = std::uint32_t;

// 学習用クラステンプレートの前方宣言
template <typename Layer>
class Trainer;

// n以上で最小のbaseの倍数を求める
template <typename IntType>
constexpr IntType CeilToMultiple(IntType n, IntType base) {
  return (n + base - 1) / base * base;
}

//! Nが2の何乗かを計算
template<IndexType N>
struct Log2;
template <>
struct Log2<64> {
  static constexpr IndexType value = 6;
};
template <>
struct Log2<128> {
  static constexpr IndexType value = 7;
};
template <>
struct Log2<256> {
  static constexpr IndexType value = 8;
};
template <>
struct Log2<512> {
  static constexpr IndexType value = 9;
};

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
