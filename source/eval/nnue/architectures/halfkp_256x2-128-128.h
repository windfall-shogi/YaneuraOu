#pragma once

#include "../features/feature_set.h"
#include "../features/half_kp.h"

#include "../layers/input_slice.h"
#include "../layers/binarization.h"
#include "../layers/binary_affine_transform.h"
#include "../layers/binary_inner_product.h"
#include "../layers/leaky_relu.h"

namespace Eval {

namespace NNUE {
// 評価関数で用いる入力特徴量
using RawFeatures = Features::FeatureSet<
  Features::HalfKP<Features::Side::kFriend>>;

// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 256;

namespace Layers {

using InputLayer = InputSlice<kTransformedFeatureDimensions * 2>;
using HiddenLayer1 = Binarization<BinaryAffineTransform<InputLayer, 128>>;
using HiddenLayer2 = LeakyReLU<BinaryAffineTransform<HiddenLayer1, 128>>;
using OutputLayer = BinaryInnerProduct<HiddenLayer2>;

}  // namespace Layers

using Network = Layers::OutputLayer;

}  // namespace NNUE

}  // namespace Eval
