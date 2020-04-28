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
// •]‰¿ŠÖ”‚Å—p‚¢‚é“ü—Í“Á’¥—Ê
using RawFeatures = Features::FeatureSet<
  Features::HalfKP<Features::Side::kFriend>>;

// •ÏŠ·Œã‚Ì“ü—Í“Á’¥—Ê‚ÌŸŒ³”
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
