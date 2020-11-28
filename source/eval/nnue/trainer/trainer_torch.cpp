#include "trainer_torch.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)
#include <climits>

#include "../evaluate_nnue.h"
#include "../evaluate_nnue_torch_model.h"
#include "../nnue_architecture.h"
#include "trainer.h"

namespace Eval {

namespace NNUE {
void TorchTrainer::QuantizeFeatureTransformer() {
  static constexpr auto scale = 127.0;

  auto& weights = feature_transformer->weights_;
  const auto& w = net->feature_transformer->embedding->weight;
  for (int i = 0; i < FeatureTransformer::kInputDimensions; ++i) {
    const auto offset = i * FeatureTransformer::kHalfDimensions;
    for (int j = 0; j < FeatureTransformer::kHalfDimensions; ++j) {
      weights[offset + j] = Round<int16_t>(w[i][j].item<float>() * scale);
    }
  }

  auto& biases = feature_transformer->biases_;
  const auto& b = net->feature_transformer->bias;
  for (int i = 0; i < FeatureTransformer::kHalfDimensions; ++i) {
    biases[i] = Round<int16_t>(b[i].item<float>() * scale);
  }
}
void TorchTrainer::DequantizeFeatureTransformer() {
  static constexpr auto scale = 127.0f;

  const auto& weights = feature_transformer->weights_;
  auto& w = net->feature_transformer->embedding->weight;
  for (int i = 0; i < FeatureTransformer::kInputDimensions; ++i) {
    const auto offset = i * FeatureTransformer::kHalfDimensions;
    for (int j = 0; j < FeatureTransformer::kHalfDimensions; ++j) {
      w[i][j] = weights[offset + j] / scale;
    }
  }

  const auto& biases = feature_transformer->biases_;
  auto& b = net->feature_transformer->bias;
  for (int i = 0; i < FeatureTransformer::kHalfDimensions; ++i) {
    b[i] = biases[i] / scale;
  }
}

void TorchTrainer::QuantizeAffine1() {
  // ˆê‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc1 =
      network->previous_layer_.previous_layer_.previous_layer_.previous_layer_;
  static constexpr auto scale = 64.0;

  auto& weights = fc1.weights_;
  const auto& w = net->affine1->weight;
  for (int i = 0; i < fc1.kInputDimensions; ++i) {
    for (int j = 0; j < fc1.kOutputDimensions; ++j) {
      weights[j * fc1.kInputDimensions + i] = Round<int8_t>(
          std::clamp(w[i][j].item<float>() * scale, -128.0, 127.0));
    }
  }

  auto& biases = fc1.biases_;
  const auto& b = net->affine1->bias;
  for (int i = 0; i < fc1.kOutputDimensions; ++i) {
    biases[i] = Round<int32_t>(
        std::clamp(b[i].item<float>() * scale * 127.0,
                   static_cast<double>(std::numeric_limits<int32_t>::min()),
                   static_cast<double>(std::numeric_limits<int32_t>::max())));
  }
}
void TorchTrainer::DequantizeAffine1() {
  // ˆê‚Â–Ú‚Ì‘SŒ‹‡‘w
  const auto& fc1 =
    network->previous_layer_.previous_layer_.previous_layer_.previous_layer_;
  static constexpr auto scale = 64.0;

  const auto& weights = fc1.weights_;
  auto& w = net->affine1->weight;
  for (int i = 0; i < fc1.kInputDimensions; ++i) {
    for (int j = 0; j < fc1.kOutputDimensions; ++j) {
      w[i][j] = weights[j * fc1.kInputDimensions + i] / scale;
    }
  }

  const auto& biases = fc1.biases_;
  auto& b = net->affine1->bias;
  for (int i = 0; i < fc1.kOutputDimensions; ++i) {
    b[i] = biases[i] / (scale * 127.0);
  }

}

void TorchTrainer::QuantizeAffine2() {}
void TorchTrainer::DequantizeAffine2() {}

void TorchTrainer::QuantizeAffine3() {}
void TorchTrainer::DequantizeAffine3() {}

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)
