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

  auto& w = net->feature_transformer->embedding->weight;
  auto weights = torch::from_blob(feature_transformer->weights_,
                                  {FeatureTransformer::kInputDimensions,
                                   FeatureTransformer::kHalfDimensions},
                                  torch::TensorOptions(torch::kInt16));
  w = weights.to(torch::kFloat32) / scale;

  auto& b = net->feature_transformer->bias;
  auto biases = torch::from_blob(feature_transformer->biases_,
                                 FeatureTransformer::kHalfDimensions,
                                 torch::TensorOptions(torch::kInt16));
  b = biases.to(torch::kFloat32) / scale;
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
  auto& fc1 =
      network->previous_layer_.previous_layer_.previous_layer_.previous_layer_;
  static constexpr auto scale = 64.0;

  auto& w = net->affine1->weight;
  auto weights = torch::from_blob(fc1.weights_,
                                  {fc1.kInputDimensions, fc1.kOutputDimensions},
                                  torch::TensorOptions(torch::kInt8));
  w = weights.transpose(1, 0).to(torch::kFloat32) / scale;

  auto& b = net->affine1->bias;
  auto biasses = torch::from_blob(fc1.biases_, fc1.kOutputDimensions,
                                  torch::TensorOptions(torch::kInt32));
  b = biasses.to(torch::kFloat32) / (scale * 127);
}

void TorchTrainer::QuantizeAffine2() {
  // “ñ‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc2 = network->previous_layer_.previous_layer_;
  static constexpr auto scale = 64.0;

  auto& weights = fc2.weights_;
  const auto& w = net->affine2->weight;
  for (int i = 0; i < fc2.kInputDimensions; ++i) {
    for (int j = 0; j < fc2.kOutputDimensions; ++j) {
      weights[j * fc2.kInputDimensions + i] = Round<int8_t>(
          std::clamp(w[i][j].item<float>() * scale, -128.0, 127.0));
    }
  }

  auto& biases = fc2.biases_;
  const auto& b = net->affine2->bias;
  for (int i = 0; i < fc2.kOutputDimensions; ++i) {
    biases[i] = Round<int32_t>(
        std::clamp(b[i].item<float>() * scale * 127.0,
                   static_cast<double>(std::numeric_limits<int32_t>::min()),
                   static_cast<double>(std::numeric_limits<int32_t>::max())));
  }
}
void TorchTrainer::DequantizeAffine2() {
  // “ñ‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc2 = network->previous_layer_.previous_layer_;
  static constexpr auto scale = 64.0;

  auto& w = net->affine2->weight;
  auto weights = torch::from_blob(fc2.weights_,
                                  {fc2.kInputDimensions, fc2.kOutputDimensions},
                                  torch::TensorOptions(torch::kInt8));
  w = weights.transpose(1, 0).to(torch::kFloat32) / scale;

  auto& b = net->affine2->bias;
  auto biasses = torch::from_blob(fc2.biases_, fc2.kOutputDimensions,
                                  torch::TensorOptions(torch::kInt32));
  b = biasses.to(torch::kFloat32) / (scale * 127);
}
void TorchTrainer::DequantizeAffine2() {
  // “ñ‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc2 = network->previous_layer_.previous_layer_;
  static constexpr auto scale = 64.0;

void TorchTrainer::QuantizeAffine3() {
  // ŽO‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc3 = *network;
  static constexpr auto scale = 600.0 * 16.0 / 127.0;

  auto& weights = fc3.weights_;
  const auto& w = net->affine3->weight;
  for (int i = 0; i < fc3.kInputDimensions; ++i) {
    for (int j = 0; j < fc3.kOutputDimensions; ++j) {
      weights[j * fc3.kInputDimensions + i] = Round<int8_t>(
          std::clamp(w[i][j].item<float>() * scale, -128.0, 127.0));
    }
  }

  auto& biases = fc3.biases_;
  const auto& b = net->affine3->bias;
  for (int i = 0; i < fc3.kOutputDimensions; ++i) {
    biases[i] = Round<int32_t>(
        std::clamp(b[i].item<float>() * scale * 127.0,
                   static_cast<double>(std::numeric_limits<int32_t>::min()),
                   static_cast<double>(std::numeric_limits<int32_t>::max())));
  }
}
void TorchTrainer::DequantizeAffine3() {
  // ŽO‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc3 = *network;
  static constexpr auto scale = 600.0 * 16.0 / 127.0;

  auto& w = net->affine3->weight;
  auto weights = torch::from_blob(fc3.weights_,
                                  {fc3.kInputDimensions, fc3.kOutputDimensions},
                                  torch::TensorOptions(torch::kInt8));
  w = weights.transpose(1, 0).to(torch::kFloat32) / scale;

  auto& b = net->affine3->bias;
  auto biasses = torch::from_blob(fc3.biases_, fc3.kOutputDimensions,
                                  torch::TensorOptions(torch::kInt32));
  b = biasses.to(torch::kFloat32) / (scale * 127);
}

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)
