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

  auto weights = torch::from_blob(feature_transformer->weights_,
                                  {FeatureTransformer::kInputDimensions,
                                   FeatureTransformer::kHalfDimensions},
                                  torch::TensorOptions(torch::kInt16));
  weights = weights.to(torch::kFloat32) / scale;
  auto& w = net->feature_transformer->embedding->weight;
  std::copy_n(weights.data_ptr<float>(),
              FeatureTransformer::kInputDimensions *
                  FeatureTransformer::kHalfDimensions,
              w.data_ptr<float>());

  auto biases = torch::from_blob(feature_transformer->biases_,
                                 FeatureTransformer::kHalfDimensions,
                                 torch::TensorOptions(torch::kInt16));
  biases = biases.to(torch::kFloat32) / scale;
  auto& b = net->feature_transformer->bias;
  std::copy_n(biases.data_ptr<float>(), FeatureTransformer::kHalfDimensions,
              b.data_ptr<float>());
}

void TorchTrainer::QuantizeAffine1() {
  // ˆê‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc1 =
      network->previous_layer_.previous_layer_.previous_layer_.previous_layer_;
  static constexpr float scale = 64.0f;

  auto& weights = fc1.weights_;
  const auto& w = net->affine1->weight;
  for (int i = 0; i < fc1.kOutputDimensions; ++i) {
    for (int j = 0; j < fc1.kInputDimensions; ++j) {
      weights[i * fc1.kInputDimensions + j] = Round<int8_t>(
          std::clamp(w[i][j].item<float>() * scale, -128.0f, 127.0f));
    }
  }

  auto& biases = fc1.biases_;
  const auto& b = net->affine1->bias;
  std::cout << (b * scale * 127.0f) << std::endl;
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
  static constexpr float scale = 64.0f;

  auto weights = torch::from_blob(fc1.weights_,
                                  {fc1.kOutputDimensions, fc1.kInputDimensions},
                                  torch::TensorOptions(torch::kInt8));
  weights = weights.to(torch::kFloat32) / scale;
  auto& w = net->affine1->weight;
  std::copy_n(weights.data_ptr<float>(),
              fc1.kOutputDimensions * fc1.kInputDimensions,
              w.data_ptr<float>());

  auto biases = torch::from_blob(fc1.biases_, fc1.kOutputDimensions,
                                 torch::TensorOptions(torch::kInt32));
  biases = biases.to(torch::kFloat32) / (scale * 127);
  auto& b = net->affine1->bias;
  std::copy_n(biases.data_ptr<float>(), fc1.kOutputDimensions,
              b.data_ptr<float>());
}

void TorchTrainer::QuantizeAffine2() {
  // “ñ‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc2 = network->previous_layer_.previous_layer_;
  static constexpr float scale = 64.0f;

  auto& weights = fc2.weights_;
  const auto& w = net->affine2->weight;
  for (int i = 0; i < fc2.kOutputDimensions; ++i) {
    for (int j = 0; j < fc2.kInputDimensions; ++j) {
      weights[i * fc2.kInputDimensions + j] = Round<int8_t>(
          std::clamp(w[i][j].item<float>() * scale, -128.0f, 127.0f));
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
  static constexpr float scale = 64.0f;

  auto weights = torch::from_blob(fc2.weights_,
                                  {fc2.kOutputDimensions, fc2.kInputDimensions},
                                  torch::TensorOptions(torch::kInt8));
  weights = weights.to(torch::kFloat32) / scale;
  auto& w = net->affine2->weight;
  std::copy_n(weights.data_ptr<float>(),
              fc2.kOutputDimensions * fc2.kInputDimensions,
              w.data_ptr<float>());

  auto biases = torch::from_blob(fc2.biases_, fc2.kOutputDimensions,
                                 torch::TensorOptions(torch::kInt32));
  biases = biases.to(torch::kFloat32) / (scale * 127);
  auto& b = net->affine2->bias;
  std::copy_n(biases.data_ptr<float>(), fc2.kOutputDimensions,
              b.data_ptr<float>());
}
void TorchTrainer::DequantizeAffine2() {
  // “ñ‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc2 = network->previous_layer_.previous_layer_;
  static constexpr auto scale = 64.0;

void TorchTrainer::QuantizeAffine3() {
  // ŽO‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc3 = *network;
  static constexpr float scale = 600.0 * 16.0 / 127.0;

  auto& weights = fc3.weights_;
  const auto& w = net->affine3->weight;
  for (int i = 0; i < fc3.kOutputDimensions; ++i) {
    for (int j = 0; j < fc3.kInputDimensions; ++j) {
      weights[i * fc3.kInputDimensions + j] = Round<int8_t>(
          std::clamp(w[i][j].item<float>() * scale, -128.0f, 127.0f));
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
  static constexpr float scale = 600.0 * 16.0 / 127.0;

  auto weights = torch::from_blob(fc3.weights_,
                                  {fc3.kOutputDimensions, fc3.kInputDimensions},
                                  torch::TensorOptions(torch::kInt8));
  weights = weights.to(torch::kFloat32) / scale;
  auto& w = net->affine3->weight;
  std::copy_n(weights.data_ptr<float>(),
              fc3.kOutputDimensions * fc3.kInputDimensions,
              w.data_ptr<float>());

  auto biases = torch::from_blob(fc3.biases_, fc3.kOutputDimensions,
                                 torch::TensorOptions(torch::kInt32));
  biases = biases.to(torch::kFloat32) / (scale * 127);
  auto& b = net->affine3->bias;
  std::copy_n(biases.data_ptr<float>(), fc3.kOutputDimensions,
              b.data_ptr<float>());
}

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)
