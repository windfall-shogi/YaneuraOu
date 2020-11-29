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
  static constexpr float scale = 127.0f;

  const auto& w = net->feature_transformer->embedding->weight;
  auto tmp_w = w * scale;
  tmp_w.round_();
  tmp_w = tmp_w.to(torch::kInt16);
  auto& weights = feature_transformer->weights_;
  std::copy_n(tmp_w.data_ptr<int16_t>(),
              FeatureTransformer::kInputDimensions *
                  FeatureTransformer::kHalfDimensions,
              weights);

  const auto& b = net->feature_transformer->bias;
  auto tmp_b = b * scale;
  tmp_b.round_();
  tmp_b = tmp_b.to(torch::kInt16);
  auto& biases = feature_transformer->biases_;
  std::copy_n(tmp_b.data_ptr<int16_t>(), FeatureTransformer::kHalfDimensions,
              biases);
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

  const auto& w = net->affine1->weight;
  auto tmp_w = w * scale;
  tmp_w.clamp_(-128.0f, 127.0f);
  tmp_w.round_();
  tmp_w = tmp_w.to(torch::kInt8);
  auto& weights = fc1.weights_;
  std::copy_n(tmp_w.data_ptr<int8_t>(),
              fc1.kInputDimensions * fc1.kOutputDimensions, weights);

  const auto& b = net->affine1->bias;
  auto tmp_b = b * scale * 127.0f;
  tmp_b.clamp_(std::numeric_limits<int32_t>::min(),
               std::numeric_limits<int32_t>::max());
  tmp_b.round_();
  tmp_b = tmp_b.to(torch::kInt32);
  auto& biases = fc1.biases_;
  std::copy_n(tmp_b.data_ptr<int32_t>(), fc1.kOutputDimensions, biases);
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
}

void TorchTrainer::QuantizeAffine2() {
  // “ñ‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc2 = network->previous_layer_.previous_layer_;
  static constexpr float scale = 64.0f;

  const auto& w = net->affine2->weight;
  auto tmp_w = w * scale;
  tmp_w.clamp_(-128.0f, 127.0f);
  tmp_w.round_();
  tmp_w = tmp_w.to(torch::kInt8);
  auto& weights = fc2.weights_;
  std::copy_n(tmp_w.data_ptr<int8_t>(),
              fc2.kInputDimensions * fc2.kOutputDimensions, weights);

  const auto& b = net->affine2->bias;
  auto tmp_b = b * scale * 127.0f;
  tmp_b.clamp_(std::numeric_limits<int32_t>::min(),
               std::numeric_limits<int32_t>::max());
  tmp_b.round_();
  tmp_b = tmp_b.to(torch::kInt32);
  auto& biases = fc2.biases_;
  std::copy_n(tmp_b.data_ptr<int32_t>(), fc2.kOutputDimensions, biases);
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
  static constexpr float scale = 600.0f * 16.0f / 127.0f;

  const auto& w = net->affine3->weight;
  auto tmp_w = w * scale;
  tmp_w.clamp_(-128.0f, 127.0f);
  tmp_w.round_();
  tmp_w = tmp_w.to(torch::kInt8);
  auto& weights = fc3.weights_;
  std::copy_n(tmp_w.data_ptr<int8_t>(),
              fc3.kInputDimensions * fc3.kOutputDimensions, weights);

  const auto& b = net->affine3->bias;
  auto tmp_b = b * scale * 127.0f;
  tmp_b.clamp_(std::numeric_limits<int32_t>::min(),
               std::numeric_limits<int32_t>::max());
  tmp_b.round_();
  tmp_b = tmp_b.to(torch::kInt32);
  auto& biases = fc3.biases_;
  std::copy_n(tmp_b.data_ptr<int32_t>(), fc3.kOutputDimensions, biases);
}
void TorchTrainer::DequantizeAffine3() {
  // ŽO‚Â–Ú‚Ì‘SŒ‹‡‘w
  auto& fc3 = *network;
  static constexpr float scale = 600.0f * 16.0f / 127.0f;

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
