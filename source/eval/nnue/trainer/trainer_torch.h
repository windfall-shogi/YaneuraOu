#pragma once
#ifndef _NNUE_TRAINER_TORCH_H_
#define _NNUE_TRAINER_TORCH_H_

#include "../../../config.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)
#include "../evaluate_nnue_torch_model.h"
#include "../../../misc.h"

namespace Eval {

namespace NNUE {
/**
 * @brief Torchのモデルと評価関数のモデルとの間でパラメータをやり取りする
*/
class TorchTrainer {
public:
  TorchTrainer()
      : net(), optimizer(net->parameters(), torch::optim::SGDOptions(0.01)) {}

  void Quantize() {
    QuantizeFeatureTransformer();
    QuantizeAffine1();
    QuantizeAffine2();
    QuantizeAffine3();
  }
  void Dequantize() {
    DequantizeFeatureTransformer();
    DequantizeAffine1();
    DequantizeAffine2();
    DequantizeAffine3();
  }

  torch::Tensor forward(torch::Tensor p, torch::Tensor q) {
    return net->forward(p, q);
  }

  void SetLearningRate(const double lr) {
    // 学習係数の変更方法
    // https://stackoverflow.com/questions/62415285/updating-learning-rate-with-libtorch-1-5-and-optimiser-options-in-c
    for (auto param_group : optimizer.param_groups()) {
      // Static cast needed as options() returns OptimizerOptions(base class)
      static_cast<torch::optim::SGDOptions&>(param_group.options()).lr(lr);
    }
  }

  void Save(const std::string& dir_name) {
    const auto model_path = Path::Combine(dir_name, "model.pt");
    torch::save(net, model_path);

    const auto optimizer_path = Path::Combine(dir_name, "optimizer.pt");
    torch::save(optimizer, optimizer_path);
  }

  void Load(const std::string& dir_name) {
    const auto model_path = Path::Combine(dir_name, "model.pt");
    torch::load(net, model_path);

    const auto optimizer_path = Path::Combine(dir_name, "optimizer.pt");
    torch::load(optimizer, optimizer_path);
  }

  Net net;
  torch::optim::SGD optimizer;

private:
  void QuantizeFeatureTransformer();
  void DequantizeFeatureTransformer();

  void QuantizeAffine1();
  void DequantizeAffine1();

  void QuantizeAffine2();
  void DequantizeAffine2();

  void QuantizeAffine3();
  void DequantizeAffine3();
};

}  // namespace NNUE

}  // namespace Eval

#endif // defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)

#endif // _NNUE_TRAINER_TORCH_H_
