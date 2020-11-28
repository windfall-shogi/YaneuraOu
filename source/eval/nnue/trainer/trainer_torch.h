#pragma once
#ifndef _NNUE_TRAINER_TORCH_H_
#define _NNUE_TRAINER_TORCH_H_

#include "../../../config.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)
#include "../evaluate_nnue_torch_model.h"

namespace Eval {

namespace NNUE {
/**
 * @brief Torchのモデルと評価関数のモデルとの間でパラメータをやり取りする
*/
class TorchTrainer {
public:
  void Quantize();
  void Dequantize();

  torch::Tensor forward(torch::Tensor p, torch::Tensor q) {
    return net->forward(p, q);
  }

private:
  void QuantizeFeatureTransformer();
  void DequantizeFeatureTransformer();

  void QuantizeAffine1();
  void DequantizeAffine1();

  void QuantizeAffine2();
  void DequantizeAffine2();

  void QuantizeAffine3();
  void DequantizeAffine3();

  Net net;
};

}  // namespace NNUE

}  // namespace Eval

#endif // defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)

#endif // _NNUE_TRAINER_TORCH_H_
