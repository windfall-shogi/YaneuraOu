#pragma once
#ifndef _EVALUATE_NNUE_TORCH_MODEL_H_
#define _EVALUATE_NNUE_TORCH_MODEL_H_

#include "../../config.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)
#include <torch/torch.h>

#include "nnue_common.h"
#include "nnue_architecture.h"

#include "../../learn/learn.h"

namespace Eval {

namespace NNUE {
struct EmbeddingImpl : torch::nn::Module {
  EmbeddingImpl()
      : embedding(register_module(
            "embedding", torch::nn::Embedding(torch::nn::EmbeddingOptions(
                                                  RawFeatures::kDimensions,
                                                  kTransformedFeatureDimensions)
                                                  .sparse(true)))),
        bias(register_parameter(
            "bias", torch::ones(kTransformedFeatureDimensions) * 0.5)) {}

  torch::Tensor forward(torch::Tensor input) {
    const auto h = embedding(input).sum(1);
    return h + bias;
  }

  torch::nn::Embedding embedding;
  torch::Tensor bias;
};
// shared_ptr�݂����ɗǂ������ɂ��Ă����
TORCH_MODULE(Embedding);

struct ClippedReLUImpl : torch::nn::Module {
  ClippedReLUImpl() = default;

  torch::Tensor forward(torch::Tensor input) {
    // �㑤���N���b�v
    auto h = 1 - torch::relu(1 - input);
    // �������N���b�v
    return h.relu_();
  }
};
// shared_ptr�݂����ɗǂ������ɂ��Ă����
TORCH_MODULE(ClippedReLU);


struct NetImpl : torch::nn::Module {
  NetImpl()
      : feature_transformer(
            register_module("feature_transformer", Embedding())),
        affine1(register_module("affine1", torch::nn::Linear(512, 32))),
        affine2(register_module("affine2", torch::nn::Linear(32, 32))),
        affine3(register_module("affine3", torch::nn::Linear(32, 1))) {}

  torch::Tensor forward(torch::Tensor input_p, torch::Tensor input_q) {
    const auto p = feature_transformer(input_p);
    const auto q = feature_transformer(input_q);
    auto h = torch::cat({ p, q }, 1);

    h = relu(h);
    h = affine1(h);

    h = relu(h);
    h = affine2(h);

    h = relu(h);
    h = affine3(h);

    return h;
  }

  Embedding feature_transformer;
  torch::nn::Linear affine1, affine2, affine3;
  ClippedReLU relu;

};
// shared_ptr�݂����ɗǂ������ɂ��Ă����
TORCH_MODULE(Net);
}  // namespace NNUE

}  // namespace Eval
#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)

#endif

