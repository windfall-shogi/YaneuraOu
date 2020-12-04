#pragma once
#ifndef _EVALUATE_NNUE_TORCH_MODEL_H_
#define _EVALUATE_NNUE_TORCH_MODEL_H_

#include "../../config.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)
#pragma warning(push)
#pragma warning(disable : 4101 4244 4251 4267 4275 4819 4996 26110 26812 26819 26439 26444 26451 26478 26495 26817)
#define _SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#include <torch/torch.h>
#pragma warning(pop)

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
                                                  .sparse(false)))),
        bias(register_parameter(
            "bias", torch::ones(kTransformedFeatureDimensions) * 0.5)) {}

  torch::Tensor forward(torch::Tensor input) {
    auto h = embedding(input);
    h = torch::sum(h, 1);
    return h + bias;
  }

  torch::nn::Embedding embedding;
  torch::Tensor bias;
};
// shared_ptrみたいに良い感じにしてくれる
TORCH_MODULE(Embedding);

struct ClippedReLUImpl : torch::nn::Module {
  ClippedReLUImpl() = default;

  torch::Tensor forward(torch::Tensor input) {
    // 上側をクリップ
    auto h = 1 - torch::relu(1 - input);
    // 下側をクリップ
    return h.relu_();
  }
};
// shared_ptrみたいに良い感じにしてくれる
TORCH_MODULE(ClippedReLU);


struct NetImpl : torch::nn::Module {
  NetImpl()
      : feature_transformer(
            register_module("feature_transformer", Embedding())),
        affine1(register_module("affine1", torch::nn::Linear(512, 32))),
        affine2(register_module("affine2", torch::nn::Linear(32, 32))),
        affine3(register_module("affine3", torch::nn::Linear(32, 1))) {
    std::fill_n(affine1->bias.data_ptr<float>(), 32, 0.5f);
    std::fill_n(affine2->bias.data_ptr<float>(), 32, 0.5f);
  }

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
// shared_ptrみたいに良い感じにしてくれる
TORCH_MODULE(Net);
}  // namespace NNUE

}  // namespace Eval
#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE) && defined(USE_LIBTORCH)

#endif

