#pragma once
#ifndef _NNUE_TRAINER_BINARIZATION_H_
#define _NNUE_TRAINER_BINARIZATION_H_

#include "../../../config.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE)

#include "../../../learn/learn.h"
#include "../layers/binarization.h"
#include "../nnue_feature_transformer.h"
#include "trainer.h"

#include <random>
#include <memory>

namespace Eval {

namespace NNUE {

template <typename PreviousLayer>
class Trainer<Layers::Binarization<PreviousLayer>> {
private:
  using LayerType = Layers::Binarization<PreviousLayer>;

public:
  static std::shared_ptr<Trainer> Create(
    LayerType* target_layer, FeatureTransformer* feature_transformer) {
    // std::make_shared から private コンストラクタを呼び出す
    // https://gintenlabo.hatenablog.com/entry/20131211/1386771626
    struct impl :Trainer<LayerType> {
      impl(LayerType* t, FeatureTransformer* f) :Trainer<LayerType>(t, f) {}
    };
    return std::move(
      std::make_shared<impl>(target_layer, feature_transformer));
  }

  void SendMessage(Message* message) {}

  template<typename RNG>
  void Initialize(RNG& rng) {
    previous_layer_trainer_->Initialize(rng);
    QuantizeParameters();
  }

  const LearnFloatType* Propagate(const std::vector<Example>& batch) {
    return output_.data();
  }

  void Backpropagate(const LearnFloatType* gradients,
    LearnFloatType learning_rate) {}

private:
  Trainer(LayerType* target_layer, FeatureTransformer* feature_transformer)
    : previous_layer_trainer_(Trainer<PreviousLayer>::Create(
      &target_layer->previous_layer_, feature_transformer)),
    target_layer_(target_layer) {}
  void QuantizeParameters() {}
  void DequantizeParameters() {}

  // 入出力の次元数
  static constexpr IndexType kInputDimensions = LayerType::kInputDimensions;
  static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;

  // 出力の次元数が1なら出力層
  static constexpr bool kIsOutputLayer = kOutputDimensions == 1;

  // ミニバッチのサンプル数
  IndexType batch_size_;

  // ミニバッチの入力
  const LearnFloatType* batch_input_;

  // 直前の層のTrainer
  const std::shared_ptr<Trainer<PreviousLayer>> previous_layer_trainer_;

  // 学習対象の層
  LayerType* const target_layer_;

  // 順伝播用バッファ
  std::vector<LearnFloatType> output_;

  // 逆伝播用バッファ
  std::vector<LearnFloatType> gradients_;
};

}  // namespace NNUE

}  // namespace Eval

#endif // defined(EVAL_LEARN) && defined(EVAL_NNUE)

#endif // !_NNUE_TRAINER_BINARIZATION_H_

