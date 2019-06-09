// NNUE評価関数の学習クラステンプレートのAffineTransform用特殊化

#ifndef _NNUE_TRAINER_AFFINE_TRANSFORM_H_
#define _NNUE_TRAINER_AFFINE_TRANSFORM_H_

#include "../../../config.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE)

#include "../../../learn/learn.h"
#include "../layers/affine_transform.h"
#include "trainer.h"

#include <array>
#include <random>

#include <boost/range/algorithm/fill.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/range/algorithm_ext/for_each.hpp>

namespace Eval {

namespace NNUE {

// 学習：アフィン変換層
template <typename PreviousLayer, IndexType OutputDimensions>
class Trainer<Layers::AffineTransform<PreviousLayer, OutputDimensions>> {
 private:
  // 学習対象の層の型
  using LayerType = Layers::AffineTransform<PreviousLayer, OutputDimensions>;

 public:
  // ファクトリ関数
  static std::shared_ptr<Trainer> Create(
      LayerType* target_layer, FeatureTransformer* feature_transformer) {
    return std::shared_ptr<Trainer>(
        new Trainer(target_layer, feature_transformer));
  }

  // ハイパーパラメータなどのオプションを設定する
  void SendMessage(Message* message) {
    previous_layer_trainer_->SendMessage(message);
#if !defined(ADAM_UPDATE)
    if (ReceiveMessage("momentum", message)) {
      momentum_ = static_cast<LearnFloatType>(std::stod(message->value));
    }
#endif
    if (ReceiveMessage("learning_rate_scale", message)) {
      learning_rate_scale_ =
          static_cast<LearnFloatType>(std::stod(message->value));
    }
    if (ReceiveMessage("reset", message)) {
      DequantizeParameters();
    }
    if (ReceiveMessage("quantize_parameters", message)) {
      QuantizeParameters();
    }
  }

  // パラメータを乱数で初期化する
  template <typename RNG>
  void Initialize(RNG& rng) {
    previous_layer_trainer_->Initialize(rng);
    if (kIsOutputLayer) {
      // 出力層は0で初期化する
      std::fill(std::begin(biases_), std::end(biases_),
                static_cast<LearnFloatType>(0.0));
      std::fill(std::begin(weights_), std::end(weights_),
                static_cast<LearnFloatType>(0.0));
    } else {
      // 入力の分布が各ユニット平均0.5、等分散であることを仮定し、
      // 出力の分布が各ユニット平均0.5、入力と同じ等分散になるように初期化する
      const double kSigma = 1.0 / std::sqrt(kInputDimensions);
      auto distribution = std::normal_distribution<double>(0.0, kSigma);
      for (IndexType i = 0; i < kOutputDimensions; ++i) {
        double sum = 0.0;
        for (IndexType j = 0; j < kInputDimensions; ++j) {
          const auto weight = static_cast<LearnFloatType>(distribution(rng));
          weights_[kInputDimensions * i + j] = weight;
          sum += weight;
        }
        biases_[i] = static_cast<LearnFloatType>(0.5 - 0.5 * sum);
      }
    }
    QuantizeParameters();
  }

  // 順伝播
  const LearnFloatType* Propagate(const std::vector<Example>& batch) {
    if (output_.size() < kOutputDimensions * batch.size()) {
      output_.resize(kOutputDimensions * batch.size());
      gradients_.resize(kInputDimensions * batch.size());
#if defined(ADAM_UPDATE)
      gradients2_.resize(gradients_.size());
#endif
    }
    batch_size_ = static_cast<IndexType>(batch.size());
    batch_input_ = previous_layer_trainer_->Propagate(batch);
#if defined(USE_BLAS)
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType batch_offset = kOutputDimensions * b;
      cblas_scopy(kOutputDimensions, biases_, 1, &output_[batch_offset], 1);
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                kOutputDimensions, batch_size_, kInputDimensions, 1.0,
                weights_, kInputDimensions,
                batch_input_, kInputDimensions,
                1.0, &output_[0], kOutputDimensions);
#else
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType input_batch_offset = kInputDimensions * b;
      const IndexType output_batch_offset = kOutputDimensions * b;
      for (IndexType i = 0; i < kOutputDimensions; ++i) {
        double sum = biases_[i];
        for (IndexType j = 0; j < kInputDimensions; ++j) {
          const IndexType index = kInputDimensions * i + j;
          sum += weights_[index] * batch_input_[input_batch_offset + j];
        }
        output_[output_batch_offset + i] = static_cast<LearnFloatType>(sum);
      }
    }
#endif
    return output_.data();
  }

  // 逆伝播
  void Backpropagate(const LearnFloatType* gradients,
                     LearnFloatType learning_rate) {
    const LearnFloatType local_learning_rate =
        learning_rate * learning_rate_scale_;
#if defined(USE_BLAS)
    // backpropagate
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                kInputDimensions, batch_size_, kOutputDimensions, 1.0,
                weights_, kInputDimensions,
                gradients, kOutputDimensions,
                0.0, &gradients_[0], kInputDimensions);
    // update
#if defined(ADAM_UPDATE)
    if (one_.size() < batch_size_) {
      one_.resize(batch_size_);
      boost::fill(one_, 1);
    }

    // biases_m = beta1 * biases_m + (1 - beta1) * grad
    cblas_sgemv(CblasColMajor, CblasNoTrans, kOutputDimensions, batch_size_,
                1 - beta1_, gradients, kOutputDimensions, one_.data(), 1,
                beta1_, biases_m_.data(), 1);
    // weights_m = beta1 * weights_m + (1 - beta1) * grad
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, kOutputDimensions,
                kInputDimensions, batch_size_, 1 - beta1_, gradients,
                kOutputDimensions, batch_input_, kInputDimensions, beta1_,
                weights_m_.data(), kInputDimensions);

#if defined(USE_IPP)
    ippsSqr_32f(gradients, gradients2_.data(), kOutputDimensions * batch_size_);
#else
    std::transform(gradients, gradients + kOutputDimensions * batch_size_,
                   gradients2_.data(),
                   [](const LearnFloatType v) { return powf(v, 2); });
#endif

    // biases_v = beta2 * biases_v + (1 - beta2) * grad^2
    cblas_sgemv(CblasColMajor, CblasNoTrans, biases_v_.size(), batch_size_,
                1 - beta2_, gradients2_.data(), biases_v_.size(), one_.data(),
                1, beta2_, biases_v_.data(), 1);
    // weights_v = beta2 * weights_v + (1 - beta2) * grad^2
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, kOutputDimensions,
                kInputDimensions, batch_size_, 1 - beta2_, gradients2_.data(),
                kOutputDimensions, batch_input_, kInputDimensions, beta2_,
                weights_v_.data(), kInputDimensions);

#if defined(USE_MKL)
    // biases_m_hat = biases_m / (1 - beta1^t)
    cblas_saxpby(biases_m_hat_.size(), 1 / (1 - powf(beta1_, t_)),
                 biases_m_.data(), 1, 0, biases_m_hat_.data(), 1);
    // biases_v_hat = biases_v / (1 - beta2^t)
    cblas_saxpby(biases_v_hat_.size(), 1 / (1 - powf(beta2_, t_)),
                 biases_v_.data(), 1, 0, biases_v_hat_.data(), 1);

    // weights_m_hat = weights_m / (1 - beta1^t)
    cblas_saxpby(weights_m_hat_.size(), 1 / (1 - powf(beta1_, t_)),
                 weights_m_.data(), 1, 0, weights_m_hat_.data(), 1);
    // weights_v_hat = weights_v / (1 - beta2^t)
    cblas_saxpby(weights_v_hat_.size(), 1 / (1 - powf(beta2_, t_)),
                 weights_v_.data(), 1, 0, weights_v_hat_.data(), 1);
#else
    // biases_m_hat = biases_m / (1 - beta1^t)
    cblas_scopy(biases_m_.size(), biases_m_.data(), 1, biases_m_hat_.data(), 1);
    cblas_sscal(biases_m_hat_.size(), 1 / (1 - powf(beta1_, t_)),
                biases_m_hat_.data(), 1);
    // biases_v_hat = biases_v / (1 - beta2^t)
    cblas_scopy(biases_v_.size(), biases_v_.data(), 1, biases_v_hat_.data(), 1);
    cblas_sscal(biases_v_hat_.size(), 1 / (1 - powf(beta2_, t_)),
                biases_v_hat_.data(), 1);

    // weights_m_hat = weights_m / (1 - beta1^t)
    cblas_scopy(weights_m_.size(), weights_m_.data(), 1, weights_m_hat_.data(),
                1);
    cblas_sscal(weights_m_hat_.size(), 1 / (1 - powf(beta1_, t_)),
                weights_m_hat_.data(), 1);
    // weights_v_hat = weights_v / (1 - beta2^t)
    cblas_scopy(weights_v_.size(), weights_v_.data(), 1, weights_v_hat_.data(),
                1);
    cblas_sscal(weights_v_hat_.size(), 1 / (1 - powf(beta2_, t_)),
                weights_v_hat_.data(), 1);
#endif  // defined(USE_MKL)
    ++t_;

#if defined(USE_IPP)
    // sqrt(in-place)
    ippsSqrt_32f_I(biases_v_hat_.data(), biases_v_hat_.size());
    ippsSqrt_32f_I(weights_v_hat_.data(), weights_v_hat_.size());

    // +epsilon(in-place)
    ippsAddC_32f_I(epsilon_, biases_v_hat_.data(), biases_v_hat_.size());
    ippsAddC_32f_I(epsilon_, weights_v_hat_.data(), weights_v_hat_.size());

    // m_hat = m_hat / v_hat
    ippsDiv_32f_I(biases_v_hat_.data(), biases_m_hat_.data(),
                  biases_v_hat_.size());
    ippsDiv_32f_I(weights_v_hat_.data(), weights_m_hat_.data(),
                  weights_v_hat_.size());
#else
    // m_hat / (sqrt(v_hat) + epsilon)
    boost::for_each(biases_v_hat_, biases_m_hat_,
                    [&](const LearnFloatType v, LearnFloatType& m) {
                      m /= sqrtf(v) + epsilon_;
                    });
    boost::for_each(weights_v_hat_, weights_m_hat_,
                    [&](const LearnFloatType v, LearnFloatType& m) {
                      m /= sqrtf(v) + epsilon_;
                    });
#endif  // defined(USE_IPP)

    cblas_saxpy(biases_m_hat_.size(), -local_learning_rate,
                biases_m_hat_.data(), 1, biases_, 1);
    cblas_saxpy(weights_m_hat_.size(), -local_learning_rate,
                weights_m_hat_.data(), 1, weights_, 1);
#else
    cblas_sscal(kOutputDimensions, momentum_, biases_diff_, 1);
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType batch_offset = kOutputDimensions * b;
      cblas_saxpy(kOutputDimensions, 1.0,
                  &gradients[batch_offset], 1, biases_diff_, 1);
    }
    cblas_saxpy(kOutputDimensions, -local_learning_rate,
                biases_diff_, 1, biases_, 1);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                kOutputDimensions, kInputDimensions, batch_size_, 1.0,
                gradients, kOutputDimensions,
                batch_input_, kInputDimensions,
                momentum_, weights_diff_, kInputDimensions);
    cblas_saxpy(kOutputDimensions * kInputDimensions, -local_learning_rate,
                weights_diff_, 1, weights_, 1);
#endif // defined(ADAM_UPDATE)
#else
    // backpropagate
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType input_batch_offset = kInputDimensions * b;
      const IndexType output_batch_offset = kOutputDimensions * b;
      for (IndexType j = 0; j < kInputDimensions; ++j) {
        double sum = 0.0;
        for (IndexType i = 0; i < kOutputDimensions; ++i) {
          const IndexType index = kInputDimensions * i + j;
          sum += weights_[index] * gradients[output_batch_offset + i];
        }
        gradients_[input_batch_offset + j] = static_cast<LearnFloatType>(sum);
      }
    }
    // update
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      biases_diff_[i] *= momentum_;
    }
    for (IndexType i = 0; i < kOutputDimensions * kInputDimensions; ++i) {
      weights_diff_[i] *= momentum_;
    }
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType input_batch_offset = kInputDimensions * b;
      const IndexType output_batch_offset = kOutputDimensions * b;
      for (IndexType i = 0; i < kOutputDimensions; ++i) {
        biases_diff_[i] += gradients[output_batch_offset + i];
      }
      for (IndexType i = 0; i < kOutputDimensions; ++i) {
        for (IndexType j = 0; j < kInputDimensions; ++j) {
          const IndexType index = kInputDimensions * i + j;
          weights_diff_[index] += gradients[output_batch_offset + i] *
              batch_input_[input_batch_offset + j];
        }
      }
    }
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      biases_[i] -= local_learning_rate * biases_diff_[i];
    }
    for (IndexType i = 0; i < kOutputDimensions * kInputDimensions; ++i) {
      weights_[i] -= local_learning_rate * weights_diff_[i];
    }
#endif
    previous_layer_trainer_->Backpropagate(gradients_.data(), learning_rate);
  }

 private:
  // コンストラクタ
  Trainer(LayerType* target_layer, FeatureTransformer* feature_transformer) :
      batch_size_(0),
      batch_input_(nullptr),
      previous_layer_trainer_(Trainer<PreviousLayer>::Create(
          &target_layer->previous_layer_, feature_transformer)),
      target_layer_(target_layer),
      biases_(),
      weights_(),
#if defined(ADAM_UPDATE)
      biases_m_(),
      biases_v_(),
      weights_m_(),
      weights_v_(),
      beta1_(0.9f),
      beta2_(0.999f),
      epsilon_(1e-8f),
      t_(1),
#else
      biases_diff_(),
      weights_diff_(),
      momentum_(0.0),
#endif // defined(ADAM_UPDATE)
      learning_rate_scale_(1.0) {
    DequantizeParameters();
  }

  // 重みの飽和とパラメータの整数化
  void QuantizeParameters() {
    for (IndexType i = 0; i < kOutputDimensions * kInputDimensions; ++i) {
      weights_[i] = std::max(-kMaxWeightMagnitude,
                             std::min(+kMaxWeightMagnitude, weights_[i]));
    }
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      target_layer_->biases_[i] =
          Round<typename LayerType::BiasType>(biases_[i] * kBiasScale);
    }
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      const auto offset = kInputDimensions * i;
      const auto padded_offset = LayerType::kPaddedInputDimensions * i;
      for (IndexType j = 0; j < kInputDimensions; ++j) {
        target_layer_->weights_[padded_offset + j] =
            Round<typename LayerType::WeightType>(
                weights_[offset + j] * kWeightScale);
      }
    }
  }

  // 整数化されたパラメータの読み込み
  void DequantizeParameters() {
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      biases_[i] = static_cast<LearnFloatType>(
          target_layer_->biases_[i] / kBiasScale);
    }
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      const auto offset = kInputDimensions * i;
      const auto padded_offset = LayerType::kPaddedInputDimensions * i;
      for (IndexType j = 0; j < kInputDimensions; ++j) {
        weights_[offset + j] = static_cast<LearnFloatType>(
            target_layer_->weights_[padded_offset + j] / kWeightScale);
      }
    }
#if defined(ADAM_UPDATE)
    boost::fill(biases_m_, 0);
    boost::fill(biases_v_, 0);
    boost::fill(weights_m_, 0);
    boost::fill(weights_v_, 0);
#else
    std::fill(std::begin(biases_diff_), std::end(biases_diff_),
              static_cast<LearnFloatType>(0.0));
    std::fill(std::begin(weights_diff_), std::end(weights_diff_),
              static_cast<LearnFloatType>(0.0));
#endif
  }

  // 入出力の次元数
  static constexpr IndexType kInputDimensions = LayerType::kInputDimensions;
  static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;

  // 出力の次元数が1なら出力層
  static constexpr bool kIsOutputLayer = kOutputDimensions == 1;

  // パラメータの整数化で用いる係数
  static constexpr LearnFloatType kActivationScale =
      std::numeric_limits<std::int8_t>::max();
  static constexpr LearnFloatType kBiasScale = kIsOutputLayer ?
      (kPonanzaConstant * FV_SCALE) :
      ((1 << kWeightScaleBits) * kActivationScale);
  static constexpr LearnFloatType kWeightScale = kBiasScale / kActivationScale;

  // パラメータの整数化でオーバーフローさせないために用いる重みの絶対値の上限
  static constexpr LearnFloatType kMaxWeightMagnitude =
      std::numeric_limits<typename LayerType::WeightType>::max() / kWeightScale;

  // ミニバッチのサンプル数
  IndexType batch_size_;

  // ミニバッチの入力
  const LearnFloatType* batch_input_;

  // 直前の層のTrainer
  const std::shared_ptr<Trainer<PreviousLayer>> previous_layer_trainer_;

  // 学習対象の層
  LayerType* const target_layer_;

  // パラメータ
  LearnFloatType biases_[kOutputDimensions];
  LearnFloatType weights_[kOutputDimensions * kInputDimensions];

  // パラメータの更新で用いるバッファ
#if defined(ADAM_UPDATE)
  std::array<LearnFloatType, kOutputDimensions> biases_m_, biases_v_, biases_m_hat_, biases_v_hat_;
  std::array<LearnFloatType, kOutputDimensions * kInputDimensions> weights_m_, weights_v_, weights_m_hat_, weights_v_hat_;
  std::vector<LearnFloatType> one_;
  std::vector<LearnFloatType> gradients2_;
  uint64_t t_;
#else
  LearnFloatType biases_diff_[kOutputDimensions];
  LearnFloatType weights_diff_[kOutputDimensions * kInputDimensions];
#endif

  // 順伝播用バッファ
  std::vector<LearnFloatType> output_;

  // 逆伝播用バッファ
  std::vector<LearnFloatType> gradients_;

  // ハイパーパラメータ
  LearnFloatType learning_rate_scale_;
#if defined(ADAM_UPDATE)
  LearnFloatType beta1_, beta2_;
  LearnFloatType epsilon_;
#else
  LearnFloatType momentum_;
#endif
};

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE)

#endif
