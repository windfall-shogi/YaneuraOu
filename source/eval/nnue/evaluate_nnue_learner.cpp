// NNUE評価関数の学習時用のコード

#include "../../config.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE)

#include <random>
#include <fstream>

#include "../../learn/learn.h"
#include "../../learn/learning_tools.h"

#include "../../position.h"
#include "../../usi.h"
#include "../../misc.h"

#include "../evaluate_common.h"

#include "evaluate_nnue.h"
#include "evaluate_nnue_learner.h"
#include "trainer/features/factorizer_feature_set.h"
// NNUE-HalfKPE9
//#include "trainer/features/factorizer_half_kp.h"
#include "trainer/features/factorizer_half_kpe9.h"
#include "trainer/trainer_feature_transformer.h"
#include "trainer/trainer_input_slice.h"
#include "trainer/trainer_affine_transform.h"
#include "trainer/trainer_clipped_relu.h"
#include "trainer/trainer_sum.h"

#if defined(USE_LIBTORCH)
#include "evaluate_nnue_torch_model.h"
#endif // defined(USE_LIBTORCH)

#if defined(USE_LIBTORCH)
namespace Learner {
// learner.cppの方で定義している
// elmo絞りのlambdaを取得する関数
torch::Tensor GetLambda(const torch::Tensor& deep);
}
#endif // defined(USE_LIBTORCH)

namespace Eval {

namespace NNUE {

namespace {

// 学習データ
std::vector<Example> examples;

// examplesの排他制御をするMutex
std::mutex examples_mutex;

// ミニバッチのサンプル数
u64 batch_size;

// 乱数生成器
std::mt19937 rng;

// 学習器
std::shared_ptr<Trainer<Network>> trainer;
#if defined(USE_LIBTORCH)
// libtorchで作った学習用のモデル
Net net;
torch::optim::SGD optimizer(net->parameters(), 0.01);
#endif // defined(USE_LIBTORCH)


// 学習率のスケール
double global_learning_rate_scale;

// 学習率のスケールを取得する
double GetGlobalLearningRateScale() {
  return global_learning_rate_scale;
}

// ハイパーパラメータなどのオプションを学習器に伝える
void SendMessages(std::vector<Message> messages) {
  for (auto& message : messages) {
    trainer->SendMessage(&message);
    ASSERT_LV3(message.num_receivers > 0);
  }
}

}  // namespace

// 学習の初期化を行う
void InitializeTraining(double eta1, u64 eta1_epoch,
                        double eta2, u64 eta2_epoch, double eta3) {
  std::cout << "Initializing NN training for "
            << GetArchitectureString() << std::endl;

  ASSERT(feature_transformer);
  ASSERT(network);
  trainer = Trainer<Network>::Create(network.get(), feature_transformer.get());

  if (Options["SkipLoadingEval"]) {
    trainer->Initialize(rng);
  }

  global_learning_rate_scale = 1.0;
  EvalLearningTools::Weight::init_eta(eta1, eta2, eta3, eta1_epoch, eta2_epoch);
}

// ミニバッチのサンプル数を設定する
void SetBatchSize(u64 size) {
  ASSERT_LV3(size > 0);
  batch_size = size;
}

// 学習率のスケールを設定する
void SetGlobalLearningRateScale(double scale) {
  global_learning_rate_scale = scale;
}

// ハイパーパラメータなどのオプションを設定する
void SetOptions(const std::string& options) {
  std::vector<Message> messages;
  for (const auto& option : Split(options, ',')) {
    const auto fields = Split(option, '=');
    ASSERT_LV3(fields.size() == 1 || fields.size() == 2);
    if (fields.size() == 1) {
      messages.emplace_back(fields[0]);
    } else {
      messages.emplace_back(fields[0], fields[1]);
    }
  }
  SendMessages(std::move(messages));
}

// 学習用評価関数パラメータをファイルから読み直す
void RestoreParameters(const std::string& dir_name) {
  const std::string file_name = Path::Combine(dir_name, NNUE::kFileName);
  std::ifstream stream(file_name, std::ios::binary);
  bool result = ReadParameters(stream);
  ASSERT(result);

  SendMessages({{"reset"}});
}

// 学習データを1サンプル追加する
void AddExample(Position& pos, Color rootColor,
                const Learner::PackedSfenValue& psv, double weight) {
  Example example;
  if (rootColor == pos.side_to_move()) {
    example.sign = 1;
  } else {
    example.sign = -1;
  }
  example.psv = psv;
  example.weight = weight;

  Features::IndexList active_indices[2];
  for (const auto trigger : kRefreshTriggers) {
    RawFeatures::AppendActiveIndices(pos, trigger, active_indices);
  }
  if (pos.side_to_move() != BLACK) {
    active_indices[0].swap(active_indices[1]);
  }
  for (const auto color : COLOR) {
    std::vector<TrainingFeature> training_features;
    for (const auto base_index : active_indices[color]) {
      static_assert(Features::Factorizer<RawFeatures>::GetDimensions() <
                    (1 << TrainingFeature::kIndexBits), "");
      Features::Factorizer<RawFeatures>::AppendTrainingFeatures(
          base_index, &training_features);
    }
    std::sort(training_features.begin(), training_features.end());

    auto& unique_features = example.training_features[color];
    for (const auto& feature : training_features) {
      if (!unique_features.empty() &&
          feature.GetIndex() == unique_features.back().GetIndex()) {
        unique_features.back() += feature;
      } else {
        unique_features.push_back(feature);
      }
    }
  }

  std::lock_guard<std::mutex> lock(examples_mutex);
  examples.push_back(std::move(example));
}

// 評価関数パラメーターを更新する
void UpdateParameters(u64 epoch) {
  ASSERT_LV3(batch_size > 0);

  EvalLearningTools::Weight::calc_eta(epoch);
  const auto learning_rate = static_cast<LearnFloatType>(
      get_eta() / batch_size);

  std::lock_guard<std::mutex> lock(examples_mutex);
  std::shuffle(examples.begin(), examples.end(), rng);
  while (examples.size() >= batch_size) {
    std::vector<Example> batch(examples.end() - batch_size, examples.end());
    examples.resize(examples.size() - batch_size);

    const auto network_output = trainer->Propagate(batch);

    std::vector<LearnFloatType> gradients(batch.size());
    for (std::size_t b = 0; b < batch.size(); ++b) {
      const auto shallow = static_cast<Value>(Round<std::int32_t>(
          batch[b].sign * network_output[b] * kPonanzaConstant));
      const auto& psv = batch[b].psv;
      const double gradient = batch[b].sign * Learner::calc_grad(shallow, psv);
      gradients[b] = static_cast<LearnFloatType>(gradient * batch[b].weight);
    }

    trainer->Backpropagate(gradients.data(), learning_rate);
  }
  SendMessages({{"quantize_parameters"}});
}

// 学習に問題が生じていないかチェックする
void CheckHealth() {
  SendMessages({{"check_health"}});
}

}  // namespace NNUE

// 評価関数パラメーターをファイルに保存する
void save_eval(std::string dir_name) {
  auto eval_dir = Path::Combine(Options["EvalSaveDir"], dir_name);
  std::cout << "save_eval() start. folder = " << eval_dir << std::endl;

  // すでにこのフォルダがあるならCreateFolder()に失敗するが、
  // 別にそれは構わない。なければ作って欲しいだけ。
  // また、EvalSaveDirまでのフォルダは掘ってあるものとする。
  Directory::CreateFolder(eval_dir);

  if (Options["SkipLoadingEval"] && NNUE::trainer) {
    NNUE::SendMessages({{"clear_unobserved_feature_weights"}});
  }

  const std::string file_name = Path::Combine(eval_dir, NNUE::kFileName);
  std::ofstream stream(file_name, std::ios::binary);
  const bool result = NNUE::WriteParameters(stream);

  if (!result)
  {
      std::cout << "Error!! : save_eval() failed." << std::endl;
      Tools::exit();
  }

  std::cout << "save_eval() finished." << std::endl;
}

// 現在のetaを取得する
double get_eta() {
  return NNUE::GetGlobalLearningRateScale() * EvalLearningTools::Weight::eta;
}

#if defined(USE_LIBTORCH)
namespace NNUE {
void UpdateParametersTorch(u64 epoch) {
  ASSERT_LV3(batch_size > 0);

  EvalLearningTools::Weight::calc_eta(epoch);
  const auto new_lr = static_cast<LearnFloatType>(get_eta() / batch_size);

  // 学習係数の変更方法
  // https://stackoverflow.com/questions/62415285/updating-learning-rate-with-libtorch-1-5-and-optimiser-options-in-c
  for (auto param_group : optimizer.param_groups()) {
    // Static cast needed as options() returns OptimizerOptions(base class)
    static_cast<torch::optim::SGDOptions&>(param_group.options()).lr(new_lr);
  }

  std::lock_guard<std::mutex> lock(examples_mutex);
  std::shuffle(examples.begin(), examples.end(), rng);

  const auto data_size = batch_size * RawFeatures::kMaxActiveDimensions;
  std::vector<int32_t> indices[2];
  for (int i = 0; i < 2; ++i) {
    indices[i].reserve(data_size);
  }
  std::vector<LearnFloatType> weights(batch_size);
  std::vector<LearnFloatType> signs(batch_size);
  std::vector<LearnFloatType> target_values(batch_size);
  std::vector<LearnFloatType> game_results(batch_size);

  while (examples.size() >= batch_size) {
    indices[0].clear();
    indices[1].clear();
    weights.clear();
    signs.clear();
    target_values.clear();
    game_results.clear();

    // サンプルごとに独立のデータを連続な配列に入れ直す
    for (auto it = examples.end() - batch_size; it != examples.end(); ++it) {
      for (int i = 0; i < 2; ++i) {
        for (const auto feature : it->training_features[i]) {
          const auto index = feature.GetIndex();
          const auto count = feature.GetCount();
          for (int j = 0; j < count; ++j) {
            indices[i].push_back(index);
          }
        }
      }
      weights.push_back(it->weight);
      signs.push_back(it->sign);
      target_values.push_back(it->psv.score);
      game_results.push_back(it->psv.game_result);
    }
    examples.resize(examples.size() - batch_size);

    torch::Tensor p = torch::from_blob(
        indices[0].data(),
        {static_cast<int>(batch_size), RawFeatures::kMaxActiveDimensions},
        torch::TensorOptions(torch::kI32));
    torch::Tensor q = torch::from_blob(
        indices[1].data(),
        {static_cast<int>(batch_size), RawFeatures::kMaxActiveDimensions},
        torch::TensorOptions(torch::kI32));
    torch::Tensor w = torch::from_blob(weights.data(), batch_size,
                                       torch::TensorOptions(torch::kF32));
    torch::Tensor s = torch::from_blob(signs.data(), batch_size,
                                       torch::TensorOptions(torch::kF32));
    torch::Tensor t = torch::from_blob(target_values.data(), batch_size,
                                       torch::TensorOptions(torch::kF32));
    torch::Tensor r = torch::from_blob(game_results.data(), batch_size,
                                       torch::TensorOptions(torch::kF32));

    net->zero_grad();
    auto outputs = net(p, q);
    outputs.squeeze_();
    auto eval_winrate = torch::sigmoid(s * outputs);

    auto teacher_winrate = torch::sigmoid(t / 600.0);
    r = (r + 1) * 0.5;
    const auto lambda = Learner::GetLambda(t);

    auto loss = w * ((1 - lambda) * (eval_winrate - r).square() +
                     lambda * (eval_winrate - teacher_winrate).square());
    loss = torch::mean(loss);
    loss.backward();

    optimizer.step();
  }
  SendMessages({ {"quantize_parameters"} });
}
} // namespace NNUE
#endif // defined(USE_LIBTORCH)

}  // namespace Eval

#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE)
