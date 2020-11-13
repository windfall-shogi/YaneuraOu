// NNUE評価関数の入力特徴量HalfKPの定義

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "half_kp.h"
#include "index_list.h"

namespace Eval {

namespace NNUE {

namespace Features {

template <Side AssociatedKing>
inline constexpr PieceNumber GetKingPieceNumber(const Color persepective) noexcept {
  return static_cast<PieceNumber>(PIECE_NUMBER_KING + (AssociatedKing == Side::kFriend ? persepective : ~persepective));
}

// 玉の位置とBonaPieceから特徴量のインデックスを求める
template <Side AssociatedKing>
inline IndexType HalfKP<AssociatedKing>::MakeIndex(Square sq_k, BonaPiece p) {
  return static_cast<IndexType>(fe_end) * static_cast<IndexType>(sq_k) + p;
}

// 駒の情報を取得する
template <Side AssociatedKing>
inline void HalfKP<AssociatedKing>::GetPieces(
    const Position& pos, Color perspective,
    BonaPiece** pieces, Square* sq_target_k) {
  *pieces = (perspective == BLACK) ?
      pos.eval_list()->piece_list_fb() :
      pos.eval_list()->piece_list_fw();
  const PieceNumber target = (AssociatedKing == Side::kFriend) ?
      static_cast<PieceNumber>(PIECE_NUMBER_KING + perspective) :
      static_cast<PieceNumber>(PIECE_NUMBER_KING + ~perspective);
  *sq_target_k = static_cast<Square>(((*pieces)[target] - f_king) % SQ_NB);
}

// 特徴量のうち、値が1であるインデックスのリストを取得する
template <Side AssociatedKing>
void HalfKP<AssociatedKing>::AppendActiveIndices(
    const Position& pos, Color perspective, IndexList* active) {
  // コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
  if (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

  BonaPiece* pieces;
  Square sq_target_k;
  GetPieces(pos, perspective, &pieces, &sq_target_k);
  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
    active->push_back(MakeIndex(sq_target_k, pieces[i]));
  }

  // perspectiveじゃない方の王を取得
  const auto opponent_king = GetKingPieceNumber<AssociatedKing>(~perspective);
  active->push_back(MakeIndex(sq_target_k, pieces[opponent_king]));
}

// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
template <Side AssociatedKing>
void HalfKP<AssociatedKing>::AppendChangedIndices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added) {
  BonaPiece* pieces;
  Square sq_target_k;
  GetPieces(pos, perspective, &pieces, &sq_target_k);
  const auto& dp = pos.state()->dirtyPiece;
  for (int i = 0; i < dp.dirty_num; ++i) {
    if (dp.pieceNo[i] == GetKingPieceNumber<AssociatedKing>(perspective)) {
      // 自分の王は何もしない
      continue;
    }
    const auto old_p = static_cast<BonaPiece>(
        dp.changed_piece[i].old_piece.from[perspective]);
    removed->push_back(MakeIndex(sq_target_k, old_p));
    const auto new_p = static_cast<BonaPiece>(
        dp.changed_piece[i].new_piece.from[perspective]);
    added->push_back(MakeIndex(sq_target_k, new_p));
  }
}

template class HalfKP<Side::kFriend>;
template class HalfKP<Side::kEnemy>;

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)
