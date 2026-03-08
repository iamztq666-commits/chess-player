
from chess_tournament import Player
import chess
import random
from typing import Optional, List, Tuple


class TransformerPlayer(Player):
    """
    LM-guided chess player with lightweight tactical filtering.

    Design:
    1) Heuristics rank all legal moves.
    2) A 1-ply opponent-response filter heavily penalizes moves that allow
       immediate tactical or promotion threats.
    3) The language model is used only as a final tie-break among the best
       surviving candidates.

    This is intentionally still lightweight, but much stronger than selecting
    moves from LM log-probability alone.
    """

    HF_MODEL_ID = "hiiamkik/Chess-1.7B-v2"
    MAX_NEW_TOKENS = 6
    TEMPERATURE = 0.10
    NUM_CANDIDATES = 20
    SHORTLIST_SIZE = 16
    FINALISTS = 4

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)
        self._model = None
        self._tokenizer = None
        self._device = None

        self._move_history: List[str] = []
        self._last_choice: Optional[Tuple[str, str]] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[TransformerPlayer] Loading {self.HF_MODEL_ID} on {self._device}...")

            self._tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)
            self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.HF_MODEL_ID,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self._model.eval()
            print(f"[TransformerPlayer] Ready on {self._device}")
        except Exception as e:
            print(f"[TransformerPlayer] Load failed: {e}")
            self._model = None

    # ------------------------------------------------------------------
    # History / game boundaries
    # ------------------------------------------------------------------
    def _maybe_reset_history(self, fen: str) -> None:
        try:
            board = chess.Board(fen)
            initial = chess.Board()
            if (
                board.board_fen() == initial.board_fen()
                and board.turn == chess.WHITE
                and board.fullmove_number == 1
            ):
                self._move_history.clear()
                self._last_choice = None
        except Exception:
            pass

    def _append_history_once(self, fen: str, move_uci: str) -> None:
        if self._last_choice == (fen, move_uci):
            return
        self._move_history.append(move_uci)
        self._last_choice = (fen, move_uci)

    # ------------------------------------------------------------------
    # LM helpers
    # ------------------------------------------------------------------
    def _build_prompt(self, fen: str) -> str:
        recent = self._move_history[-8:]
        if recent:
            return f"FEN: {fen} Moves: {' '.join(recent)} Best Move:"
        return f"FEN: {fen} Best Move:"

    def _generate_candidates(self, fen: str) -> List[str]:
        import torch

        prompt = self._build_prompt(fen)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=self.TEMPERATURE,
                num_return_sequences=self.NUM_CANDIDATES,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        seen = set()
        result: List[str] = []

        for out in outputs:
            text = self._tokenizer.decode(out, skip_special_tokens=True)
            if "Move:" not in text:
                continue
            tail = text.split("Move:")[-1].strip()
            if not tail:
                continue
            move = tail.split()[0].strip().lower()
            if len(move) >= 4 and move not in seen:
                seen.add(move)
                result.append(move)

        return result

    def _lm_score_move(self, fen: str, move_uci: str) -> float:
        import torch
        import torch.nn.functional as F

        prompt = self._build_prompt(fen)
        full_text = prompt + " " + move_uci

        prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        inputs = self._tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        log_probs = F.log_softmax(logits, dim=-1)
        input_ids = inputs["input_ids"][0]
        move_ids = input_ids[len(prompt_ids):]

        if len(move_ids) == 0:
            return float("-inf")

        score = 0.0
        for i in range(len(move_ids)):
            pos = len(prompt_ids) - 1 + i
            score += log_probs[0, pos, move_ids[i]].item()
        return score

    # ------------------------------------------------------------------
    # Chess helpers
    # ------------------------------------------------------------------
    def _is_back_and_forth(self, move_uci: str) -> bool:
        if not self._move_history:
            return False
        last = self._move_history[-1]
        if len(last) < 4 or len(move_uci) < 4:
            return False
        return last[:2] == move_uci[2:4] and last[2:4] == move_uci[:2]

    def _same_piece_overused_penalty(self, move_uci: str) -> float:
        """
        Penalize repeatedly moving the same piece origin among recent own moves.
        """
        if len(move_uci) < 4:
            return 0.0
        frm = move_uci[:2]
        recent = self._move_history[-6:]
        count = 0
        for m in recent:
            if len(m) >= 4 and m[:2] == frm:
                count += 1
        if count >= 2:
            return -160.0
        if count == 1:
            return -50.0
        return 0.0

    def _count_undeveloped_minors(self, board: chess.Board, color: chess.Color) -> int:
        back_rank = 0 if color == chess.WHITE else 7
        total = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == color and p.piece_type in (chess.KNIGHT, chess.BISHOP):
                if chess.square_rank(sq) == back_rank:
                    total += 1
        return total

    def _queen_rook_early_shuffle_penalty(self, board: chess.Board, move: chess.Move) -> float:
        piece = board.piece_at(move.from_square)
        if piece is None:
            return 0.0

        undeveloped = self._count_undeveloped_minors(board, board.turn)
        uncastled = board.has_kingside_castling_rights(board.turn) or board.has_queenside_castling_rights(board.turn)

        if piece.piece_type == chess.QUEEN and (undeveloped > 0 or uncastled):
            return -180.0

        if piece.piece_type == chess.ROOK and (undeveloped >= 2 or uncastled):
            return -140.0

        return 0.0

    def _move_is_progress(self, board: chess.Board, move: chess.Move) -> bool:
        piece = board.piece_at(move.from_square)

        if board.is_capture(move) or move.promotion or board.is_castling(move):
            return True

        board.push(move)
        try:
            if board.is_check():
                return True
        finally:
            board.pop()

        if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            if board.turn == chess.WHITE and from_rank == 0 and to_rank > from_rank:
                return True
            if board.turn == chess.BLACK and from_rank == 7 and to_rank < from_rank:
                return True

        if piece and piece.piece_type == chess.PAWN:
            return True

        return False

    def _non_progress_penalty(self, board: chess.Board, move: chess.Move) -> float:
        penalty = 0.0
        if not self._move_is_progress(board, move):
            penalty -= 140.0

        if self._is_back_and_forth(move.uci()):
            penalty -= 320.0

        penalty += self._same_piece_overused_penalty(move.uci())
        penalty += self._queen_rook_early_shuffle_penalty(board, move)
        return penalty

    def _move_gives_check(self, board: chess.Board, move: chess.Move) -> bool:
        board.push(move)
        try:
            return board.is_check()
        finally:
            board.pop()

    def _mvvlva_score(self, board: chess.Board, move: chess.Move) -> float:
        if not board.is_capture(move):
            return 0.0
        mover = board.piece_at(move.from_square)
        if board.is_en_passant(move):
            captured_value = self.PIECE_VALUES[chess.PAWN]
        else:
            captured = board.piece_at(move.to_square)
            captured_value = self.PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
        mover_value = self.PIECE_VALUES.get(mover.piece_type, 0) if mover else 0
        return captured_value - 0.2 * mover_value

    def _hanging_penalty_after_move(self, board: chess.Board, move: chess.Move) -> float:
        board.push(move)
        try:
            moved_piece = board.piece_at(move.to_square)
            if moved_piece is None:
                return 0.0

            us = not board.turn
            them = board.turn

            attackers = board.attackers(them, move.to_square)
            defenders = board.attackers(us, move.to_square)

            penalty = 0.0
            if len(attackers) > len(defenders):
                penalty -= 50.0
            if board.is_attacked_by(them, move.to_square):
                penalty -= 15.0
            return penalty
        finally:
            board.pop()

    def _passed_pawn_push_bonus(self, board: chess.Board, move: chess.Move) -> float:
        piece = board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return 0.0

        bonus = 10.0
        to_rank = chess.square_rank(move.to_square)
        if board.turn == chess.WHITE:
            bonus += 8.0 * to_rank
        else:
            bonus += 8.0 * (7 - to_rank)
        return bonus

    def _heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0.0
        piece = board.piece_at(move.from_square)

        score += self._mvvlva_score(board, move)

        if move.promotion:
            score += 1100.0

        if self._move_gives_check(board, move):
            score += 120.0

        if board.is_castling(move):
            score += 100.0

        if piece and piece.piece_type == chess.PAWN:
            score += self._passed_pawn_push_bonus(board, move)

        if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            score += 20.0

        score += self._hanging_penalty_after_move(board, move)
        score += self._non_progress_penalty(board, move)

        board.push(move)
        try:
            score += 0.3 * board.legal_moves.count()
        finally:
            board.pop()

        return score

    # ------------------------------------------------------------------
    # 1-ply threat filter
    # ------------------------------------------------------------------
    def _opponent_best_reply_penalty(self, board: chess.Board, our_move: chess.Move) -> float:
        """
        After our move, evaluate the strongest immediate opponent reply.
        Large negative means our move allows dangerous immediate tactics.
        """
        board.push(our_move)
        try:
            if board.is_checkmate():
                return 100000.0  # we just mated
            if board.is_stalemate():
                return -50.0

            penalty = 0.0
            opp_moves = list(board.legal_moves)
            if not opp_moves:
                return penalty

            worst = 0.0
            our_color_after_push = not board.turn  # side that made our_move

            for reply in opp_moves:
                local = 0.0

                if reply.promotion:
                    local -= 2500.0

                if board.is_capture(reply):
                    captured_piece = None
                    if board.is_en_passant(reply):
                        local -= 140.0
                    else:
                        captured_piece = board.piece_at(reply.to_square)
                        if captured_piece:
                            local -= 1.8 * self.PIECE_VALUES.get(captured_piece.piece_type, 0)

                board.push(reply)
                try:
                    if board.is_checkmate():
                        local -= 100000.0
                    elif board.is_check():
                        local -= 260.0

                    # Opponent passed / advanced pawn danger
                    for sq in chess.SQUARES:
                        p = board.piece_at(sq)
                        if p and p.color != our_color_after_push and p.piece_type == chess.PAWN:
                            rank = chess.square_rank(sq)
                            if p.color == chess.WHITE and rank >= 5:
                                local -= 20.0 * (rank - 4)
                            if p.color == chess.BLACK and rank <= 2:
                                local -= 20.0 * (3 - rank)

                    # Our king under direct heat
                    king_sq = board.king(our_color_after_push)
                    if king_sq is not None:
                        attackers = board.attackers(not our_color_after_push, king_sq)
                        if len(attackers) >= 2:
                            local -= 120.0
                finally:
                    board.pop()

                if local < worst:
                    worst = local

            penalty += worst

            # Penalize repeated position unless we have no good alternative.
            if board.can_claim_threefold_repetition() or board.is_repetition(2):
                penalty -= 140.0

            return penalty
        finally:
            board.pop()

    def _combined_score(self, board: chess.Board, move: chess.Move) -> float:
        return self._heuristic_score(board, move) + self._opponent_best_reply_penalty(board, move)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def get_move(self, fen: str) -> Optional[str]:
        self._maybe_reset_history(fen)

        if self._model is None:
            self._load_model()

        try:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None

            # Stage 1: full-board tactical/heuristic scoring
            scored = []
            for move in legal_moves:
                score = self._combined_score(board, move)
                scored.append((score, move))

            scored.sort(key=lambda x: x[0], reverse=True)
            shortlist = [m for _, m in scored[: min(self.SHORTLIST_SIZE, len(scored))]]

            # If no LM, return best shortlist move
            if self._model is None:
                chosen = shortlist[0].uci()
                self._append_history_once(fen, chosen)
                return chosen

            # Stage 2: keep only best finalists for LM tie-break
            finalists = shortlist[: min(self.FINALISTS, len(shortlist))]
            finalists_uci = [m.uci() for m in finalists]

            # Stage 3: LM-generated candidate intersection, but only among finalists
            generated = self._generate_candidates(fen)
            lm_pool = [m for m in generated if m in finalists_uci]

            if not lm_pool:
                lm_pool = finalists_uci

            # Final tie-break: small LM preference on already-vetted moves
            best_move = lm_pool[0]
            best_score = float("-inf")

            finalist_scores = {m.uci(): s for s, m in scored[: min(self.SHORTLIST_SIZE, len(scored))]}

            for move_uci in lm_pool:
                lm_score = self._lm_score_move(fen, move_uci)
                base = finalist_scores.get(move_uci, 0.0)

                # Chess score dominates; LM only breaks ties.
                total = base + 0.15 * lm_score

                if total > best_score:
                    best_score = total
                    best_move = move_uci

            self._append_history_once(fen, best_move)
            return best_move

        except Exception as e:
            print(f"[TransformerPlayer] get_move error: {e}")
            try:
                board = chess.Board(fen)
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    return None
                chosen = max(legal_moves, key=lambda m: self._combined_score(board, m)).uci()
                self._append_history_once(fen, chosen)
                return chosen
            except Exception:
                return None
