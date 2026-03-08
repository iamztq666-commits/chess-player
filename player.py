from chess_tournament import Player
import chess
import random
from typing import Optional


class TransformerPlayer(Player):

    HF_MODEL_ID    = "hiiamkik/Chess-1.7B-v2"
    MAX_NEW_TOKENS = 6
    TEMPERATURE    = 0.3
    NUM_CANDIDATES = 15

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)
        self._model        = None
        self._tokenizer    = None
        self._device       = None
        self._move_history = []
        self._pos_history  = {}

    def _position_key_from_board(self, board: chess.Board) -> str:
        ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
        turn = "w" if board.turn == chess.WHITE else "b"
        return f"{board.board_fen()} {turn} {board.castling_xfen()} {ep}"

    def _position_key_from_fen(self, fen: str) -> str:
        return self._position_key_from_board(chess.Board(fen))

    def _maybe_reset(self, fen: str):
        board = chess.Board(fen)
        if board.fullmove_number <= 1 and board.turn == chess.WHITE:
            self._move_history = []
            self._pos_history  = {}

        key = self._position_key_from_board(board)
        self._pos_history[key] = self._pos_history.get(key, 0) + 1

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

    def _recent_moves_for_prompt(self, k: int = 10) -> list[str]:
        hist = self._move_history[-k:]
        if len(hist) >= 4:
            tail = hist[-4:]
            if tail[0] == tail[2] and tail[1] == tail[3] and tail[0] != tail[1]:
                return hist[:-4]
        return hist

    def _build_prompt(self, fen: str) -> str:
        recent = self._recent_moves_for_prompt(10)
        if recent:
            history_str = " ".join(recent)
            return f"FEN: {fen} Moves: {history_str} Best Move:"
        return f"FEN: {fen} Best Move:"

    def _is_immediate_backtrack(self, move: chess.Move) -> bool:
        if not self._move_history:
            return False
        try:
            last = chess.Move.from_uci(self._move_history[-1])
            return move.from_square == last.to_square and move.to_square == last.from_square
        except Exception:
            return False

    def _creates_two_step_cycle(self, move_uci: str) -> bool:
        return len(self._move_history) >= 2 and self._move_history[-2] == move_uci

    def _heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0.0
        piece_value = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }

        moving_piece = board.piece_at(move.from_square)
        captured_piece = None

        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece is None and board.ep_square == move.to_square:
                captured_piece = chess.Piece(chess.PAWN, not board.turn)

        if captured_piece:
            score += piece_value.get(captured_piece.piece_type, 0)

        if move.promotion:
            score += piece_value.get(move.promotion, 0)

        if moving_piece and moving_piece.piece_type == chess.PAWN:
            score += 0.25

        board.push(move)

        if board.is_check():
            score += 0.5

        next_key = self._position_key_from_board(board)
        repeat_count = self._pos_history.get(next_key, 0)
        score -= repeat_count * 4.0

        board.pop()

        if self._is_immediate_backtrack(move):
            score -= 12.0

        if self._creates_two_step_cycle(move.uci()):
            score -= 10.0

        if moving_piece and moving_piece.piece_type != chess.PAWN and not captured_piece:
            score -= 0.4

        if moving_piece and moving_piece.piece_type in (chess.KING, chess.ROOK):
            score -= 0.35

        return score

    def _heuristic_top_n(self, board: chess.Board, legal_moves: list, n: int = 8):
        scored = [(m, self._heuristic_score(board, m)) for m in legal_moves]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m.uci() for m, _ in scored[:n]]

    def _generate_candidates(self, fen: str) -> list:
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
        candidates = []
        for out in outputs:
            text = self._tokenizer.decode(out, skip_special_tokens=True)
            if "Move:" in text:
                after_move = text.split("Move:")[-1].strip()
                if after_move:
                    move_str = after_move.split()[0].strip().lower()
                    candidates.append(move_str)
        return candidates

    def _score_legal_moves(self, fen: str, legal_uci: list) -> str:
        import torch
        import torch.nn.functional as F

        prompt = self._build_prompt(fen)
        prefix_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_ids)

        best_score = float("-inf")
        best_move  = legal_uci[0]
        board = chess.Board(fen)

        for move in legal_uci:
            full_text = prompt + " " + move
            inputs = self._tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits

            log_probs = F.log_softmax(logits, dim=-1)
            input_ids = inputs["input_ids"][0]
            move_ids  = input_ids[prefix_len:]

            if len(move_ids) == 0:
                continue

            lm_score = sum(
                log_probs[0, prefix_len - 1 + i, move_ids[i]].item()
                for i in range(len(move_ids))
            )

            h_score = self._heuristic_score(board, chess.Move.from_uci(move))
            total_score = lm_score + 0.8 * h_score

            if total_score > best_score:
                best_score = total_score
                best_move = move

        return best_move

    def get_move(self, fen: str) -> Optional[str]:
        if self._model is None:
            self._load_model()

        try:
            self._maybe_reset(fen)
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None

            # 硬过滤循环招法
            filtered_moves = []
            for m in legal_moves:
                u = m.uci()
                if self._is_immediate_backtrack(m):
                    continue
                if self._creates_two_step_cycle(u):
                    continue
                filtered_moves.append(m)

            if filtered_moves:
                legal_moves = filtered_moves

            legal_uci = [m.uci() for m in legal_moves]

            if self._model is None:
                chosen = random.choice(legal_uci)
                self._move_history.append(chosen)
                return chosen

            pos_key = self._position_key_from_board(board)

            if self._pos_history.get(pos_key, 0) >= 2:
                top = self._heuristic_top_n(board, legal_moves, n=5)
                chosen = top[0] if top else random.choice(legal_uci)
                self._move_history.append(chosen)
                return chosen

            top_moves = self._heuristic_top_n(board, legal_moves, n=8)
            candidates = self._generate_candidates(fen)

            candidate_moves = []
            for c in candidates:
                if c in legal_uci and c in top_moves:
                    candidate_moves.append(c)

            if candidate_moves:
                candidate_moves = sorted(
                    candidate_moves,
                    key=lambda u: self._heuristic_score(board, chess.Move.from_uci(u)),
                    reverse=True
                )
                chosen = candidate_moves[0]
            else:
                chosen = self._score_legal_moves(fen, top_moves)

            self._move_history.append(chosen)
            return chosen

        except Exception as e:
            print(f"[TransformerPlayer] get_move error: {e}")
            try:
                board = chess.Board(fen)
                moves = list(board.legal_moves)
                chosen = random.choice(moves).uci() if moves else None
                if chosen:
                    self._move_history.append(chosen)
                return chosen
            except Exception:
                return None
