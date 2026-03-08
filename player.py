from chess_tournament import Player
import chess
import random
from typing import Optional

class TransformerPlayerV3_1(Player):

    HF_MODEL_ID    = "hiiamkik/Chess-1.7B-v2"
    MAX_NEW_TOKENS = 8
    TEMPERATURE    = 0.25
    NUM_CANDIDATES = 20

    def __init__(self, name: str = "TransformerPlayerV3_1"):
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

    def _maybe_reset(self, fen: str) -> chess.Board:
        board = chess.Board(fen)
        if board.fullmove_number <= 1 and board.turn == chess.WHITE:
            self._move_history = []
            self._pos_history  = {}
        key = self._position_key_from_board(board)
        self._pos_history[key] = self._pos_history.get(key, 0) + 1
        return board

    def _load_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[TransformerPlayerV3_1] Loading {self.HF_MODEL_ID} on {self._device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.HF_MODEL_ID,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self._model.eval()
            print(f"[TransformerPlayerV3_1] Ready on {self._device}")
        except Exception as e:
            print(f"[TransformerPlayerV3_1] Load failed: {e}. Falling back to CPU.")
            self._device = "cpu"
            self._model = None

    def _recent_moves_for_prompt(self, k: int = 12) -> list[str]:
        hist = self._move_history[-k:]
        if len(hist) >= 4:
            tail = hist[-4:]
            if tail[0] == tail[2] and tail[1] == tail[3] and tail[0] != tail[1]:
                return hist[:-4]
        return hist

    def _build_prompt(self, fen: str) -> str:
        recent = self._recent_moves_for_prompt()
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
        piece_value = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}

        moving_piece = board.piece_at(move.from_square)
        captured_piece = None
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece is None and board.ep_square == move.to_square:
                captured_piece = chess.Piece(chess.PAWN, not board.turn)
        if captured_piece:
            score += piece_value.get(captured_piece.piece_type,0) * 0.8  # 捕获奖励
        if move.promotion:
            score += piece_value.get(move.promotion,0)  # 升变奖励
        if moving_piece and moving_piece.piece_type == chess.PAWN:
            score += 0.25  # 兵推进奖励

        board.push(move)

        # 王翼安全 + castling 奖励
        if board.is_castling(move):
            score += 0.3

        # 检查/将死奖励
        if board.is_checkmate():
            score += 1000
        elif board.is_check():
            score += 1.0

        # 中心控制奖励
        for sq in [chess.D4, chess.D5, chess.E4, chess.E5]:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                score += 0.15

        # 威胁对方子力奖励
        for sq in board.attacks(move.to_square):
            piece = board.piece_at(sq)
            if piece and piece.color != board.turn:
                score += 0.1

        # 避免重复局面
        next_key = self._position_key_from_board(board)
        repeat_count = self._pos_history.get(next_key,0)
        score -= repeat_count * 5.0

        board.pop()

        # 避免退回走法 / 循环
        if self._is_immediate_backtrack(move):
            score -= 12.0
        if self._creates_two_step_cycle(move.uci()):
            score -= 10.0

        # 非捕获大子力移动惩罚
        if moving_piece and moving_piece.piece_type != chess.PAWN and not captured_piece:
            score -= 0.4
        if moving_piece and moving_piece.piece_type in (chess.KING, chess.ROOK):
            score -= 0.35

        return score

    def _heuristic_top_n(self, board: chess.Board, legal_moves: list, n: int = 12):
        scored = [(m, self._heuristic_score(board, m)) for m in legal_moves]
        scored.sort(key=lambda x:x[1], reverse=True)
        return [m.uci() for m,_ in scored[:n]]

    def _generate_candidates(self, fen: str) -> list:
        import torch
        if self._model is None:
            return []
        prompt = self._build_prompt(fen)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=self.TEMPERATURE,
                num_return_sequences=self.NUM_CANDIDATES,
                pad_token_id=self._tokenizer.eos_token_id
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
        if self._model is None:
            return random.choice(legal_uci)
        prompt = self._build_prompt(fen)
        prefix_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_ids)
        best_score = float("-inf")
        best_move  = legal_uci[0]
        board = chess.Board(fen)

        # 启发式剪枝 top 10~12
        legal_moves_objs = [chess.Move.from_uci(m) for m in legal_uci]
        top_moves_uci = self._heuristic_top_n(board, legal_moves_objs, n=min(12,len(legal_moves_objs)))

        for move in top_moves_uci:
            full_text = prompt + " " + move
            inputs = self._tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            log_probs = F.log_softmax(logits, dim=-1)
            input_ids = inputs["input_ids"][0]
            move_ids  = input_ids[prefix_len:]
            if len(move_ids) == 0:
                continue
            lm_score = sum(log_probs[0, prefix_len-1+i, move_ids[i]].item() for i in range(len(move_ids)))
            h_score = self._heuristic_score(board, chess.Move.from_uci(move))
            total_score = lm_score + 0.9*h_score
            if total_score > best_score:
                best_score = total_score
                best_move = move
        return best_move

    def get_move(self, fen: str) -> Optional[str]:
        if self._model is None:
            self._load_model()
        board = self._maybe_reset(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # 过滤退回和循环走法
        filtered_moves = [m for m in legal_moves
                          if not self._is_immediate_backtrack(m) and not self._creates_two_step_cycle(m.uci())]
        if filtered_moves:
            legal_moves = filtered_moves
        legal_uci = [m.uci() for m in legal_moves]

        chosen = self._score_legal_moves(fen, legal_uci)
        self._move_history.append(chosen)
        return chosen        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[TransformerPlayerV3_1] Loading {self.HF_MODEL_ID} on {self._device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.HF_MODEL_ID,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self._model.eval()
            print(f"[TransformerPlayerV3_1] Ready on {self._device}")
        except Exception as e:
            print(f"[TransformerPlayerV3_1] Load failed: {e}")
            self._model = None

    def _recent_moves_for_prompt(self, k: int = 12) -> list[str]:
        hist = self._move_history[-k:]
        if len(hist) >= 4:
            tail = hist[-4:]
            if tail[0] == tail[2] and tail[1] == tail[3] and tail[0] != tail[1]:
                return hist[:-4]
        return hist

    def _build_prompt(self, fen: str) -> str:
        recent = self._recent_moves_for_prompt()
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
        piece_value = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
        moving_piece = board.piece_at(move.from_square)
        captured_piece = None
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece is None and board.ep_square == move.to_square:
                captured_piece = chess.Piece(chess.PAWN, not board.turn)
        if captured_piece:
            score += piece_value.get(captured_piece.piece_type,0)
        if move.promotion:
            score += piece_value.get(move.promotion,0)
        if moving_piece and moving_piece.piece_type == chess.PAWN:
            score += 0.25
        board.push(move)
        if board.is_check():
            score += 0.5
        # 中心控制奖励
        for sq in [chess.D4, chess.D5, chess.E4, chess.E5]:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                score += 0.15
        # 王翼安全
        if board.is_castling(move):
            score += 0.3
        next_key = self._position_key_from_board(board)
        repeat_count = self._pos_history.get(next_key,0)
        score -= repeat_count*4.0
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

    def _heuristic_top_n(self, board: chess.Board, legal_moves: list, n: int = 10):
        scored = [(m, self._heuristic_score(board, m)) for m in legal_moves]
        scored.sort(key=lambda x:x[1], reverse=True)
        return [m.uci() for m,_ in scored[:n]]

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
                pad_token_id=self._tokenizer.eos_token_id
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
            inputs = self._tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            log_probs = F.log_softmax(logits, dim=-1)
            input_ids = inputs["input_ids"][0]
            move_ids  = input_ids[prefix_len:]
            if len(move_ids) == 0:
                continue
            lm_score = sum(log_probs[0, prefix_len-1+i, move_ids[i]].item() for i in range(len(move_ids)))
            h_score = self._heuristic_score(board, chess.Move.from_uci(move))
            total_score = lm_score + 0.9*h_score
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
            if self._pos_history.get(pos_key,0) >= 2:
                top = self._heuristic_top_n(board, legal_moves, n=6)
                chosen = to
