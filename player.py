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
        self._move_history = []        # 走法历史
        self._pos_history  = {}        # 局面出现次数 {fen: count}
        self._prev_fen     = None      # 上一局面，用于检测新对局

    # ── 新对局检测：初始局面出现说明新游戏开始 ──────────────────────
    def _maybe_reset(self, fen: str):
        board = chess.Board(fen)
        if board.fullmove_number <= 1 and board.turn == chess.WHITE:
            self._move_history = []
            self._pos_history  = {}
        self._pos_history[fen] = self._pos_history.get(fen, 0) + 1

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

    def _build_prompt(self, fen: str) -> str:
        if self._move_history:
            history_str = " ".join(self._move_history[-10:])
            return f"FEN: {fen} Moves: {history_str} Best Move:"
        return f"FEN: {fen} Best Move:"

    # ── Heuristic打分：优先吃子、升变、将军 ─────────────────────────
    def _heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0.0
        piece_value = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        # 吃子奖励
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                score += piece_value.get(captured.piece_type, 0)
        # 升变奖励
        if move.promotion:
            score += piece_value.get(move.promotion, 0)
        # 将军奖励
        board.push(move)
        if board.is_check():
            score += 0.5
        board.pop()
        # 反复横跳惩罚：走完后的局面如果出现过，扣分
        board.push(move)
        next_fen = board.fen()
        board.pop()
        repeat_count = self._pos_history.get(next_fen, 0)
        score -= repeat_count * 2.0
        return score

    # ── 筛选候选：用heuristic排序，取top-N ──────────────────────────
    def _heuristic_top_n(self, board: chess.Board, legal_moves: list, n: int = 40):
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

        prompt     = self._build_prompt(fen)
        prefix_len = len(self._tokenizer(prompt)["input_ids"])

        best_score = float("-inf")
        best_move  = legal_uci[0]

        for move in legal_uci:
            full_text = prompt + " " + move
            inputs    = self._tokenizer(full_text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            log_probs = F.log_softmax(logits, dim=-1)
            input_ids = inputs["input_ids"][0]
            move_ids  = input_ids[prefix_len:]
            if len(move_ids) == 0:
                continue
            score = sum(
                log_probs[0, prefix_len - 1 + i, move_ids[i]].item()
                for i in range(len(move_ids))
            )
            if score > best_score:
                best_score = score
                best_move  = move

        return best_move

    def get_move(self, fen: str) -> Optional[str]:
        if self._model is None:
            self._load_model()

        try:
            self._maybe_reset(fen)
            board       = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            legal_uci = [m.uci() for m in legal_moves]

            if self._model is None:
                return random.choice(legal_uci)

            # 重复局面：用heuristic选最佳，避开重复
            if self._pos_history.get(fen, 0) >= 2:
                top = self._heuristic_top_n(board, legal_moves, n=5)
                chosen = top[0] if top else random.choice(legal_uci)
                self._move_history.append(chosen)
                return chosen

            # Step 1: 生成候选，在heuristic top-40里找合法的
            top40 = self._heuristic_top_n(board, legal_moves, n=40)
            candidates = self._generate_candidates(fen)
            chosen = None
            for candidate in candidates:
                if candidate in top40:
                    chosen = candidate
                    break

            # Step 2: 对top40打分选最优
            if chosen is None:
                chosen = self._score_legal_moves(fen, top40)

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
