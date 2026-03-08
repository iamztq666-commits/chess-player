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
        self._model     = None
        self._tokenizer = None
        self._device    = None
        self._move_history = []   # 记录历史走法
        self._last_fen  = None

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
        # 加入最近10步历史
        if self._move_history:
            recent = self._move_history[-10:]
            history_str = " ".join(recent)
            return f"FEN: {fen} Moves: {history_str} Best Move:"
        return f"FEN: {fen} Best Move:"

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

        candidates = legal_uci if len(legal_uci) <= 40 else random.sample(legal_uci, 40)

        best_score = float("-inf")
        best_move  = candidates[0]

        for move in candidates:
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
            board       = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            legal_uci = [m.uci() for m in legal_moves]

            if self._model is None:
                return random.choice(legal_uci)

            if board.is_repetition(2):
                captures = [m.uci() for m in legal_moves if board.is_capture(m)]
                if captures:
                    return random.choice(captures)
                return random.choice(legal_uci)

            # Step 1: 自由生成
            candidates = self._generate_candidates(fen)
            chosen = None
            for candidate in candidates:
                if candidate in legal_uci:
                    chosen = candidate
                    break

            # Step 2: 打分选最优
            if chosen is None:
                chosen = self._score_legal_moves(fen, legal_uci)

            # 记录走法到历史
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
