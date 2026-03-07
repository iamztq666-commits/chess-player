from chess_tournament import Player
import chess
import random
from typing import Optional

class TransformerPlayer(Player):

    HF_MODEL_ID = "hiiamkik/Chess"

    MAX_NEW_TOKENS   = 6
    TEMPERATURE      = 0.3
    NUM_CANDIDATES   = 10

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)
        self._model     = None
        self._tokenizer = None
        self._device    = None

    def _load_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[TransformerPlayer] Loading {self.HF_MODEL_ID}...")

            self._tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)
            self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.HF_MODEL_ID,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self._model.eval()
            print(f"[TransformerPlayer] Loaded on {self._device}")

        except Exception as e:
            print(f"[TransformerPlayer] Failed to load: {e}")
            self._model = None

    @staticmethod
    def _build_prompt(fen: str) -> str:
        return f"FEN: {fen} Move:"

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
        best_score = float("-inf")
        best_move  = legal_uci[0]
        candidates = legal_uci if len(legal_uci) <= 40 else random.sample(legal_uci, 40)
        for move in candidates:
            full_text  = prompt + " " + move
            inputs     = self._tokenizer(full_text, return_tensors="pt").to(self._device)
            prefix_len = len(self._tokenizer(prompt)["input_ids"])
            with torch.no_grad():
                logits = self._model(**inputs).logits
            log_probs = F.log_softmax(logits, dim=-1)
            input_ids = inputs["input_ids"][0]
            move_ids  = input_ids[prefix_len:]
            if len(move_ids) == 0:
                continue
            score = log_probs[0, prefix_len-1:prefix_len-1+len(move_ids), move_ids].sum().item()
            if score > best_score:
                best_score = score
                best_move  = move
        return best_move

    def get_move(self, fen: str) -> Optional[str]:
        if self._model is None:
            self._load_model()
        try:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            legal_uci = [m.uci() for m in legal_moves]
            if self._model is None:
                return random.choice(legal_uci)
            candidates = self._generate_candidates(fen)
            for candidate in candidates:
                if candidate in legal_uci:
                    return candidate
            return self._score_legal_moves(fen, legal_uci)
        except Exception as e:
            print(f"[TransformerPlayer] Error: {e}")
            try:
                board = chess.Board(fen)
                moves = list(board.legal_moves)
                return random.choice(moves).uci() if moves else None
            except:
                return None
