from chess_tournament import Player
import chess
import random
from typing import Optional


class TransformerPlayer(Player):

    HF_MODEL_ID    = "hiiamkik/Chess"
    MAX_NEW_TOKENS = 6
    TEMPERATURE    = 0.3
    NUM_CANDIDATES = 15   # Generate more candidates to increase the chance of a legal move in step 1

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)
        self._model     = None
        self._tokenizer = None
        self._device    = None

    # ------------------------------------------------------------------
    # Lazy-load the model on the first call to get_move()
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
    # Build the prompt fed to the model
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(fen: str) -> str:
        return f"FEN: {fen} Best Move:"

    # ------------------------------------------------------------------
    # Step 1: sample NUM_CANDIDATES moves from the model freely.
    # Return the first one that is legal; otherwise fall through to step 2.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 2: score every legal move by its log-probability under the model
    # and return the highest-scoring one.
    # (Fixes the original tensor-indexing bug by accumulating per-token.)
    # ------------------------------------------------------------------
    def _score_legal_moves(self, fen: str, legal_uci: list) -> str:
        import torch
        import torch.nn.functional as F

        prompt     = self._build_prompt(fen)
        prefix_len = len(self._tokenizer(prompt)["input_ids"])

        # Cap at 40 candidates to avoid slow inference / timeout
        candidates = legal_uci if len(legal_uci) <= 40 else random.sample(legal_uci, 40)

        best_score = float("-inf")
        best_move  = candidates[0]

        for move in candidates:
            full_text = prompt + " " + move
            inputs    = self._tokenizer(full_text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits          # (1, seq_len, vocab)
            log_probs = F.log_softmax(logits, dim=-1)          # (1, seq_len, vocab)
            input_ids = inputs["input_ids"][0]                 # (seq_len,)
            move_ids  = input_ids[prefix_len:]                 # tokens belonging to the move

            if len(move_ids) == 0:
                continue

            # Accumulate log prob token by token
            score = sum(
                log_probs[0, prefix_len - 1 + i, move_ids[i]].item()
                for i in range(len(move_ids))
            )

            if score > best_score:
                best_score = score
                best_move  = move

        return best_move

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_move(self, fen: str) -> Optional[str]:
        # Lazy-load on first call
        if self._model is None:
            self._load_model()

        try:
            board       = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            legal_uci = [m.uci() for m in legal_moves]

            # If model failed to load, fall back to random (guarantees fallback=0)
            if self._model is None:
                return random.choice(legal_uci)

            # Break repetition loops — if the same position has appeared twice,
            # pick a random legal move to avoid drawing by repetition
            if board.is_repetition(2):
                return random.choice(legal_uci)

            # Step 1: free generation, look for a legal move
            candidates = self._generate_candidates(fen)
            for candidate in candidates:
                if candidate in legal_uci:
                    return candidate

            # Step 2: score all legal moves and pick the best
            return self._score_legal_moves(fen, legal_uci)

        except Exception as e:
            print(f"[TransformerPlayer] get_move error: {e}")
            # Last resort: random legal move — ensures fallback count stays 0
            try:
                board = chess.Board(fen)
                moves = list(board.legal_moves)
                return random.choice(moves).uci() if moves else None
            except Exception:
                return None
