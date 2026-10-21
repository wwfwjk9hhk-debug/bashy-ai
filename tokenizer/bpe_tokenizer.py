import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

PYTHON_RE_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]+|[0-9]{1,3}|_|[^\s\w]+|\s+|."""

class BashyTokenizer:
    def __init__(self, tokenizer_dir: Optional[str] = None):
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.pattern = re.compile(PYTHON_RE_SPLIT_PATTERN)
        if tokenizer_dir is not None:
            self.load(tokenizer_dir)
    def load(self, tokenizer_dir: str):
        dir_path = Path(tokenizer_dir)
        with open(dir_path / "vocab.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        for s_id, v in raw.items():
            idx, bv = int(s_id), v.encode("latin1")
            self.vocab[idx], self.inverse_vocab[bv] = bv, idx
            try:
                dec = bv.decode("utf-8")
                if dec.startswith("<|") and dec.endswith("|>"):
                    self.special_tokens[dec], self.inverse_special_tokens[idx] = idx, dec
            except UnicodeDecodeError: pass
        with open(dir_path / "merges.txt", "r", encoding="utf-8") as f:
            rank = 0
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) == 2:
                    self.merges[(parts[0].encode("latin1"), parts[1].encode("latin1"))] = rank
                    rank += 1
    @property
    def vocab_size(self): return len(self.vocab)
    @property
    def pad_token_id(self): return self.special_tokens.get("<|pad|>", 0)
    @property
    def bos_token_id(self): return self.special_tokens.get("<|bos|>", 1)
    @property
    def eos_token_id(self): return self.special_tokens.get("<|eos|>", 2)
    def _merge(self, pieces: List[bytes]) -> List[bytes]:
        while len(pieces) >= 2:
            best_rank, best_idx, best_pair = float("inf"), -1, None
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank, best_idx, best_pair = rank, i, pair
            if best_pair is None: break
            pieces = pieces[:best_idx] + [best_pair[0] + best_pair[1]] + pieces[best_idx + 2:]
        return pieces
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        chunks = self.pattern.findall(text)
        token_ids = [self.bos_token_id] if add_special_tokens and self.bos_token_id else []
        for c in chunks:
            if c in self.special_tokens:
                token_ids.append(self.special_tokens[c])
            else:
                pieces = [bytes([b]) for b in c.encode("utf-8")]
                for p in self._merge(pieces):
                    token_ids.append(self.inverse_vocab.get(p, p[0]))
        if add_special_tokens and self.eos_token_id:
            token_ids.append(self.eos_token_id)
        return token_ids
    def decode(self, token_ids: List[int], skip_special: bool = False) -> str:
        ba = bytearray()
        for idx in token_ids:
            if idx in self.inverse_special_tokens:
                if not skip_special: ba.extend(self.inverse_special_tokens[idx].encode("utf-8"))
            elif idx in self.vocab:
                ba.extend(self.vocab[idx])
        return ba.decode("utf-8", errors="replace")
    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        p = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt: p += "<|im_start|>assistant\n"
        return p
