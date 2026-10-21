import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

PYTHON_RE_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]+|[0-9]{1,3}|_|[^\s\w]+|\s+|."""

class BPETrainer:
    SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|im_start|>", "<|im_end|>"]
    def __init__(self, target_vocab_size: int = 16384):
        self.target_vocab_size = target_vocab_size
        self.merges: List[Tuple[bytes, bytes]] = []
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
    def _get_stats(self, words: Dict[Tuple[bytes, ...], int]) -> Counter:
        pairs = Counter()
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    def _merge_word(self, word: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
        new_word = []
        i = 0
        p0, p1 = pair
        while i < len(word):
            if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                new_word.append(p0 + p1)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
    def train_from_text(self, text: str, min_frequency: int = 2, verbose: bool = True) -> Dict[str, Any]:
        chunks = re.findall(PYTHON_RE_SPLIT_PATTERN, text)
        word_counts = Counter(chunks)
        words: Dict[Tuple[bytes, ...], int] = {tuple(bytes([b]) for b in c.encode("utf-8")): cnt for c, cnt in word_counts.items()}
        current_id = 0
        self.vocab, self.inverse_vocab = {}, {}
        for spec in self.SPECIAL_TOKENS:
            sb = spec.encode("utf-8")
            self.vocab[current_id], self.inverse_vocab[sb] = sb, current_id
            current_id += 1
        for b in range(256):
            bb = bytes([b])
            self.vocab[current_id], self.inverse_vocab[bb] = bb, current_id
            current_id += 1
        num_merges = self.target_vocab_size - len(self.vocab)
        self.merges = []
        for step in range(1, num_merges + 1):
            pairs = self._get_stats(words)
            if not pairs: break
            best_pair, freq = pairs.most_common(1)[0]
            if freq < min_frequency: break
            self.merges.append(best_pair)
            merged = best_pair[0] + best_pair[1]
            self.vocab[current_id], self.inverse_vocab[merged] = merged, current_id
            current_id += 1
            words = {self._merge_word(w, best_pair): cnt for w, cnt in words.items()}
        return {"vocab_size": len(self.vocab), "merges_count": len(self.merges)}
    def save(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({idx: b.decode("latin1") for idx, b in self.vocab.items()}, f, indent=2)
        with open(out / "merges.txt", "w", encoding="utf-8") as f:
            for p0, p1 in self.merges:
                f.write(f"{p0.decode('latin1')} {p1.decode('latin1')}\n")
