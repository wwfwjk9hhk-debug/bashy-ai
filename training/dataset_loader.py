from pathlib import Path
from typing import List, Generator, Tuple
import torch
from tokenizer.bpe_tokenizer import BashyTokenizer

class TextBlockDataset:
    def __init__(self, text_files_or_dir: str, tokenizer: BashyTokenizer, seq_len: int = 512, batch_size: int = 8):
        self.tokenizer, self.seq_len, self.batch_size = tokenizer, seq_len, batch_size
        p = Path(text_files_or_dir)
        self.files = list(p.glob("**/*.txt")) if p.is_dir() else ([p] if p.exists() else [])
    def get_token_stream(self) -> Generator[int, None, None]:
        eos = self.tokenizer.eos_token_id
        for fp in self.files:
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        l = line.strip()
                        if l:
                            for t in self.tokenizer.encode(l): yield t
                            yield eos
            except Exception: pass
    def get_batches(self, device: torch.device) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        buf, bx, by = [], [], []
        for t in self.get_token_stream():
            buf.append(t)
            if len(buf) == self.seq_len + 1:
                bx.append(buf[:-1]); by.append(buf[1:]); buf = []
                if len(bx) == self.batch_size:
                    yield torch.tensor(bx, dtype=torch.long, device=device), torch.tensor(by, dtype=torch.long, device=device)
                    bx, by = [], []
