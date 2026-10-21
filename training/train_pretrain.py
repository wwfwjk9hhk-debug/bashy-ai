import math, time, sys, argparse
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
import torch, torch.optim as optim
from models.bashy_v1 import BashyV1LM, BashyConfig
from tokenizer.bpe_tokenizer import BashyTokenizer
from training.dataset_loader import TextBlockDataset
from training.checkpoint_manager import CheckpointManager

def pretrain():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer/bashy_bpe")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    tokenizer = BashyTokenizer(args.tokenizer_dir)
    config = BashyConfig(vocab_size=tokenizer.vocab_size, embed_dim=256, num_layers=6, num_heads=8, num_kv_heads=2, max_seq_len=args.seq_len)
    model = BashyV1LM(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    chk = CheckpointManager(checkpoints_dir="checkpoints")
    dataset = TextBlockDataset(args.data_dir, tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
    model.train()
    step, gen = 0, dataset.get_batches(device)
    start_t = time.time()
    while step < args.max_steps:
        try: x, y = next(gen)
        except StopIteration:
            gen = dataset.get_batches(device)
            try: x, y = next(gen)
            except StopIteration:
                x = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.seq_len), device=device)
                y = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.seq_len), device=device)
        step += 1
        optimizer.zero_grad()
        _, loss, metrics, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == 1 or step == args.max_steps:
            print(f"Paso {step}/{args.max_steps} | Loss: {loss.item():.4f}")
    chk.save_checkpoint(model, optimizer, step, loss.item(), config.__dict__, filename="bashy_pretrain_final.pt")
    print("Pre-entrenamiento completado.")
if __name__ == "__main__": pretrain()
