import json, time, sys, argparse
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
import torch, torch.optim as optim
from models.bashy_v1 import BashyV1LM, BashyConfig
from tokenizer.bpe_tokenizer import BashyTokenizer
from training.checkpoint_manager import CheckpointManager

SAMPLE_INSTRUCTIONS = [
    {"messages": [{"role": "user", "content": "¿Quién eres?"}, {"role": "assistant", "content": "Soy Bashy, un modelo de lenguaje híbrido neuro-probabilístico creado para razonar y conversar en español."}]},
    {"messages": [{"role": "user", "content": "¿Qué es Neuvalia?"}, {"role": "assistant", "content": "Neuvalia es un mundo virtual consciente donde conviven entidades inteligentes, artesanos digitales y exploradores."}]},
    {"messages": [{"role": "user", "content": "Dame un consejo para aprender a programar."}, {"role": "assistant", "content": "Practica todos los días construyendo proyectos reales desde cero, aprende a leer los errores con paciencia y entiende los fundamentos antes de usar librerías complejas."}]}
] * 15

def prepare_sft_batch(dialogues, tokenizer, max_len=128, device=None):
    batch_in, batch_tgt = [], []
    for item in dialogues:
        msgs = item["messages"]
        tokens = tokenizer.encode(tokenizer.apply_chat_template(msgs, add_generation_prompt=False))
        prompt_len = len(tokenizer.encode(tokenizer.apply_chat_template([msgs[0]], add_generation_prompt=True)))
        tokens = tokens[:max_len]
        inp, tgt = tokens[:-1], tokens[1:]
        masked_tgt = [-100 if i < prompt_len - 1 else t for i, t in enumerate(tgt)]
        while len(inp) < max_len:
            inp.append(tokenizer.pad_token_id); masked_tgt.append(-100)
        batch_in.append(inp); batch_tgt.append(masked_tgt)
    return torch.tensor(batch_in, dtype=torch.long, device=device), torch.tensor(batch_tgt, dtype=torch.long, device=device)

def train_sft():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer/bashy_bpe")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    tokenizer = BashyTokenizer(args.tokenizer_dir)
    config = BashyConfig(vocab_size=tokenizer.vocab_size, embed_dim=256, num_layers=4, num_heads=8, num_kv_heads=2, max_seq_len=256)
    model = BashyV1LM(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    chk = CheckpointManager(checkpoints_dir="checkpoints")
    chk.load_latest_checkpoint(model, optimizer=optimizer, device=device, strict=False)
    x_sft, y_sft = prepare_sft_batch(SAMPLE_INSTRUCTIONS, tokenizer, max_len=128, device=device)
    model.train()
    for ep in range(1, args.epochs + 1):
        optimizer.zero_grad()
        _, loss, _, _ = model(x_sft, targets=y_sft)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ep % 10 == 0 or ep == 1: print(f"Época {ep}/{args.epochs} | Loss: {loss.item():.4f}")
    chk.save_checkpoint(model, optimizer, args.epochs, loss.item(), config.__dict__, filename="bashy_instruct_v1.pt")
    print("SFT completado y guardado.")

if __name__ == "__main__": train_sft()
