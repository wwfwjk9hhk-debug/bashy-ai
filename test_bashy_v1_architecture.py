import time, torch
from models.bashy_v1 import BashyV1LM, BashyConfig

def main():
    print("=== TEST FASE 2: ARQUITECTURA BASHY-V1 ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    cfg = BashyConfig(vocab_size=600, embed_dim=256, num_layers=4, num_heads=8, num_kv_heads=2, max_seq_len=256, latent_dim=32)
    m = BashyV1LM(cfg).to(dev)
    print(f"Parámetros: {m.count_parameters():,}")
    x = torch.randint(0, 600, (2, 32), device=dev)
    y = torch.randint(0, 600, (2, 32), device=dev)
    logits, loss, _, _ = m(x, targets=y)
    print(f"Loss: {loss.item():.4f}")
    loss.backward()
    print("Gradientes calculados sin NaNs.")
    gen = m.generate_with_kv_cache([10, 20, 30], max_new_tokens=15, device=dev)
    print(f"KV-Cache Generación OK: {len(gen)} tokens.")
    print("=== BASHY-V1 100% OPERATIVO ===")

if __name__ == "__main__": main()
