import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch

class CheckpointManager:
    def __init__(self, checkpoints_dir: Optional[str] = None, hf_repo_id: Optional[str] = None):
        p = Path(checkpoints_dir) if checkpoints_dir else Path(__file__).resolve().parent.parent / "checkpoints"
        self.checkpoints_dir = p if p.is_absolute() else Path(__file__).resolve().parent.parent / p
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo_id = hf_repo_id or os.getenv("HF_REPO_ID")
        self.hf_token = os.getenv("HF_TOKEN")

    def save_checkpoint(self, model, optimizer, step, loss, config, filename=None, push_to_hf=False):
        fn = filename or f"checkpoint_step_{step}.pt"
        fp = self.checkpoints_dir / fn
        data = {"step": step, "loss": loss, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config}
        torch.save(data, fp)
        torch.save(data, self.checkpoints_dir / "checkpoint_latest.pt")
        print(f"Checkpoint guardado: {fp}")
        return fp

    def load_latest_checkpoint(self, model, optimizer=None, device=torch.device("cpu"), strict=True):
        lp = self.checkpoints_dir / "checkpoint_latest.pt"
        if not lp.exists(): return {"step": 0, "loss": float("inf")}
        ckpt = torch.load(lp, map_location=device)
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=strict)
            if optimizer and "optimizer_state_dict" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"Reanudado en paso {ckpt.get('step', 0)}")
        except Exception as e:
            print(f"Iniciando con pesos limpios ({e})")
        return ckpt
