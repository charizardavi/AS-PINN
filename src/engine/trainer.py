import torch, time, hydra
from torch.utils.data import DataLoader
import random
import numpy as np


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device(cfg.device)

        random.seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        torch.manual_seed(cfg.train.seed)

        backbone = hydra.utils.instantiate(cfg.model)
        for p in backbone.parameters():
            p.requires_grad = False

        adapter = hydra.utils.instantiate(cfg.adapter)

        self.model = adapter.attach(backbone).to(self.device)
        self.task = hydra.utils.instantiate(cfg.task)

        trainables = filter(lambda p: p.requires_grad, self.model.parameters())
        self.opt = torch.optim.Adam(trainables, lr=cfg.train.lr)

    def fit(self):
        start = time.time()
        for step in range(self.cfg.train.max_steps):
            k_fields = self.task.sample_batch(self.cfg.train.batch_size)
            preds = self.model(k_fields)
            loss = self.task.loss(preds, k_fields)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if step == 0:
                self.start_time = time.time()
                self.start_mem = torch.cuda.max_memory_allocated(self.cfg.device)

            if step % self.cfg.train.log_every == 0:
                elapsed = time.time() - self.start_time
                peak_mb = (
                    torch.cuda.max_memory_allocated(self.cfg.device) - self.start_mem
                ) / 1e6
                print(
                    f"{step:05d} | loss={loss.item():.2e} | time={elapsed:.1f}s | peak_mem={peak_mb:.1f} MB"
                )

            if loss.item() < self.cfg.train.target_loss:
                break

        wall_clock = time.time() - start
        print(f"Finished in {wall_clock/60:.2f} min")

        self.model.eval()
        
        with torch.no_grad():
            k_eval = self.task.sample_batch(self.cfg.train.batch_size)
            preds_eval = self.model(k_eval)
            eval_loss = self.task.loss(preds_eval, k_eval).item()

        print(f"Final eval loss = {eval_loss:.2e}")
        self.model.train()
