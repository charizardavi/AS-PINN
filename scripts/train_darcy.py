import hydra, torch, sys
from omegaconf import DictConfig
from neuralop.models import FNO  # <- already in your env
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.training import AdamW
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.utils import count_model_params
from pathlib import Path



@hydra.main(version_base=None, config_path="../conf", config_name="train_darcy")
def main(cfg: DictConfig):
    device = torch.device(cfg.device)

    # 1) Dataset ------------------------------------------------------------
    train_loader, test_loaders, data_proc = load_darcy_flow_small(
        n_train=cfg.data.n_train,
        batch_size=cfg.data.batch_size,
        test_resolutions=cfg.data.test_res,
        n_tests=cfg.data.n_tests,
        test_batch_sizes=[cfg.data.batch_size] * len(cfg.data.test_res),
        data_root="data/datasets",
    )
    data_proc = data_proc.to(device)

    model = FNO(
        n_modes=(cfg.model.modes_h, cfg.model.modes_w),
        hidden_channels=cfg.model.width,
        in_channels=1,
        out_channels=1,
        projection_channel_ratio=2,
    ).to(device)
    print(f"Model params: {count_model_params(model)}", file=sys.stderr)

    optimizer = AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.optim.t_max
    )

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss
    eval_losses = {"h1": h1loss, "l2": l2loss}

    trainer = Trainer(
        model=model,
        n_epochs=cfg.training.epochs,
        device=device,
        data_processor=data_proc,
        eval_interval=cfg.training.eval_every,
        use_distributed=False,
        wandb_log=False,
        verbose = True, 
    )
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    ckpt_path = Path(cfg.out_ckpt).expanduser()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)   # ← NEW
    torch.save({"model": model.state_dict()}, ckpt_path)
    print("Saved FNO weights to", ckpt_path.resolve())


if __name__ == "__main__":
    main()
