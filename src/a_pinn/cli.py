import sys
from importlib import import_module
from pathlib import Path
import yaml
import logging

from a_pinn.trainers.vanilla import VanillaTrainer
from a_pinn.utils.metrics import benchmark_burgers, benchmark_residual

PDE_MAP = {
    "burgers": "a_pinn.pdes.burgers",
    "allen": "a_pinn.pdes.allen_cahn",
    "shallow_water": "a_pinn.pdes.shallow_water",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_pde_module(key: str):
    try:
        mod = import_module(PDE_MAP[key])
    except KeyError:
        logging.error("Unknown PDE '%s'. Allowed: %s", key, list(PDE_MAP))
        sys.exit(1)
    if not hasattr(mod, "output_size"):
        logging.error("PDE module '%s' must define `output_size`", key)
        sys.exit(1)
    return mod


def _run_from_cfg(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text())
    pde_key = cfg.get("pde")
    pde_mod = _load_pde_module(pde_key)

    train_cfg = cfg.get("train", {})
    trainer = VanillaTrainer(
        name=pde_key,
        pde_module=pde_mod,
        num_domain=int(train_cfg.get("num_domain", 20_000)),
        num_bc=int(train_cfg.get("num_bc", 2_000)),
        hidden_layers=int(train_cfg.get("hidden_layers", 8)),
        hidden_units=int(train_cfg.get("hidden_units", 128)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        output_dir=Path(train_cfg.get("out_dir", "./experiments")),
    )
    logging.info("Starting training on %s", pde_key)
    n_iter = train_cfg.get("iterations", 20_000)
    trainer.train(iterations=n_iter)
    ckpt = trainer.output_dir / "model.ckpt"

    eval_cfg = cfg.get("eval", {})
    n_test = eval_cfg.get("n_test", 1000)
    if pde_key == "burgers":
        benchmark_burgers(pde_mod, ckpt, n=n_test)
    else:
        benchmark_residual(pde_mod, ckpt, n=n_test)


def _usage():
    print(
        "Usage:\n"
        "python -m a_pinn.cli run <config.yaml>\n"
        "python -m a_pinn.cli train <pde>\n"
        "python -m a_pinn.cli eval <pde> ckpt_path=<file>\n"
    )
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        _usage()

    cmd = sys.argv[1].lower()
    if cmd == "run":
        cfg = Path(sys.argv[2])
        if not cfg.exists():
            logging.error("Config file not found: %s", cfg)
            sys.exit(1)
        _run_from_cfg(cfg)
    elif cmd == "train":
        pde_key = sys.argv[2]
        mod = _load_pde_module(pde_key)
        VanillaTrainer(name=pde_key, pde_module=mod).train()
    elif cmd == "eval":
        pde_key = sys.argv[2]
        ckpt_path = Path(sys.argv[3].split("=", 1)[1])
        mod = _load_pde_module(pde_key)
        if pde_key == "burgers":
            benchmark_burgers(mod, ckpt_path)
        else:
            benchmark_residual(mod, ckpt_path)
    else:
        _usage()


if __name__ == "__main__":
    main()
