from __future__ import annotations
import deepxde as dde
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from ..nets.fnn import make_fnn


class VanillaTrainer:
    def __init__(
        self,
        name: str,
        pde_module,
        num_domain: int = 20_000,
        num_bc: int = 2_000,
        learning_rate: float = 1e-3,
        hidden_layers: int = 8,
        hidden_units: int = 128,
        output_dir: str | Path = "./experiments",
    ) -> None:
        dde.config.set_random_seed(21)

        self.name = name
        self.pde = pde_module
        self.geomtime = pde_module.geom_time()
        self.data = dde.data.TimePDE(
            self.geomtime,
            pde_module.residual,
            pde_module.conditions(self.geomtime),
            num_domain=num_domain,
            num_boundary=num_bc,
            num_initial=num_bc,
        )

        net = make_fnn([2] + [hidden_units] * hidden_layers + [pde_module.output_size])

        self.model = dde.Model(self.data, net)
        self.model.compile("adam", lr=learning_rate)
        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, iterations: int = 20_000) -> None:
        history = self.model.train(iterations=iterations)
        self.model.save(self.output_dir / "model.ckpt")
        
        df = pd.DataFrame(history.loss_history, columns=["loss", *history.metrics])

        df["iterations"] = range(1, len(df) + 1)
        df.to_csv(self.output_dir / "loss_history.csv", index=False)

        plt.figure()
        plt.plot(df["iterations"], df["loss"], label="Total Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss over Iterations")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / "loss_plot.png")
        plt.close()
