from __future__ import annotations
import deepxde as dde
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch


from ..utils.autoscale import make_scaled_fnn

dde.backend.set_default_backend("pytorch")
dde.config.set_random_seed(21)


class AutoScaleTrainer:
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

        net = make_scaled_fnn(
            [pde_module.input_dim]
            + [hidden_units] * hidden_layers
            + [pde_module.output_size]
        )

        self.model = dde.Model(self.data, net)
        self.model.compile("adam", lr=learning_rate)

        net = self.model.net

        first_linear = net.fnn.linears[0]
        for param in first_linear.parameters():
            param.requires_grad = False

        import torch

        self.model.opt = torch.optim.Adam(
            [
                {
                    "params": net.scaler.parameters(),
                    "weight_decay": float(net.scaler.reg_lambda or 0.0),
                },
                {
                    "params": [
                        p
                        for name, p in net.fnn.named_parameters()
                        if not name.startswith("linears.0.")
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=learning_rate,
        )

        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, iterations: int = 20_000) -> None:
        losshistory, train_state = self.model.train(
            iterations=iterations, display_every=1
        )
        self.model.save(self.output_dir / "model_autoscale")

        total_loss = [sum(l) for l in losshistory.loss_train]
        steps = losshistory.steps

        df = pd.DataFrame({"step": steps, "loss": total_loss})
        df.to_csv(self.output_dir / "loss_history_autoscale.csv", index=False)

        print(df)

        plt.figure()
        plt.plot(df["step"], df["loss"], label="Total Loss (AutoScale)")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / "loss_plot_autoscale.png")
        plt.close()

        scaler = self.model.net.scaler
        alphas = torch.exp(scaler.log_alpha).detach().cpu().numpy()
        print("Learned alphas:", alphas)
