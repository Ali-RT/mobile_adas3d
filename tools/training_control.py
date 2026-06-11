from typing import Optional


class EarlyStopping:
    def __init__(
        self,
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        start_epoch: int = 1,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch

        self.best: Optional[float] = None
        self.num_bad_epochs = 0
        self.should_stop = False
        self.stop_reason = ""

    def is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True

        if self.mode == "min":
            return current < self.best - self.min_delta

        return current > self.best + self.min_delta

    def step(self, current: float, epoch: int) -> bool:
        if epoch < self.start_epoch:
            if self.best is None or self.is_improvement(current):
                self.best = current
            return False

        if self.is_improvement(current):
            self.best = current
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
            self.stop_reason = (
                f"Early stopping triggered at epoch {epoch}. "
                f"No improvement for {self.num_bad_epochs} epochs. "
                f"Best metric: {self.best:.6f}, current metric: {current:.6f}."
            )

        return self.should_stop


def get_current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])