import json
import time
from datetime import datetime
from pathlib import Path

import torch


class Callback:
    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch, logs):
        pass


class SaveCheckpointCallback(Callback):
    def __init__(self, save_path):
        self.save_path = save_path

    def on_epoch_end(self, trainer, epoch, logs):
        filepath = self.save_path / f"checkpoint-{epoch:04d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": trainer.model.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
            },
            filepath,
        )
        print(f"[INFO] Checkpoint saved: {filepath}")


class LoggingCallback(Callback):
    def __init__(self, log_path, total_epochs):
        self.log_path = Path(log_path)
        self.total_epochs = total_epochs
        self.start_time = 0

    def on_epoch_begin(self, trainer, epoch):
        print(f"[INFO] Epoch {epoch}/{self.total_epochs}")
        self.start_time = time.time()

    def on_epoch_end(self, trainer, epoch, logs):
        log_entry = {
            "epoch": epoch,
            "train_loss": logs.get("train_loss", None),
            "val_metrics": logs.get("val_metrics", None),
            "epoch_time": time.time() - self.start_time,
            "logging_time": datetime.now().astimezone().isoformat(),
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")

        print(
            f"[INFO] Epoch {log_entry['epoch']}: "
            + f"Train loss = {log_entry['train_loss']:.4f}, "
            + f"Val metrics = {log_entry['val_metrics']}, "
            + f"Epoch time = {log_entry['epoch_time']:.2f}, "
            + f"Logging time = {log_entry['logging_time']}"
        )


class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        metric_fname,
        mode,
        patience=5,
        best_val_metric=None,
        best_epoch=None,
        last_wait=0,
    ):
        if mode == "min":
            self.is_better = lambda current: current < self.best_val_metric
        elif mode == "max":
            self.is_better = lambda current: current > self.best_val_metric
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'.")

        self.metric_fname = metric_fname
        self.patience = patience
        self.wait = last_wait
        self.best_epoch = best_epoch
        self.stopped_epoch = None

        if best_val_metric is not None:
            self.best_val_metric = best_val_metric
        else:
            self.best_val_metric = float("inf") if mode == "min" else -float("inf")

    def on_epoch_end(self, trainer, epoch, logs):
        val_loss = logs.get("val_metrics", None).get(self.metric_fname, None)

        if val_loss is None:
            print("[ERROR] Can not get validation loss")
            return

        if self.is_better(val_loss):
            self.best_val_metric = val_loss
            self.best_epoch = epoch
            self.wait = 0
            print("[INFO] New best model saved")
        else:
            self.wait += 1
            print(f"[INFO] No improvement for {self.wait} epoch(s)")

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            trainer.stop()
            print("[INFO] Early stopping triggered")

    def on_train_end(self, trainer):
        print(f"[INFO] Best model: {self.best_epoch}")
