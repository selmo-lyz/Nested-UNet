import numpy as np
import torch
from tqdm import tqdm


class NestedUNetTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        callbacks=[],
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.last_epoch = 0
        self.stop_fit = False

    def stop(self):
        self.stop_fit = True

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.last_epoch = checkpoint["epoch"]

    def _get_metric_name(self, metric):
        if hasattr(metric, "__name__"):
            return metric.__name__
        elif hasattr(metric, "__class__"):
            return metric.__class__.__name__
        return str(metric)

    def _run_epochs(self, dataloader, is_train=True, metric_fns=[]):
        self.model.train() if is_train else self.model.eval()
        running_loss = 0.0
        running_metrics = [0 for i in range(len(metric_fns))]
        total_samples = 0

        for batch in tqdm(dataloader, ncols=80):
            inputs = batch["image"].to(self.device)
            labels = batch["mask"].to(self.device)
            batch_size = inputs.size(0)

            with torch.set_grad_enabled(is_train):
                outputs = self.model(inputs)
                outputs = torch.stack(outputs, dim=0)
                preds = torch.mean(outputs, dim=0)

                if is_train:
                    loss = self.loss_fn(preds, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item() * batch_size
                else:
                    for idx, metric_fn in enumerate(metric_fns):
                        metric = metric_fn(preds, labels)
                        running_metrics[idx] += metric * batch_size

            total_samples += batch_size

        avg_loss = running_loss / total_samples
        avg_metrics = {
            self._get_metric_name(metric_fn): float(running_metric) / total_samples
            for (metric_fn, running_metric) in zip(metric_fns, running_metrics)
        }
        return avg_loss if len(metric_fns) == 0 else avg_metrics

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs,
        callbacks,
        metric_fns=[],
    ):
        if isinstance(metric_fns, list):
            metric_fns.append(self.loss_fn)

        for cb in callbacks:
            cb.on_train_begin(self)

        for epoch in range(self.last_epoch + 1, num_epochs + self.last_epoch + 1):
            for cb in callbacks:
                cb.on_epoch_begin(self, epoch)

            train_loss = self._run_epochs(train_dataloader, is_train=True)
            val_metrics = self._run_epochs(
                val_dataloader,
                is_train=False,
                metric_fns=metric_fns,
            )

            logs = {
                "train_loss": train_loss,
                "val_metrics": val_metrics,
            }

            for cb in callbacks:
                cb.on_epoch_end(self, epoch, logs)

            if self.stop_fit:
                self.stop_fit = False
                break

        for cb in callbacks:
            cb.on_train_end(self)
