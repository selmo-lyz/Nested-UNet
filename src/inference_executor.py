from pathlib import Path

import torch
from tqdm import tqdm


class NestedUNetInferenceRunner:
    def __init__(
        self,
        model,
        checkpoint_path,
        device,
    ):
        self.model = model.to(device)
        self.device = device

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state"])

    def default_postprocess(self, outputs):
        return (outputs > 0.5).float()

    def run(
        self,
        dataloader,
        metric_fns,
    ):
        if len(metric_fns) <= 0:
            print("[ERROR] There are no metric functions in the parameter metric_fns")
            return

        self.model.eval()
        running_metrics = [0 for i in range(len(metric_fns))]
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference", ncols=80):
                inputs = batch["image"].to(self.device)
                labels = batch["mask"].to(self.device)
                batch_size = inputs.size(0)

                outputs = self.model(inputs)
                outputs = torch.stack(outputs, dim=0)
                preds = torch.mean(outputs, dim=0)

                # preds = outputs[3]

                for idx, metric_fn in enumerate(metric_fns):
                    metric = metric_fn(preds, labels)
                    running_metrics[idx] += metric * batch_size
                total_samples += batch_size

        avg_metrics = [
            running_metric / total_samples for running_metric in running_metrics
        ]
        return avg_metrics
