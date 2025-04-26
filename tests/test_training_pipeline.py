from pathlib import Path

import torch

from callbacks import EarlyStoppingCallback, LoggingCallback, SaveCheckpointCallback
from dataset.list import get_dataloader, transform
from loss_func import BCEDiceLoss
from metrics import f1_score, f2_score, get_metric_name, sensitivity, specificity
from model import NestedUNet
from train import data_filter, generate_layer_configs
from trainer import NestedUNetTrainer


def test_training_pipeline_runs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_dir = Path("datasets/MICCAI 2017 LiTS/npz_compressed")
    result_dir = Path("results/checkpoints/Test-Local-test")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "log.jsonl"

    learning_rate = 3e-4
    batch_size = 1

    model = NestedUNet(layer_configs=generate_layer_configs()).to(device)
    loss_fn = BCEDiceLoss(alpha=0.5, beta=1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    callbacks = [
        SaveCheckpointCallback(save_path=result_dir),
        LoggingCallback(log_path=log_path, total_epochs=1),
        EarlyStoppingCallback(metric_fname=get_metric_name(f1_score), mode="max"),
    ]
    metric_fns = [
        sensitivity,
        specificity,
        f1_score,
        f2_score,
        loss_fn,
    ]

    dataloader = get_dataloader(
        src_dir=src_dir,
        patient_ids=[0],
        batch_size=batch_size,
        transform=transform,
        data_filter=data_filter,
    )

    trainer = NestedUNetTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    trainer.fit(
        train_dataloader=dataloader,
        val_dataloader=dataloader,
        num_epochs=1,
        callbacks=callbacks,
        metric_fns=metric_fns,
    )
