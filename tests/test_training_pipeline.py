import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from callbacks import EarlyStoppingCallback, LoggingCallback, SaveCheckpointCallback
from dataset.dataset_lits import LiTSSliceDataset
from loss_func import BCEDiceLoss
from metrics import f1_score, f2_score, sensitivity, specificity
from model import NestedUNet
from train import get_dataloader, transform
from trainer import NestedUNetTrainer


def test_training_pipeline_runs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_dir = Path("datasets/MICCAI 2017 LiTS/npz_compressed")
    result_dir = Path("results/checkpoints/Test-Local-test")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "log.jsonl"

    learning_rate = 3e-4
    batch_size = 2

    model = NestedUNet().to(device)
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.99),
    )
    callbacks = [
        SaveCheckpointCallback(save_path=result_dir),
        LoggingCallback(log_path=log_path, total_epochs=1),
        EarlyStoppingCallback(metric_fname=loss_fn.__class__.__name__),
    ]
    metric_fns = [
        sensitivity,
        specificity,
        f1_score,
        f2_score,
    ]

    dataloader = get_dataloader(
        src_dir=src_dir,
        patient_ids=[0],
        batch_size=batch_size,
        transform=transform,
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
