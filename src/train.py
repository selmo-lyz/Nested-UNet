import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from callbacks import EarlyStoppingCallback, LoggingCallback, SaveCheckpointCallback
from dataset.list import get_dataloader, transform
from loss_func import BCEDiceLoss, DiceLoss
from metrics import f1_score, f2_score, get_metric_name, sensitivity, specificity
from model import NestedUNet
from trainer import NestedUNetTrainer


def data_filter(paths):
    retain_empty_ratio = 0.1

    non_empty_mask_paths = []
    empty_mask_paths = []
    for path in paths:
        sample = np.load(path)
        if np.any(sample["mask"]):
            non_empty_mask_paths.append(path)
        else:
            empty_mask_paths.append(path)

    keep_empty_mask_path = random.sample(
        empty_mask_paths,
        int(len(empty_mask_paths) * retain_empty_ratio),
    )
    filtered_paths = non_empty_mask_paths + keep_empty_mask_path
    print(
        "[INFO] Filtered paths: "
        f"{len(non_empty_mask_paths)} non-empty + {len(keep_empty_mask_path)} empty "
        f"= {len(filtered_paths)} of {len(paths)} total"
    )
    return filtered_paths


if __name__ == "__main__":
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_dir = Path("datasets/MICCAI 2017 LiTS/npz_compressed")
    result_dir = Path("results/checkpoints/Test-Local-06")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "log.jsonl"
    checkpoint_path = None  # result_dir / "checkpoint-0003.pth"
    train_patient_ids = [i for i in range(0, 101)]
    val_patient_ids = [i for i in range(101, 116)]
    train_slice_info_path = result_dir / "train_slice_info.pkl"
    val_slice_info_path = result_dir / "val_slice_info.pkl"

    learning_rate = 3e-4
    batch_size = 2
    num_epochs = 30

    model = NestedUNet().to(device)
    loss_fn = BCEDiceLoss(alpha=0.2, beta=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    callbacks = [
        SaveCheckpointCallback(save_path=result_dir),
        LoggingCallback(log_path=log_path, total_epochs=num_epochs),
        EarlyStoppingCallback(
            metric_fname=get_metric_name(f1_score),
            patience=5,
            last_wait=0,
        ),
    ]
    metric_fns = [
        sensitivity,
        specificity,
        f1_score,
        f2_score,
    ]

    train_loader = get_dataloader(
        src_dir=src_dir,
        patient_ids=train_patient_ids,
        batch_size=batch_size,
        transform=transform,
        data_filter=data_filter,
        cache_slice_info_path=train_slice_info_path,
    )
    val_loader = get_dataloader(
        src_dir=src_dir,
        patient_ids=val_patient_ids,
        batch_size=batch_size,
        transform=transform,
        cache_slice_info_path=val_slice_info_path,
    )
    if not train_slice_info_path.exists():
        train_loader.dataset.save_slice_info(train_slice_info_path)
    if not val_slice_info_path.exists():
        val_loader.dataset.save_slice_info(val_slice_info_path)

    trainer = NestedUNetTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    if checkpoint_path is not None:
        trainer.load_checkpoint(checkpoint_path)
    trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=num_epochs,
        callbacks=callbacks,
        metric_fns=metric_fns,
    )
