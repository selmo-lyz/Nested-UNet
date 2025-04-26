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
    retain_empty_ratio = 0.25

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


def generate_layer_configs():
    encoder_channels = [32, 64, 128, 256, 512]
    layer_configs = {
        "EncoderNode": {
            "node0_0": {
                "in_channels": 1,
                "out_channels": encoder_channels[0],
                "sampling_method": "conv",
            },
            "node1_0": {
                "in_channels": encoder_channels[0],
                "out_channels": encoder_channels[1],
                "sampling_method": "conv",
            },
            "node2_0": {
                "in_channels": encoder_channels[1],
                "out_channels": encoder_channels[2],
                "sampling_method": "conv",
            },
            "node3_0": {
                "in_channels": encoder_channels[2],
                "out_channels": encoder_channels[3],
                "sampling_method": "conv",
            },
            "node4_0": {
                "in_channels": encoder_channels[3],
                "out_channels": encoder_channels[4],
                "sampling_method": "none",
            },
        },
        "DecoderNode": {
            # NestedUNet: Level 1 Decoder
            "node0_1": {
                "in_channels": encoder_channels[0] + encoder_channels[1],
                "upsampling_in_channels": encoder_channels[1],
                "out_channels": encoder_channels[0],
                "sampling_method": "conv",
            },
            # NestedUNet: Level 2 Decoder
            "node1_1": {
                "in_channels": encoder_channels[1] + encoder_channels[2],
                "upsampling_in_channels": encoder_channels[2],
                "out_channels": encoder_channels[1],
                "sampling_method": "conv",
            },
            "node0_2": {
                "in_channels": encoder_channels[0] * 2 + encoder_channels[1],
                "upsampling_in_channels": encoder_channels[1],
                "out_channels": encoder_channels[0],
                "sampling_method": "conv",
            },
            # NestedUNet: Level 3 Decoder
            "node2_1": {
                "in_channels": encoder_channels[2] + encoder_channels[3],
                "upsampling_in_channels": encoder_channels[3],
                "out_channels": encoder_channels[2],
                "sampling_method": "conv",
            },
            "node1_2": {
                "in_channels": encoder_channels[1] * 2 + encoder_channels[2],
                "upsampling_in_channels": encoder_channels[2],
                "out_channels": encoder_channels[1],
                "sampling_method": "conv",
            },
            "node0_3": {
                "in_channels": encoder_channels[0] * 3 + encoder_channels[1],
                "upsampling_in_channels": encoder_channels[1],
                "out_channels": encoder_channels[0],
                "sampling_method": "conv",
            },
            # NestedUNet: Level 4 Decoder
            "node3_1": {
                "in_channels": encoder_channels[3] + encoder_channels[4],
                "upsampling_in_channels": encoder_channels[4],
                "out_channels": encoder_channels[3],
                "sampling_method": "conv",
            },
            "node2_2": {
                "in_channels": encoder_channels[2] * 2 + encoder_channels[3],
                "upsampling_in_channels": encoder_channels[3],
                "out_channels": encoder_channels[2],
                "sampling_method": "conv",
            },
            "node1_3": {
                "in_channels": encoder_channels[1] * 3 + encoder_channels[2],
                "upsampling_in_channels": encoder_channels[2],
                "out_channels": encoder_channels[1],
                "sampling_method": "conv",
            },
            "node0_4": {
                "in_channels": encoder_channels[0] * 4 + encoder_channels[1],
                "upsampling_in_channels": encoder_channels[1],
                "out_channels": encoder_channels[0],
                "sampling_method": "conv",
            },
        },
        "DeepSupervisionModule": {
            "deep_supervision": {
                "in_channels": encoder_channels[0],
                "out_channels": 1,
                "num_level": 4,
            },
        },
    }

    return layer_configs


if __name__ == "__main__":
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_dir = Path("datasets/MICCAI 2017 LiTS/npz_compressed")
    result_dir = Path("results/checkpoints/Test-Local-07")
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

    model = NestedUNet(layer_configs=generate_layer_configs()).to(device)
    loss_fn = BCEDiceLoss(alpha=0.5, beta=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    callbacks = [
        SaveCheckpointCallback(save_path=result_dir),
        LoggingCallback(log_path=log_path, total_epochs=num_epochs),
        EarlyStoppingCallback(
            metric_fname=get_metric_name(f1_score),
            mode="max",
            patience=5,
            last_wait=0,
        ),
    ]
    metric_fns = [
        sensitivity,
        specificity,
        f1_score,
        f2_score,
        loss_fn,
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
