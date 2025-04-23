import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from callbacks import EarlyStoppingCallback, LoggingCallback, SaveCheckpointCallback
from dataset.dataset_lits import LiTSSliceDataset
from loss_func import BCEDiceLoss, DiceLoss
from metrics import f1_score, f2_score, sensitivity, specificity
from model import NestedUNet
from trainer import NestedUNetTrainer


def get_dataloader(
    src_dir,
    patient_ids,
    batch_size,
    cache_slice_info_path=None,
):
    dataset = LiTSSliceDataset(
        src_dir=src_dir,
        patient_ids=patient_ids,
        cache_slice_info_path=cache_slice_info_path,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


if __name__ == "__main__":
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_dir = Path("datasets/MICCAI 2017 LiTS/npz_compressed")
    result_dir = Path("results/checkpoints/Test-Local-03")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "log.jsonl"
    checkpoint_path = None  # result_dir / "checkpoint-0015.pth"
    train_patient_ids = [i for i in range(0, 101, 1)]
    val_patient_ids = [i for i in range(101, 116, 1)]
    train_slice_info_path = result_dir / "train_slice_info.pkl"
    val_slice_info_path = result_dir / "val_slice_info.pkl"

    learning_rate = 3e-4
    batch_size = 2
    num_epochs = 100

    model = NestedUNet()
    model.to(device)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    loss_fn = BCEDiceLoss(alpha=0.5, beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    callbacks = [
        SaveCheckpointCallback(save_path=result_dir),
        LoggingCallback(log_path=log_path, total_epochs=num_epochs),
        EarlyStoppingCallback(patience=10, last_wait=0),
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
        cache_slice_info_path=train_slice_info_path,
    )
    val_loader = get_dataloader(
        src_dir=src_dir,
        patient_ids=val_patient_ids,
        batch_size=batch_size,
        cache_slice_info_path=val_slice_info_path,
    )
    if not train_slice_info_path.exists():
        train_loader.dataset.save_slice_info(result_dir / "train_slice_info.pkl")
    if not val_slice_info_path.exists():
        val_loader.dataset.save_slice_info(result_dir / "val_slice_info.pkl")

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
