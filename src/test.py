import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.list import get_dataloader, transform
from inference_executor import NestedUNetInferenceRunner
from metrics import f1_score, f2_score, get_metric_name, sensitivity, specificity
from model import NestedUNet

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_dir = Path("datasets/MICCAI 2017 LiTS/npz_compressed")
    result_dir = Path("results/checkpoints/Test-Local-01")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "testing-log.jsonl"
    checkpoint_path = result_dir / "checkpoint-0027.pth"
    test_patient_ids = [i for i in range(116, 131, 1)]
    test_slice_info_path = result_dir / "test_slice_info.pkl"

    batch_size = 8

    model = NestedUNet()
    model.to(device)
    metric_fns = [
        sensitivity,
        specificity,
        f1_score,
        f2_score,
    ]

    test_loader = get_dataloader(
        src_dir=src_dir,
        patient_ids=test_patient_ids,
        batch_size=batch_size,
        transform=transform,
        cache_slice_info_path=test_slice_info_path,
    )
    if not test_slice_info_path.exists():
        test_loader.dataset.save_slice_info(test_slice_info_path)

    runner = NestedUNetInferenceRunner(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    metrics = runner.run(
        dataloader=test_loader,
        metric_fns=metric_fns,
    )

    for idx, (metric_fn, metric) in enumerate(zip(metric_fns, metrics)):
        print(f"{idx}. {get_metric_name(metric_fn)}: {metric:.2f}")
