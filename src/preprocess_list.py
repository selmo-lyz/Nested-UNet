from pathlib import Path

from dataset.dataset_lits import ParallelLiTSDataPreprocessor

if __name__ == "__main__":
    src_dir = Path(
        "./datasets/MICCAI 2017 LiTS/compressed/media/nas/01_Datasets/CT/LITS"
    )
    dest_dir = Path("./datasets/MICCAI 2017 LiTS/tmp")

    ParallelLiTSDataPreprocessor.convert_lits_to_2d_slices(
        src_dir=src_dir,
        dest_dir=dest_dir,
        patient_ids=[i for i in range(0, 131)],
        num_workers=4,
    )
