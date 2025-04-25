import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LiTSDataPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def _get_patient_dir(patient_id):
        if patient_id <= 27:
            return "Training Batch 1"
        elif patient_id <= 130:
            return "Training Batch 2"

        print(f"[Error] This patient ID {patient_id} doesn't exist.")
        return ""

    @staticmethod
    def _extract_liver_mask(mask_array):
        return (mask_array == 1).astype(np.uint8)

    @staticmethod
    def process_one_list_patient(src_dir, dest_dir, patient_id, slice_axis):
        try:
            src_dir = Path(src_dir)
            dest_dir = Path(dest_dir)

            ct_path = (
                src_dir
                / LiTSDataPreprocessor._get_patient_dir(patient_id)
                / f"volume-{patient_id}.nii"
            )
            seg_path = (
                src_dir
                / LiTSDataPreprocessor._get_patient_dir(patient_id)
                / f"segmentation-{patient_id}.nii"
            )
            if not ct_path.exists() or not seg_path.exists():
                print(f"[Error] Missing files for the patient {patient_id}")
                return

            img_data = nib.load(ct_path).get_fdata()
            seg_data = nib.load(seg_path).get_fdata()
            if img_data.shape != seg_data.shape:
                print(f"[Error] Shape mismatch in the patient {patient_id}")
                return

            for slice_idx in range(img_data.shape[slice_axis]):
                ct_slice = np.take(img_data, indices=slice_idx, axis=slice_axis)
                seg_slice = LiTSDataPreprocessor._extract_liver_mask(
                    np.take(seg_data, indices=slice_idx, axis=slice_axis),
                )

                slice_name = f"{ct_path.stem}_slice-{slice_idx}.npz_compressed"
                np.savez_compressed(
                    dest_dir / slice_name,
                    image=ct_slice.astype(np.float32),
                    mask=seg_slice.astype(np.uint8),
                )
        except Exception as e:
            print(f"[Error] Error occurred while processing patient {patient_id}: {e}")

    @classmethod
    def convert_lits_to_2d_slices(
        cls,
        src_dir,
        dest_dir,
        patient_ids=None,
        slice_axis=2,
    ):
        if patient_ids is None:
            print("[Error] No patient IDs contained.")
            return

        src_dir = Path(src_dir)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for patient_id in tqdm(patient_ids, desc="Processing", ncols=80):
            LiTSDataPreprocessor.process_one_list_patient(
                src_dir,
                dest_dir,
                patient_id,
                slice_axis,
            )


class ParallelLiTSDataPreprocessor:
    def __init__(self):
        pass

    @classmethod
    def convert_lits_to_2d_slices(
        cls,
        src_dir,
        dest_dir,
        patient_ids=None,
        slice_axis=2,
        num_workers=4,
    ):
        if patient_ids is None:
            print("[Error] No patient IDs contained.")
            return

        src_dir = Path(src_dir)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    LiTSDataPreprocessor.process_one_list_patient,
                    src_dir,
                    dest_dir,
                    patient_id,
                    slice_axis,
                )
                for patient_id in patient_ids
            ]

            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parallel processing",
                ncols=80,
            ):
                pass


class LiTSSliceDataset(Dataset):
    def __init__(
        self,
        src_dir,
        patient_ids,
        transform=None,
        cache_slice_info_path=None,
        slice_axis=2,
    ):
        self.src_dir = Path(src_dir)
        self.patient_ids = patient_ids
        self.transform = transform
        self.slice_axis = slice_axis
        self.slice_info = []

        if cache_slice_info_path is not None and Path(cache_slice_info_path).exists():
            self._load_slice_info(Path(cache_slice_info_path))
        else:
            self._generate_slice_info()

    def save_slice_info(self, dest_path):
        with open(dest_path, "wb") as f:
            pickle.dump(
                {
                    "patient_ids": self.patient_ids,
                    "slice_info": self.slice_info,
                },
                f,
            )

    def _load_slice_info(self, src_path):
        cache_info = None
        with open(src_path, "rb") as f:
            cache_info = pickle.load(f)

        if cache_info["patient_ids"] != self.patient_ids:
            print(
                "[INFO] Patient IDs mismatch."
                + "Cache will be rebuilt with current patient_ids."
            )
            self._generate_slice_info()
            return

        self.slice_info = cache_info["slice_info"]

    def _generate_slice_info(self):
        for patient_id in self.patient_ids:
            slice_path_pattern = f"volume-{patient_id}_slice-*.npz_compressed.npz"
            slice_paths = sorted(self.src_dir.glob(slice_path_pattern))
            self.slice_info += slice_paths

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        sample = np.load(self.slice_info[idx])

        if self.transform is not None:
            sample = self.transform(sample)

        return {
            "image": torch.from_numpy(sample["image"]).unsqueeze(0),
            "mask": torch.from_numpy(sample["mask"]).unsqueeze(0),
        }
