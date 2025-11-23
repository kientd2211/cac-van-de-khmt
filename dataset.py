import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from glob import glob

class LungNoduleDataset(Dataset):
    def __init__(self, csv_file, data_folder, patch_size=32):
        """
        csv_file: file CSV annotations LUNA25
        data_folder: folder chứa các thư mục luna25_images_XX/*.mha
        patch_size: size của patch cubic (patch_size^3)
        """
        self.data = pd.read_csv(csv_file)
        self.data_folder = data_folder
        self.patch_size = patch_size

        # Build map SeriesInstanceUID -> .mha file
        self.uid_to_path = {}
        for folder in glob(os.path.join(data_folder, "*")):
            for file in glob(os.path.join(folder, "*.mha")):
                uid = os.path.splitext(os.path.basename(file))[0]
                self.uid_to_path[uid] = file

        # Keep only rows with available CT
        self.valid_rows = []
        for i, row in self.data.iterrows():
            uid = row["SeriesInstanceUID"]
            if uid in self.uid_to_path:
                self.valid_rows.append(i)
            else:
                print(f"[WARNING] ❌ Missing CT for UID: {uid}")

        print(f"[INFO] ✔ Valid samples: {len(self.valid_rows)} / {len(self.data)}")

    def __len__(self):
        return len(self.valid_rows)

    def __getitem__(self, idx):
        real_idx = self.valid_rows[idx]
        row = self.data.iloc[real_idx]
        uid = row["SeriesInstanceUID"]
        file_path = self.uid_to_path[uid]

        # Try read CT image
        try:
            ct_image = sitk.ReadImage(file_path)
            ct = sitk.GetArrayFromImage(ct_image)  # shape (D, H, W)
        except Exception as e:
            print(f"[ERROR] Cannot read CT: {file_path}, skipping. {e}")
            # fallback: return next sample
            return self.__getitem__((idx+1) % len(self))

        # Convert physical coordinates to voxel index
        try:
            x, y, z = ct_image.TransformPhysicalPointToIndex(
                (row["CoordX"], row["CoordY"], row["CoordZ"])
            )
        except Exception as e:
            print(f"[ERROR] Cannot convert Coord to voxel: {uid}, skipping. {e}")
            return self.__getitem__((idx+1) % len(self))

        # Crop patch
        ps = self.patch_size // 2
        z1, z2 = z - ps, z + ps
        y1, y2 = y - ps, y + ps
        x1, x2 = x - ps, x + ps

        # Clip to volume
        z1c, z2c = max(z1, 0), min(z2, ct.shape[0])
        y1c, y2c = max(y1, 0), min(y2, ct.shape[1])
        x1c, x2c = max(x1, 0), min(x2, ct.shape[2])

        patch = ct[z1c:z2c, y1c:y2c, x1c:x2c]

        # Check empty
        if patch.size == 0:
            print(f"[WARNING] Empty patch: {uid}, skipping.")
            return self.__getitem__((idx+1) % len(self))

        # Pad to fixed size
        patch = self._pad_patch(patch, self.patch_size)
        patch = patch.astype(np.float32)
        patch = np.expand_dims(patch, 0)  # add channel dim

        label = np.float32(row["label"])
        return torch.tensor(patch), torch.tensor(label)

    def _pad_patch(self, patch, size):
        padded = np.zeros((size, size, size), dtype=np.float32)
        dz, dy, dx = patch.shape
        # Clip if patch > size
        dz = min(dz, size)
        dy = min(dy, size)
        dx = min(dx, size)
        padded[:dz, :dy, :dx] = patch[:dz, :dy, :dx]

        return padded
