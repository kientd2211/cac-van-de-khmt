import SimpleITK as sitk
import numpy as np
import torch
import os

def pad_patch(patch, size=32):
    padded = np.zeros((size, size, size), dtype=np.float32)
    dz, dy, dx = patch.shape
    
    dz = min(dz, size)
    dy = min(dy, size)
    dx = min(dx, size)
    
    padded[:dz, :dy, :dx] = patch[:dz, :dy, :dx]
    return padded

def process_mha_file(mha_path, coord_x, coord_y, coord_z, patch_size=32):
    try:
        ct_image = sitk.ReadImage(mha_path)
        ct_array = sitk.GetArrayFromImage(ct_image)
        
        x_idx, y_idx, z_idx = ct_image.TransformPhysicalPointToIndex((coord_x, coord_y, coord_z))

        ps = patch_size // 2
        
        z1, z2 = z_idx - ps, z_idx + ps
        y1, y2 = y_idx - ps, y_idx + ps
        x1, x2 = x_idx - ps, x_idx + ps

        z1c, z2c = max(z1, 0), min(z2, ct_array.shape[0])
        y1c, y2c = max(y1, 0), min(y2, ct_array.shape[1])
        x1c, x2c = max(x1, 0), min(x2, ct_array.shape[2])

        patch = ct_array[z1c:z2c, y1c:y2c, x1c:x2c]

        if patch.size == 0:
            raise ValueError("Vùng cắt rỗng (Tọa độ nằm ngoài ảnh?)")
        
        if patch.shape != (patch_size, patch_size, patch_size):
            patch = pad_patch(patch, patch_size)
            

        patch = patch.astype(np.float32)
        tensor = torch.from_numpy(patch).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor

    except Exception as e:
        raise RuntimeError(f"Lỗi xử lý ảnh CT: {str(e)}")