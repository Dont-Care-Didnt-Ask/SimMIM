import torch
import numpy as np

def simple_remix_fast(images: torch.Tensor, masks: torch.Tensor, scale: int):
    b, c, h, w = images.shape
    _, h_patches, w_patches = masks[:, ::scale, ::scale].shape
    
    ph = h // h_patches
    pw = w // w_patches

    images_as_patches = images.reshape(b, c, h//ph, ph, w//pw, pw).permute(0, 2, 4, 1, 3, 5).flatten(1, 2) # [b, n_patches, c, ph, pw]
    num_patches = images_as_patches.shape[1]

    destination_batch_indices, destination_patch_indices = np.argwhere(masks[:, ::scale, ::scale].flatten(1, 2).cpu())

    sources_batch_indices = torch.randint_like(destination_batch_indices, 0, b)
    sources_patch_indices = torch.randint_like(destination_patch_indices, 0, num_patches)

    images_as_patches[destination_batch_indices, destination_patch_indices] = images_as_patches[sources_batch_indices, sources_patch_indices]
    images_replaced = images_as_patches.unflatten(1, (h//ph, w//pw)).permute(0, 3, 1, 4, 2, 5).reshape_as(images)

    return images_replaced