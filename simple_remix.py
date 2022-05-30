import torch
import numpy as np
from typing import Optional

def simple_remix(images: torch.Tensor, masks: torch.Tensor):
    B, C, H, W = images.shape
    B, n_tokens_H, n_tokens_W = masks.shape

    assert H == W, f"Expected H == W, got H = {H}, W = {W}"
    assert n_tokens_H == n_tokens_W, f"Expected n_tokens_H == n_tokens_W, got n_tokens_H = {n_tokens_H}, n_tokens_W = {n_tokens_W}"

    batch_masked_indices, x_masked_indices, y_masked_indices = np.argwhere(masks)

    assert H % n_tokens_H == 0, f"Expected H % n_tokens_H == 0, got H = {H}, n_tokens_H = {n_tokens_H}"
    patch_size = H // n_tokens_H

    mixed = images.clone()
    for i, (b, x, y) in enumerate(zip(batch_masked_indices, x_masked_indices, y_masked_indices)):
        x_low = x * patch_size
        x_high = (x + 1) * patch_size
        y_low = y * patch_size
        y_high = (y + 1) * patch_size

        mixed[b, x_low:x_high , y_low:y_high] = images.roll(i, 0, 0, 0)[b, x_low:x_high, y_low:y_high]

    return mixed


def simple_remix_fast(images: torch.Tensor, masks: torch.Tensor):
    b, c, h, w = images.shape
    _, h_patches, w_patches = masks.shape
    
    ph = h // h_patches
    pw = w // w_patches

    images_as_patches = images.reshape(b, c, h//ph, ph, w//pw, pw).permute(0, 2, 4, 1, 3, 5).flatten(1, 2) # [b, n_patches, c, ph, pw]
    num_patches = images_as_patches.shape[1]

    destination_batch_indices, destination_patch_indices = np.argwhere(masks.flatten(1, 2))

    sources_batch_indices = torch.randint_like(destination_batch_indices, 0, b)
    sources_patch_indices = torch.randint_like(destination_patch_indices, 0, num_patches)

    images_as_patches[destination_batch_indices, destination_patch_indices] = images_as_patches[sources_batch_indices, sources_patch_indices]
    images_replaced = images_as_patches.unflatten(1, (h//ph, w//pw)).permute(0, 3, 1, 4, 2, 5).reshape_as(images)

    return images_replaced