import torch
import numpy as np
from torch.nn.functional import pad

def simple_remix_fast(images: torch.Tensor, masks: torch.Tensor, scale: int):
    """
    Replaces masked patches with random patches from a batch of images.

    images: torch.Tensor of shape (b, c, h, w)
        a batch of images
    masks: torch.Tensor of shape (b, height_n_model_patches, width_n_model_patches)
        '1' means masked, '0' means unmasked
    scale: int,
        mask_patch_size / model_patch_size ratio
    """
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


def shifted_remix(images: torch.Tensor, masks: torch.Tensor, scale: int):
    """
    Randomly shifts masked positions up to 1 mask patch height/width,
    then replaces masked patches with random patches from a batch of images.

    images: torch.Tensor of shape (b, c, h, w)
        a batch of images
    masks: torch.Tensor of shape (b, height_n_model_patches, width_n_model_patches)
        '1' means masked, '0' means unmasked
    scale: int,
        mask_patch_size / model_patch_size ratio
    """
    b, c, h, w = images.shape
    _, h_patches, w_patches = masks[:, ::scale, ::scale].shape

    ph = h // h_patches
    pw = w // w_patches

    padded_images = pad(images, (ph, ph, pw, pw))
    h_shift = torch.randint(0, 2 * ph, (1,))
    w_shift = torch.randint(0, 2 * pw, (1,))

    remixed = simple_remix_fast(padded_images[..., h_shift:h_shift - 2 * ph, w_shift:w_shift - 2 * pw], masks, scale)

    padded_images[..., h_shift:h_shift - 2 * ph, w_shift:w_shift - 2 * pw] = remixed

    return padded_images[..., ph:-ph, pw:-pw]
