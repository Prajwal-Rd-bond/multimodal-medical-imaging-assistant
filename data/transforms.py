"""Preprocessing / augmentation pipelines shared by both modalities."""
from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _to_rgb(image):
    """Named (not lambda) so it survives pickling when DataLoader workers
    are spawned -- macOS/Windows default to 'spawn', which pickles the
    transform pipeline to hand to each worker process."""
    return image.convert("RGB")


def train_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(_to_rgb),
            # RandomResizedCrop instead of a plain Resize -- a mild crop (0.85-1.0
            # of the image) gives some translation/scale augmentation on top of
            # flip/rotation/jitter, useful on a small (5-6k image) training set.
            T.RandomResizedCrop((image_size, image_size), scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(_to_rgb),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def denormalize(tensor):
    """Undo ImageNet normalization for visualization (Grad-CAM overlays etc.)."""
    import torch

    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
