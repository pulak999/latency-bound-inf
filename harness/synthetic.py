"""Seeded synthetic images.

Every image has the same size, so every request has the same input shape and
the model does the same amount of work. The same seed always gives the same
pixels, so a run's inputs can be regenerated from its config.
"""

import io

import numpy as np
from PIL import Image

# Gemma 3's vision encoder works at 896x896. Other sizes get resized to this,
# so generating at this size keeps resizing out of the measured path.
GEMMA3_IMAGE_SIZE = 896


def synthetic_png(seed: int, size: int = GEMMA3_IMAGE_SIZE) -> bytes:
    """Return a PNG of uniform RGB noise, size x size pixels."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def image_pool(n_images: int, first_seed: int = 0) -> list[bytes]:
    """Generate n distinct images up front, so no encoding happens while timing.

    Request i uses pool[i % n_images]. Distinct images stop the engine from
    reusing work across consecutive requests.
    """
    return [synthetic_png(seed) for seed in range(first_seed, first_seed + n_images)]
