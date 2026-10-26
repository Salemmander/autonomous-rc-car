"""Procedural floor textures, so the policy sees more than one kind of floor.

All are grayscale and tile seamlessly; the env tints them a random color and scale
per episode (env/kernels.py randomize_look). The original checker floor
(assets/car.py build_scene) stays the "nominal" look that eval.py and view.py use.

Preview them: uv run python show_views.py
"""

import numpy as np

SIZE = 256  # px per texture side


def _smooth_noise(rng: np.random.Generator, scale: float) -> np.ndarray:
    """Tileable noise in [0, 1] with features about `scale` px across (low-pass
    filtered white noise; the FFT makes it wrap around seamlessly)."""
    f = np.fft.fftfreq(SIZE)
    radius = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    spectrum = np.fft.fft2(rng.standard_normal((SIZE, SIZE))) * np.exp(
        -((radius * scale) ** 2)
    )
    x = np.real(np.fft.ifft2(spectrum))
    return (x - x.min()) / (x.max() - x.min())


def checker(rng: np.random.Generator) -> np.ndarray:
    """Two-tone checker with smaller squares than the original (8x8 per tile)."""
    y, x = np.mgrid[0:SIZE, 0:SIZE] // (SIZE // 8)
    return np.where((x + y) % 2 == 0, 0.55, 0.75)


def concrete(rng: np.random.Generator) -> np.ndarray:
    return (
        0.55
        + 0.25 * (_smooth_noise(rng, 40) - 0.5)
        + 0.15 * (_smooth_noise(rng, 4) - 0.5)
    )


def carpet(rng: np.random.Generator) -> np.ndarray:
    return (
        0.5
        + 0.35 * (_smooth_noise(rng, 1.5) - 0.5)
        + 0.1 * (_smooth_noise(rng, 30) - 0.5)
    )


def planks(rng: np.random.Generator) -> np.ndarray:
    """Floorboards: each a slightly different shade, with grain and dark seams.
    The seams are lines on the floor that aren't tape, which the policy must ignore."""
    board = SIZE // 8
    rows = np.arange(SIZE) // board
    shade = rng.uniform(0.45, 0.75, 8)[rows][:, None]
    # Grain: noise stretched along the boards
    grain = np.repeat(_smooth_noise(rng, 2)[:, :: SIZE // 16], SIZE // 16, axis=1)
    grain = np.roll(grain, rng.integers(SIZE), axis=1)
    tex = shade + 0.12 * (grain - 0.5) + 0.06 * (_smooth_noise(rng, 20) - 0.5)
    tex[np.arange(SIZE) % board < 2] *= 0.6  # seams
    return tex


def tiles(rng: np.random.Generator) -> np.ndarray:
    """Square tiles with grout lines (4x4 per texture tile)."""
    tile = SIZE // 4
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    shade = rng.uniform(0.7, 0.8, (4, 4))[y // tile, x // tile]
    tex = shade + 0.05 * (_smooth_noise(rng, 10) - 0.5)
    grout = (y % tile < 3) | (x % tile < 3)
    tex[grout] = 0.45
    return tex


def speckle(rng: np.random.Generator) -> np.ndarray:
    """Terrazzo / speckled linoleum."""
    tex = 0.6 + 0.08 * (_smooth_noise(rng, 15) - 0.5)
    for _ in range(600):
        y, x = rng.integers(0, SIZE, 2)
        r = rng.integers(1, 4)
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk = yy**2 + xx**2 <= r**2
        rows = (y + yy) % SIZE
        cols = (x + xx) % SIZE
        tex[rows, cols] = np.where(disk, rng.uniform(0.2, 0.95), tex[rows, cols])
    return tex


def flat(rng: np.random.Generator) -> np.ndarray:
    return 0.65 + 0.04 * (_smooth_noise(rng, 6) - 0.5)


FLOORS = [checker, concrete, carpet, planks, tiles, speckle, flat]


def floor_textures(seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """[(name, (SIZE, SIZE, 3) uint8 RGB)], the same every time."""
    out = []
    for k, make in enumerate(FLOORS):
        gray = np.clip(make(np.random.default_rng([seed, k])), 0, 1)
        rgb = np.repeat((gray * 255).round().astype(np.uint8)[:, :, None], 3, axis=2)
        out.append((make.__name__, rgb))
    return out
