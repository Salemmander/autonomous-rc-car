"""Preview random tracks from the generator, top-down, as a grid of pictures.

Run from sim/:
    uv run python show_tracks.py                        # tracks 0-15, [tracks] from baseline.toml
    uv run python show_tracks.py --first 16 --count 32  # the next ones
    uv run python show_tracks.py --test                 # the held-out test tracks
    uv run python show_tracks.py experiments/twisty.toml --set tracks.gap=0.5

Each tile is the 10 x 10 m floor with 1 m grid lines: white = tape, green dot +
arrow = start and driving direction. Under it: track number, length, width, and
tightest curve radius. Saves tracks.png and opens it.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from assets.track import Track
from assets.track_gen import acceptance, generate_track, turn_radii
from experiment import load_experiment

TILE = 320  # px
FLOOR = 10.0  # m shown per tile
LABEL = 18  # px for the text under each tile


def draw_track(track: Track, index: int) -> Image.Image:
    img = Image.new("RGB", (TILE, TILE + LABEL), (25, 25, 25))
    draw = ImageDraw.Draw(img)
    scale = TILE / FLOOR

    def px(xy: np.ndarray) -> list[tuple[float, float]]:
        """meters (origin in the middle, y up) -> pixels (origin top left, y down)"""
        return [(TILE / 2 + x * scale, TILE / 2 - y * scale) for x, y in xy]

    for m in range(-5, 6):  # 1 m grid
        a, b = px(np.array([[m, -5.0], [m, 5.0]])), px(np.array([[-5.0, m], [5.0, m]]))
        draw.line(a, fill=(50, 50, 50))
        draw.line(b, fill=(50, 50, 50))

    edge = track.normals * track.width / 2
    for side in (track.centerline + edge, track.centerline - edge):
        draw.line(px(np.vstack([side, side[:1]])), fill=(240, 240, 240), width=2)

    (x0, y0), (x1, y1) = px(track.centerline[[0, 10]])  # start, 0.5 m ahead
    draw.line([(x0, y0), (x1, y1)], fill=(80, 220, 80), width=3)
    draw.ellipse([x0 - 4, y0 - 4, x0 + 4, y0 + 4], fill=(80, 220, 80))

    label = (
        f"#{index}  {track.length:.1f} m  w {track.width:.2f}  "
        f"r_min {turn_radii(track).min():.2f}"
    )
    draw.text((6, TILE + 3), label, fill=(220, 220, 220))
    return img


def main():
    parser = argparse.ArgumentParser(description="Preview random tracks")
    parser.add_argument("experiment", nargs="?", type=Path)
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--first", type=int, default=0, help="first track number")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, help="default: the experiment's seed")
    parser.add_argument("--out", type=Path, default=Path("tracks.png"))
    parser.add_argument("--test", action="store_true", help="held-out test tracks")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    exp = load_experiment(args.experiment, args.set)
    seed = exp.seed if args.seed is None else args.seed
    cfg = exp.tracks

    stats = acceptance(cfg, tries=300)
    print(f"proposals that pass: {stats.pop('ok', 0) / 3:.0f}%  rejected for: {stats}")

    indices = range(args.first, args.first + args.count)
    tiles = [draw_track(generate_track(seed, i, cfg, args.test), i) for i in indices]
    cols = min(4, len(tiles))
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * TILE, rows * (TILE + LABEL)), (0, 0, 0))
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % cols) * TILE, (k // cols) * (TILE + LABEL)))
    sheet.save(args.out)
    print(f"saved {args.out}")
    if not args.no_open:
        sheet.show()


if __name__ == "__main__":
    main()
