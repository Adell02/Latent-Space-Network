# paper_figures_code/arc_re_arc_samples.py
import os
import sys
import json
import random
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Resolve repo root and import re-arc modules
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
RE_ARC_DIR = os.path.join(REPO_ROOT, "re_arc")
ARC_ORIG_TRAIN_DIR = os.path.join(RE_ARC_DIR, "arc_original", "training")
FIGURES_DIR = os.path.join(REPO_ROOT, "paper_figures")

# Make repo importable
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import re-arc generator pipeline
from re_arc.main import generate_and_process_tasks  # returns (generated_examples, input_grids, output_grids, input_seqs, output_seqs)
import re_arc.generators as generators


def get_two_keys_from_original_training(seed: int = 42) -> List[str]:
    files = [f for f in os.listdir(ARC_ORIG_TRAIN_DIR) if f.endswith(".json")]
    keys = [os.path.splitext(f)[0] for f in files]
    # Prefer keys that have a generator
    keys_with_gen = [k for k in keys if hasattr(generators, f"generate_{k}")]
    if len(keys_with_gen) >= 2:
        random.Random(seed).shuffle(keys_with_gen)
        return keys_with_gen[:2]
    # Fallback: first 2 found
    keys.sort()
    return keys[:2]


def load_original_task(key: str) -> dict:
    path = os.path.join(ARC_ORIG_TRAIN_DIR, f"{key}.json")
    with open(path, "r") as fp:
        return json.load(fp)


def to_numpy_grid(x) -> np.ndarray:
    arr = np.array(x, dtype=int)
    return arr


def get_arc_colormap() -> ListedColormap:
    # 0-9 colors (typical palette)
    colors = [
        "#000000",  # 0 black
        "#0074D9",  # 1 blue
        "#FF4136",  # 2 red
        "#2ECC40",  # 3 green
        "#FFDC00",  # 4 yellow
        "#AAAAAA",  # 5 gray
        "#F012BE",  # 6 magenta
        "#FF851B",  # 7 orange
        "#7FDBFF",  # 8 cyan
        "#8B4513",  # 9 brown
    ]
    return ListedColormap(colors, name="arc10")


def plot_pair(input_grid: np.ndarray, output_grid: np.ndarray, save_path: str, title: str = ""):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cmap = get_arc_colormap()
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
    axs[0].imshow(input_grid, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    axs[0].set_title("Input")
    axs[0].axis("off")
    axs[1].imshow(output_grid, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    axs[1].set_title("Output")
    axs[1].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _pick_example_for_key(key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick two representative ARC examples for a key from original data.

    Prefer (train[0], test[0]) when available; fall back to other examples.
    Returns (in1, out1, in2, out2) as numpy arrays.
    """
    data = load_original_task(key)
    train = data.get("train", [])
    test = data.get("test", [])

    def to_pair(example):
        return to_numpy_grid(example["input"]), to_numpy_grid(example["output"])

    in1 = out1 = in2 = out2 = None
    if len(train) > 0:
        in1, out1 = to_pair(train[0])
    elif len(test) > 0:
        in1, out1 = to_pair(test[0])

    if len(test) > 0:
        in2, out2 = to_pair(test[0])
    elif len(train) > 1:
        in2, out2 = to_pair(train[1])
    elif len(train) == 1:
        in2, out2 = to_pair(train[0])
    elif len(test) > 1:
        in2, out2 = to_pair(test[1])

    return np.array(in1), np.array(out1), np.array(in2), np.array(out2)


def _draw_pair_column(gs_cell, input_grid: np.ndarray, output_grid: np.ndarray, mask_output: bool = False):
    """Within a GridSpec cell, create a 3-row layout: input, arrow, output.
    If mask_output is True, output is a black box with a question mark.
    """
    cmap = get_arc_colormap()
    inner = gs_cell.subgridspec(3, 1, height_ratios=[1.0, 0.25, 1.0])

    ax_in = plt.Subplot(plt.gcf(), inner[0])
    ax_arrow = plt.Subplot(plt.gcf(), inner[1])
    ax_out = plt.Subplot(plt.gcf(), inner[2])

    plt.gcf().add_subplot(ax_in)
    plt.gcf().add_subplot(ax_arrow)
    plt.gcf().add_subplot(ax_out)

    ax_in.imshow(input_grid, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax_in.set_xlabel("Input", fontsize=12)
    ax_in.set_xticks([])
    ax_in.set_yticks([])

    ax_arrow.axis("off")
    ax_arrow.text(0.5, 0.5, "\u2193", ha="center", va="center", fontsize=30)

    if mask_output:
        masked = np.zeros_like(input_grid)
        ax_out.imshow(masked, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
        ax_out.text(0.5, 0.5, "?", ha="center", va="center", color="white", fontsize=28, transform=ax_out.transAxes)
    else:
        ax_out.imshow(output_grid, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax_out.set_xlabel("Output", fontsize=12)
    ax_out.set_xticks([])
    ax_out.set_yticks([])


def build_arc_samples_figure(keys: List[str], save_path: str):
    """Compose a 3-column figure for two ARC keys.

    - Col 1: key1 pair (prefer train[0]) and key2 pair stacked vertically
    - Col 2: key1 alternate pair (prefer test[0]) and key2 alternate pair
    - Col 3: same inputs as Col 1 but outputs masked with a question mark
    """
    assert len(keys) >= 2, "Need two keys for arc_samples figure"
    key1, key2 = keys[0], keys[1]
    k1_in1, k1_out1, k1_in2, k1_out2 = _pick_example_for_key(key1)
    k2_in1, k2_out1, k2_in2, k2_out2 = _pick_example_for_key(key2)

    fig = plt.figure(figsize=(9, 4.5), dpi=200, constrained_layout=True)
    outer = fig.add_gridspec(2, 7, height_ratios=[0.2, 0.8], wspace=0, hspace=1)

    # Place row titles in row 0 and row 3 centered across all 3 columns
    fig.canvas.draw()
    def place_row_title(fig_obj, outer_spec, row_index: int, text: str):
        left_bbox = outer_spec[0, row_index].get_position(fig_obj)
        right_bbox = outer_spec[1, row_index+1].get_position(fig_obj)
        x = (left_bbox.x0 + right_bbox.x1) / 2.0 - 0.02 + (0.002 * row_index)
        y = (left_bbox.y0 + left_bbox.y1) / 2.0
        fig_obj.text(x, y, text, ha="center", va="center", fontsize=12)

    place_row_title(fig, outer, 0, f"Key \'{key1}\'")
    place_row_title(fig, outer, 5, f"Key \'{key2}\'")


    # Column 1
    _draw_pair_column(outer[1, 0], k1_in1, k1_out1, mask_output=False)
    _draw_pair_column(outer[1, 4], k2_in1, k2_out1, mask_output=False)

    # Column 2
    _draw_pair_column(outer[1, 1], k1_in2, k1_out2, mask_output=False)
    _draw_pair_column(outer[1, 5], k2_in2, k2_out2, mask_output=False)

    # Column 3 masked
    _draw_pair_column(outer[1, 2], k1_in1, k1_out1, mask_output=True)
    _draw_pair_column(outer[1, 6], k2_in1, k2_out1, mask_output=True)

    # Column labels
    #fig.text(0.17, 0.96, "ARC", ha="center", va="center", fontsize=10, weight="bold")
    #fig.text(0.50, 0.96, "ARC", ha="center", va="center", fontsize=10, weight="bold")
    #fig.text(0.83, 0.96, "ARC", ha="center", va="center", fontsize=10, weight="bold")
    

    # Draw a vertical dotted line in column 3 spanning the figure height
    fig.canvas.draw()
    from matplotlib.lines import Line2D

    def add_vertical_dotted_line(fig_obj, outer_spec, col_index: int, color: str = "0.6", lw: float = 1.2, alpha: float = 0.8):
        top_bbox = outer_spec[0, col_index].get_position(fig_obj)
        bottom_bbox = outer_spec[1, col_index].get_position(fig_obj)
        x = (top_bbox.x0 + top_bbox.x1) / 2.0
        y0 = min(top_bbox.y0, bottom_bbox.y0)
        y1 = max(top_bbox.y1, bottom_bbox.y1)
        line = Line2D([x, x], [y0, y1], transform=fig_obj.transFigure,
                      linestyle=(0, (4, 4)), color=color, linewidth=lw,
                      alpha=alpha, zorder=1000)
        fig_obj.add_artist(line)

    add_vertical_dotted_line(fig, outer, 3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def build_re_arc_samples_figure(key: str, save_path: str, n_rearc: int = 2):
    """Compose a 3-column figure for a single key:
    - Col 1: original ARC sample (input \u2193 output)
    - Col 2-3: two re-ARC generated samples (input \u2193 output)
    Includes column labels (ARC / re-ARC) and a title with the key.
    """
    in1, out1, _, _ = _pick_example_for_key(key)

    # Generate re-ARC samples
    try:
        _, re_in_grids, re_out_grids, _, _ = generate_and_process_tasks(
            key, n_examples=n_rearc, plot=False, print_data=False
        )
    except Exception as e:
        print(f"[WARN] Failed to generate re-arc examples for key {key}: {e}")
        re_in_grids, re_out_grids = [], []

    # Pad if fewer than requested
    while len(re_in_grids) < n_rearc:
        re_in_grids.append(in1)
        re_out_grids.append(out1)

    # Figure with top spacing row and a right subgrid for tighter last two columns
    fig = plt.figure(figsize=(6, 4.5), dpi=200, constrained_layout=True)
    outer = fig.add_gridspec(2, 3, height_ratios=[0.1, 0.9], width_ratios=[0.8, 0.6,0.9], wspace=0)

    # Left column (ARC sample)
    _draw_pair_column(outer[1, 0], np.array(in1), np.array(out1), mask_output=False)

    # Right two columns packed tightly using a subgridspec
    right_inner = outer[1, 2].subgridspec(1, 2, wspace=0.5)
    _draw_pair_column(right_inner[0, 0], np.array(re_in_grids[0]), np.array(re_out_grids[0]), mask_output=False)
    _draw_pair_column(right_inner[0, 1], np.array(re_in_grids[1]), np.array(re_out_grids[1]), mask_output=False)

    # Column labels
    #fig.text(0.2, 0.96, f"ARC key: {key}", ha="center", va="center", fontsize=15)
    #fig.text(0.8, 0.96, "re-ARC", ha="center", va="center", fontsize=15)

    fig.canvas.draw()
    from matplotlib.patches import FancyArrowPatch
    left_bbox = outer[1, 0].get_position(fig)
    right0_bbox = right_inner[0, 0].get_position(fig)
    x0 = left_bbox.x1 + 0
    x1 = right0_bbox.x0 - 0.05
    y_mid = (left_bbox.y0 + left_bbox.y1) / 2.0
    arrow = FancyArrowPatch((x0, y_mid), (x1, y_mid), transform=fig.transFigure,
                            arrowstyle='->', mutation_scale=20, linewidth=4,
                            color='red', zorder=1500)
    fig.add_artist(arrow)
    fig.text((x0 + x1) / 2.0, y_mid + 0.05, "re-ARC\naugmentation", ha='center', va='bottom', fontsize=12,color="red")
    
    #fig.suptitle(f"ARC vs re-ARC Samples (Key {key})", fontsize=12)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plot_original_examples_for_key(key: str, out_dir: str):
    data = load_original_task(key)
    # original format: {'train': [...], 'test': [...]}, each example has 'input', 'output'
    for split_name in ["train", "test"]:
        if split_name not in data:
            continue
        examples = data[split_name]
        for idx, ex in enumerate(examples):
            x = to_numpy_grid(ex["input"])
            y = to_numpy_grid(ex["output"])
            save_path = os.path.join(out_dir, f"{key}_original_{split_name}_{idx}.png")
            plot_pair(x, y, save_path, title=f"{key} ({split_name} {idx})")


def generate_and_plot_rearc_examples_for_key(key: str, out_dir: str, n: int = 3):
    try:
        generated_examples, input_grids, output_grids, _, _ = generate_and_process_tasks(
            key, n_examples=n, plot=False, print_data=False
        )
    except Exception as e:
        print(f"[WARN] Failed to generate re-arc examples for key {key}: {e}")
        return

    # input_grids and output_grids are lists of np.ndarrays
    for i, (x, y) in enumerate(zip(input_grids, output_grids)):
        save_path = os.path.join(out_dir, f"{key}_rearc_{i}.png")
        plot_pair(np.array(x), np.array(y), save_path, title=f"{key} (re-arc {i})")


def main(keys: List[str] = None, n_rearc: int = 3, seed: int = 42):
    if keys is None or len(keys) == 0:
        keys = get_two_keys_from_original_training(seed)
    elif len(keys) == 1:
        # pick one more automatically
        auto_keys = get_two_keys_from_original_training(seed)
        if auto_keys and auto_keys[0] != keys[0]:
            keys = [keys[0], auto_keys[0]]
        elif len(auto_keys) > 1:
            keys = [keys[0], auto_keys[1]]
        else:
            keys = keys * 2  # duplicate

    print(f"[INFO] Selected keys: {keys}")

    # Ensure output directory
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Build requested composite figures
    arc_samples_path = os.path.join(FIGURES_DIR, "arc_samples.png")
    print(f"[INFO] Building composite figure: {arc_samples_path}")
    build_arc_samples_figure(keys[:2], arc_samples_path)

    re_arc_samples_path = os.path.join(FIGURES_DIR, "re_arc_samples.png")
    print(f"[INFO] Building composite figure: {re_arc_samples_path}")
    build_re_arc_samples_figure(keys[0], re_arc_samples_path, n_rearc=2)

    # Additionally save individual re-ARC example pairs for reference
    re_arc_dir = os.path.join(FIGURES_DIR, "re_arc_examples")
    os.makedirs(re_arc_dir, exist_ok=True)
    for key in keys:
        print(f"[INFO] Saving individual re-ARC examples for key: {key}")
        generate_and_plot_rearc_examples_for_key(key, re_arc_dir, n=n_rearc)

    print(f"[OK] Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ARC original + re-arc examples for paper figures.")
    parser.add_argument("--keys", nargs="*", default=None, help="Two ARC keys (omit to auto-pick).")
    parser.add_argument("--n_rearc", type=int, default=3, help="Number of re-arc examples to generate per key.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for key selection.")
    args = parser.parse_args()

    main(keys=args.keys, n_rearc=args.n_rearc, seed=args.seed)