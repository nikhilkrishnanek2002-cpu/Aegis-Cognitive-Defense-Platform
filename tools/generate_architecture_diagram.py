#!/usr/bin/env python3
"""Generate the system architecture diagram for the Aegis platform."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PIPELINE_BLOCKS: Sequence[str] = (
    "Signal Generator / RTL-SDR",
    "Feature Extraction",
    "AI Model",
    "Tracking",
    "Cognitive Controller",
    "Dashboard",
)

SECURITY_BLOCK = "Security Layer"

NODE_COLORS: Dict[str, str] = {
    "Signal Generator / RTL-SDR": "#7FB3D5",
    "Feature Extraction": "#76D7C4",
    "AI Model": "#F8C471",
    "Tracking": "#F5B7B1",
    "Cognitive Controller": "#BB8FCE",
    "Dashboard": "#85C1E9",
    SECURITY_BLOCK: "#F7DC6F",
}

BLOCK_WIDTH = 0.16
BLOCK_HEIGHT = 0.17


def _add_block(ax, label: str, center: Tuple[float, float], color: str) -> None:
    lower_left = (center[0] - BLOCK_WIDTH / 2, center[1] - BLOCK_HEIGHT / 2)
    block = FancyBboxPatch(
        lower_left,
        BLOCK_WIDTH,
        BLOCK_HEIGHT,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        linewidth=1.8,
        edgecolor="#34495E",
        facecolor=color,
    )
    ax.add_patch(block)
    ax.text(
        center[0],
        center[1],
        label,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#1B2631",
        wrap=True,
    )


def _connect(ax, start: Tuple[float, float], end: Tuple[float, float], **arrow_kwargs) -> None:
    default_kwargs = dict(arrowstyle="->", linewidth=1.8, color="#2C3E50", shrinkA=15, shrinkB=15)
    default_kwargs.update(arrow_kwargs)
    ax.annotate("", xy=end, xytext=start, arrowprops=default_kwargs)


def create_system_architecture_diagram(output_path: str = "docs/figures/system_architecture.png") -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = [0.08 + idx * 0.15 for idx in range(len(PIPELINE_BLOCKS))]
    pipeline_positions = {block: (x, 0.65) for block, x in zip(PIPELINE_BLOCKS, x_positions)}

    for block, center in pipeline_positions.items():
        _add_block(ax, block, center, NODE_COLORS.get(block, "#D5DBDB"))

    security_center = (0.5, 0.25)
    _add_block(ax, SECURITY_BLOCK, security_center, NODE_COLORS[SECURITY_BLOCK])

    for first, second in zip(PIPELINE_BLOCKS[:-1], PIPELINE_BLOCKS[1:]):
        _connect(ax, pipeline_positions[first], pipeline_positions[second])

    for block in PIPELINE_BLOCKS:
        target_center = pipeline_positions[block]
        _connect(
            ax,
            security_center,
            target_center,
            arrowstyle="-|>",
            linestyle="dashed",
            color="#AF601A",
            linewidth=1.4,
            shrinkA=15,
            shrinkB=18,
        )

    ax.set_title(
        "Aegis Cognitive Defense Platform - System Architecture",
        fontsize=14,
        fontweight="bold",
        pad=20,
        color="#1C2833",
    )

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    return output


def main() -> None:
    parser = ArgumentParser(description="Generate the architecture diagram for publications.")
    parser.add_argument(
        "--output",
        default="docs/figures/system_architecture.png",
        help="Output path for the generated PNG (default: %(default)s)",
    )
    args = parser.parse_args()
    png_path = create_system_architecture_diagram(args.output)
    print(f"✅ System architecture diagram created at: {png_path}")


if __name__ == "__main__":
    main()
