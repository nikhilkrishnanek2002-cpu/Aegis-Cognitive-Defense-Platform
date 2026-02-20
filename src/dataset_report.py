"""High-level dataset reporting utilities for photonic radar experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .dataset_reporting import DatasetReporter


@dataclass
class DatasetReportArtifacts:
    """Paths to generated dataset reporting artifacts."""

    text_summary: Path
    csv_summary: Path
    plot_path: Path
    json_stats: Path
    stats: Dict[str, Any]


def generate_dataset_report(
    signals: Iterable[np.ndarray],
    labels: Iterable[int],
    *,
    snr_values: Optional[Iterable[float]] = None,
    class_names: Optional[Iterable[str]] = None,
    output_root: Path | str = "outputs",
) -> DatasetReportArtifacts:
    """Generate dataset summaries, CSVs, and plots in a single call.

    Parameters
    ----------
    signals : Iterable[np.ndarray]
        Collection of radar signals or features.
    labels : Iterable[int]
        Integer class labels aligned with ``signals``.
    snr_values : Iterable[float], optional
        Optional SNR annotations (dB) per sample.
    class_names : Iterable[str], optional
        Human-readable class names. Length must cover all label indices.
    output_root : Path | str, default="outputs"
        Root directory where ``reports/`` and ``plots/`` folders will be created.

    Returns
    -------
    DatasetReportArtifacts
        Structured container with artifact paths and computed statistics.
    """

    reports_dir = Path(output_root) / "reports"
    plots_dir = Path(output_root) / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    reporter = DatasetReporter(
        signals=list(signals),
        labels=np.asarray(labels),
        class_names=list(class_names) if class_names is not None else None,
        snr_values=np.asarray(list(snr_values)) if snr_values is not None else None,
    )
    stats = reporter.compute_statistics()

    text_summary = reports_dir / "dataset_summary.txt"
    csv_summary = reports_dir / "dataset_summary.csv"
    plot_path = plots_dir / "dataset_distribution.png"
    json_stats = reports_dir / "dataset_stats.json"

    reporter.save_text_report(str(text_summary))
    reporter.save_csv_summary(str(csv_summary))
    reporter.plot_distributions(output_file=str(plot_path))

    def _to_serializable(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        return obj

    with open(json_stats, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(stats), handle, indent=2)

    return DatasetReportArtifacts(
        text_summary=text_summary,
        csv_summary=csv_summary,
        plot_path=plot_path,
        json_stats=json_stats,
        stats=stats,
    )
