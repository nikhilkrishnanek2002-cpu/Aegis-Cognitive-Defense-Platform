"""
Dataset Reporting Utility
==========================

Comprehensive analysis and reporting for radar signal datasets:
  - Class distribution analysis
  - Signal length statistics (min, max, mean, std)
  - SNR distribution and statistics
  - Summary text reports
  - CSV exports for further analysis
  - Bar charts and visualizations

Usage:
    from src.dataset_reporting import DatasetReporter
    
    reporter = DatasetReporter(
        signals=signals,           # List/array of signal arrays
        labels=labels,             # Class labels
        signal_names=['Signal 1',  # Optional signal identifiers
                      'Signal 2'],
        class_names=['Class 0', 'Class 1', 'Class 2'],  # Optional
        snr_values=snr_array       # Optional SNR values in dB
    )
    
    # Generate all reports
    reporter.generate_all_reports(output_dir="outputs/reports")
    
    # Or individual reports
    stats = reporter.compute_statistics()
    reporter.save_text_report(output_file="outputs/reports/dataset_summary.txt")
    reporter.save_csv_summary(output_file="outputs/reports/dataset_summary.csv")
    reporter.plot_distributions(output_file="outputs/plots/dataset_distribution.png")
"""

import os
import csv
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter


class DatasetReporter:
    """Comprehensive dataset analysis and reporting."""

    def __init__(
        self,
        signals: Union[List[np.ndarray], np.ndarray],
        labels: np.ndarray,
        signal_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        snr_values: Optional[np.ndarray] = None,
    ):
        """
        Initialize DatasetReporter.

        Parameters
        ----------
        signals : Union[List[np.ndarray], np.ndarray]
            List or array of signal data. Each signal can be 1D or multi-dimensional.
            - If list: each element is a signal array
            - If array: signals along first axis
        labels : np.ndarray
            Class labels for each signal (shape: (n_samples,))
        signal_names : Optional[List[str]]
            Optional names for signals (for display)
        class_names : Optional[List[str]]
            Optional names for classes. If None, uses 'Class 0', 'Class 1', etc.
        snr_values : Optional[np.ndarray]
            SNR values in dB for each signal (shape: (n_samples,))
            Can be computed if not provided.
        """
        self.signals = self._process_signals(signals)
        self.labels = np.asarray(labels)
        self.snr_values = snr_values

        # Validation
        assert len(self.signals) == len(self.labels), "Signals and labels must have same length"

        # Setup class names
        n_classes = len(np.unique(self.labels))
        if class_names is None:
            self.class_names = [f"Class {i}" for i in range(n_classes)]
        else:
            self.class_names = class_names
            assert len(self.class_names) >= n_classes

        # Setup signal names
        self.signal_names = signal_names

        # Metadata
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(self.signals),
            "n_classes": n_classes,
            "has_snr": snr_values is not None,
        }

    @staticmethod
    def _process_signals(signals: Union[List, np.ndarray]) -> List[np.ndarray]:
        """Convert signals to list of numpy arrays."""
        if isinstance(signals, list):
            return [np.asarray(s) for s in signals]
        elif isinstance(signals, np.ndarray):
            if signals.ndim == 1:
                return [signals]  # Single signal
            else:
                return [signals[i] for i in range(len(signals))]
        else:
            raise ValueError("Signals must be list or numpy array")

    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive dataset statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - class_distribution: Counter of class frequencies
            - signal_lengths: Statistics on signal lengths
            - signal_total_samples: Statistics on total samples per signal
            - snr_stats: Statistics on SNR (if available)
            - per_class_stats: Per-class breakdowns
        """
        stats = {}

        # 1. Class distribution
        class_counts = Counter(self.labels)
        stats["class_distribution"] = dict(
            sorted(class_counts.items())
        )

        # 2. Signal length statistics
        signal_lengths = np.array([len(s) for s in self.signals])
        stats["signal_lengths"] = {
            "min": int(signal_lengths.min()),
            "max": int(signal_lengths.max()),
            "mean": float(signal_lengths.mean()),
            "median": float(np.median(signal_lengths)),
            "std": float(signal_lengths.std()),
            "total_samples": int(signal_lengths.sum()),
        }

        # 3. Total samples per signal (for multi-dimensional)
        total_per_signal = np.array([np.prod(s.shape) for s in self.signals])
        stats["signal_total_samples"] = {
            "min": int(total_per_signal.min()),
            "max": int(total_per_signal.max()),
            "mean": float(total_per_signal.mean()),
            "median": float(np.median(total_per_signal)),
            "std": float(total_per_signal.std()),
        }

        # 4. SNR statistics (if provided)
        if self.snr_values is not None:
            snr = np.asarray(self.snr_values)
            stats["snr_stats"] = {
                "min": float(snr.min()),
                "max": float(snr.max()),
                "mean": float(snr.mean()),
                "median": float(np.median(snr)),
                "std": float(snr.std()),
                "q1": float(np.percentile(snr, 25)),
                "q3": float(np.percentile(snr, 75)),
            }

        # 5. Per-class statistics
        per_class = {}
        for class_idx in sorted(np.unique(self.labels)):
            class_mask = self.labels == class_idx
            class_signals = [s for s, m in zip(self.signals, class_mask) if m]
            class_lengths = np.array([len(s) for s in class_signals])

            per_class_stat = {
                "count": int(np.sum(class_mask)),
                "percentage": float(100 * np.sum(class_mask) / len(self.labels)),
                "signal_lengths": {
                    "min": int(class_lengths.min()),
                    "max": int(class_lengths.max()),
                    "mean": float(class_lengths.mean()),
                },
            }

            if self.snr_values is not None:
                class_snr = self.snr_values[class_mask]
                per_class_stat["snr"] = {
                    "min": float(class_snr.min()),
                    "max": float(class_snr.max()),
                    "mean": float(class_snr.mean()),
                }

            per_class[self.class_names[class_idx]] = per_class_stat

        stats["per_class"] = per_class

        return stats

    def generate_text_report(self) -> str:
        """
        Generate human-readable text report.

        Returns
        -------
        str
            Formatted text report
        """
        stats = self.compute_statistics()
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("DATASET SUMMARY REPORT".center(80))
        lines.append("=" * 80)
        lines.append(f"\nGenerated: {self.metadata['timestamp']}")
        lines.append(f"Total Samples: {self.metadata['n_samples']:,}")
        lines.append(f"Number of Classes: {self.metadata['n_classes']}")
        lines.append(f"SNR Data Available: {'Yes' if self.metadata['has_snr'] else 'No'}\n")

        # Class Distribution
        lines.append("─" * 80)
        lines.append("CLASS DISTRIBUTION")
        lines.append("─" * 80)
        for class_idx, (class_name, per_class_stat) in enumerate(stats["per_class"].items()):
            count = per_class_stat["count"]
            percentage = per_class_stat["percentage"]
            lines.append(f"{class_name:20} {count:6,} samples ({percentage:5.1f}%)")

        # Signal Length Statistics
        lines.append("\n" + "─" * 80)
        lines.append("SIGNAL LENGTH STATISTICS (Primary Dimension)")
        lines.append("─" * 80)
        sig_len = stats["signal_lengths"]
        lines.append(f"Minimum:       {sig_len['min']:,} samples")
        lines.append(f"Maximum:       {sig_len['max']:,} samples")
        lines.append(f"Mean:          {sig_len['mean']:.1f} samples")
        lines.append(f"Median:        {sig_len['median']:.1f} samples")
        lines.append(f"Std Dev:       {sig_len['std']:.1f} samples")
        lines.append(f"Total:         {sig_len['total_samples']:,} samples")

        # Total Samples per Signal
        lines.append("\n" + "─" * 80)
        lines.append("TOTAL SAMPLES PER SIGNAL (All Dimensions)")
        lines.append("─" * 80)
        tot_samp = stats["signal_total_samples"]
        lines.append(f"Minimum:       {tot_samp['min']:,} values")
        lines.append(f"Maximum:       {tot_samp['max']:,} values")
        lines.append(f"Mean:          {tot_samp['mean']:,.0f} values")
        lines.append(f"Median:        {tot_samp['median']:,.0f} values")
        lines.append(f"Std Dev:       {tot_samp['std']:,.0f} values")

        # SNR Statistics (if available)
        if self.metadata["has_snr"]:
            lines.append("\n" + "─" * 80)
            lines.append("SNR STATISTICS")
            lines.append("─" * 80)
            snr_stat = stats["snr_stats"]
            lines.append(f"Minimum:       {snr_stat['min']:.2f} dB")
            lines.append(f"Maximum:       {snr_stat['max']:.2f} dB")
            lines.append(f"Mean:          {snr_stat['mean']:.2f} dB")
            lines.append(f"Median:        {snr_stat['median']:.2f} dB")
            lines.append(f"Std Dev:       {snr_stat['std']:.2f} dB")
            lines.append(f"Q1 (25%):      {snr_stat['q1']:.2f} dB")
            lines.append(f"Q3 (75%):      {snr_stat['q3']:.2f} dB")

        # Per-Class Detailed Statistics
        lines.append("\n" + "─" * 80)
        lines.append("PER-CLASS DETAILED STATISTICS")
        lines.append("─" * 80)

        for class_name, per_class_stat in stats["per_class"].items():
            lines.append(f"\n{class_name}:")
            lines.append(f"  Count:           {per_class_stat['count']:,} samples")
            lines.append(
                f"  Percentage:      {per_class_stat['percentage']:.1f}% of dataset"
            )

            sig_len_class = per_class_stat["signal_lengths"]
            lines.append(
                f"  Signal Length:   "
                f"min={sig_len_class['min']:,}, "
                f"max={sig_len_class['max']:,}, "
                f"mean={sig_len_class['mean']:.1f}"
            )

            if "snr" in per_class_stat:
                snr_class = per_class_stat["snr"]
                lines.append(
                    f"  SNR:             "
                    f"min={snr_class['min']:.2f} dB, "
                    f"max={snr_class['max']:.2f} dB, "
                    f"mean={snr_class['mean']:.2f} dB"
                )

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def save_text_report(self, output_file: str) -> None:
        """
        Save text report to file.

        Parameters
        ----------
        output_file : str
            Path to output text file
        """
        report = self.generate_text_report()
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(report)

    def save_csv_summary(self, output_file: str) -> None:
        """
        Save CSV summary of statistics.

        Parameters
        ----------
        output_file : str
            Path to output CSV file
        """
        stats = self.compute_statistics()
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Header rows
            writer.writerow(["Dataset Statistics Summary"])
            writer.writerow(["Generated", self.metadata["timestamp"]])
            writer.writerow(["Total Samples", self.metadata["n_samples"]])
            writer.writerow(["Number of Classes", self.metadata["n_classes"]])
            writer.writerow(["SNR Available", "Yes" if self.metadata["has_snr"] else "No"])
            writer.writerow([])

            # Class Distribution
            writer.writerow(["Class Distribution"])
            writer.writerow(["Class", "Count", "Percentage"])
            for class_name, per_class_stat in stats["per_class"].items():
                writer.writerow(
                    [
                        class_name,
                        per_class_stat["count"],
                        f"{per_class_stat['percentage']:.2f}%",
                    ]
                )
            writer.writerow([])

            # Signal Length Statistics
            writer.writerow(["Signal Length Statistics"])
            writer.writerow(["Metric", "Value"])
            sig_len = stats["signal_lengths"]
            for key, value in sig_len.items():
                writer.writerow([key, value])
            writer.writerow([])

            # Total Samples Statistics
            writer.writerow(["Total Samples Per Signal"])
            writer.writerow(["Metric", "Value"])
            tot_samp = stats["signal_total_samples"]
            for key, value in tot_samp.items():
                writer.writerow([key, value])
            writer.writerow([])

            # SNR Statistics
            if self.metadata["has_snr"]:
                writer.writerow(["SNR Statistics (dB)"])
                writer.writerow(["Metric", "Value"])
                snr_stat = stats["snr_stats"]
                for key, value in snr_stat.items():
                    writer.writerow([key, f"{value:.4f}"])

    def plot_distributions(
        self,
        output_file: str = "outputs/plots/dataset_distribution.png",
        figsize: Tuple[int, int] = (16, 10),
        dpi: int = 300,
    ) -> None:
        """
        Generate comprehensive distribution plots.

        Parameters
        ----------
        output_file : str
            Path to output PNG file
        figsize : Tuple[int, int]
            Figure size (width, height)
        dpi : int
            Resolution in dots per inch
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        stats = self.compute_statistics()

        # 1. Class Distribution (Bar Chart)
        ax1 = fig.add_subplot(gs[0, 0])
        class_counts = list(stats["class_distribution"].values())
        class_labels = [
            self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            for i in sorted(stats["class_distribution"].keys())
        ]
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_labels)))
        bars1 = ax1.bar(class_labels, class_counts, color=colors, edgecolor="black", linewidth=1.5)
        ax1.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
        ax1.set_title("Class Distribution", fontsize=12, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 2. Class Distribution (Pie Chart)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.pie(
            class_counts,
            labels=class_labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax2.set_title("Class Distribution (%)", fontsize=12, fontweight="bold")

        # 3. Signal Length Distribution (Histogram)
        ax3 = fig.add_subplot(gs[0, 2])
        signal_lengths = np.array([len(s) for s in self.signals])
        ax3.hist(signal_lengths, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        ax3.axvline(
            signal_lengths.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {signal_lengths.mean():.1f}",
        )
        ax3.axvline(
            np.median(signal_lengths),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(signal_lengths):.1f}",
        )
        ax3.set_xlabel("Signal Length (samples)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax3.set_title("Signal Length Distribution", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3, linestyle="--")

        # 4. Per-Class Signal Length Box Plot
        ax4 = fig.add_subplot(gs[1, 0])
        class_signal_lengths = []
        for class_idx in sorted(np.unique(self.labels)):
            class_mask = self.labels == class_idx
            class_lengths = np.array([s for s, m in zip(signal_lengths, class_mask) if m])
            class_signal_lengths.append(class_lengths)

        bp = ax4.boxplot(
            class_signal_lengths,
            labels=class_labels,
            patch_artist=True,
            medianprops=dict(color="red", linewidth=2),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax4.set_ylabel("Signal Length (samples)", fontsize=11, fontweight="bold")
        ax4.set_title("Signal Length by Class", fontsize=12, fontweight="bold")
        ax4.grid(axis="y", alpha=0.3, linestyle="--")

        # 5. SNR Distribution (if available)
        ax5 = fig.add_subplot(gs[1, 1])
        if self.metadata["has_snr"]:
            snr = np.asarray(self.snr_values)
            ax5.hist(snr, bins=30, color="coral", edgecolor="black", alpha=0.7)
            ax5.axvline(
               snr.mean(),
                color="darkred",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {snr.mean():.2f} dB",
            )
            ax5.axvline(
                np.median(snr),
                color="darkgreen",
                linestyle="--",
                linewidth=2,
                label=f"Median: {np.median(snr):.2f} dB",
            )
            ax5.set_xlabel("SNR (dB)", fontsize=11, fontweight="bold")
            ax5.set_ylabel("Frequency", fontsize=11, fontweight="bold")
            ax5.set_title("SNR Distribution", fontsize=12, fontweight="bold")
            ax5.legend(fontsize=9)
            ax5.grid(alpha=0.3, linestyle="--")
        else:
            ax5.text(
                0.5,
                0.5,
                "SNR Data\nNot Available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax5.transAxes,
            )
            ax5.set_title("SNR Distribution", fontsize=12, fontweight="bold")
            ax5.axis("off")

        # 6. Per-Class SNR Box Plot (if available)
        ax6 = fig.add_subplot(gs[1, 2])
        if self.metadata["has_snr"]:
            class_snr_values = []
            for class_idx in sorted(np.unique(self.labels)):
                class_mask = self.labels == class_idx
                class_snr = self.snr_values[class_mask]
                class_snr_values.append(class_snr)

            bp2 = ax6.boxplot(
                class_snr_values,
                labels=class_labels,
                patch_artist=True,
                medianprops=dict(color="red", linewidth=2),
            )
            for patch, color in zip(bp2["boxes"], colors):
                patch.set_facecolor(color)
            ax6.set_ylabel("SNR (dB)", fontsize=11, fontweight="bold")
            ax6.set_title("SNR by Class", fontsize=12, fontweight="bold")
            ax6.grid(axis="y", alpha=0.3, linestyle="--")
        else:
            ax6.text(
                0.5,
                0.5,
                "SNR Data\nNot Available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax6.transAxes,
            )
            ax6.set_title("SNR by Class", fontsize=12, fontweight="bold")
            ax6.axis("off")

        # Overall title with metadata
        fig.suptitle(
            f"Dataset Analysis Report - {self.metadata['n_samples']:,} samples, {self.metadata['n_classes']} classes",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close()

    def generate_all_reports(
        self,
        output_dir: str = "outputs/reports",
        plot_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate all reports (text, CSV, plots).

        Parameters
        ----------
        output_dir : str
            Directory for text and CSV files
        plot_dir : Optional[str]
            Directory for plot files. If None, uses output_dir/../plots

        Returns
        -------
        Dict[str, str]
            Dictionary with paths to generated files
        """
        if plot_dir is None:
            plot_dir = os.path.join(os.path.dirname(output_dir), "plots")

        text_file = os.path.join(output_dir, "dataset_summary.txt")
        csv_file = os.path.join(output_dir, "dataset_summary.csv")
        plot_file = os.path.join(plot_dir, "dataset_distribution.png")

        print(f"📊 Generating dataset reports...")
        self.save_text_report(text_file)
        print(f"  ✓ Text report: {text_file}")

        self.save_csv_summary(csv_file)
        print(f"  ✓ CSV summary: {csv_file}")

        self.plot_distributions(plot_file)
        print(f"  ✓ Distribution plots: {plot_file}")

        return {
            "text_report": text_file,
            "csv_summary": csv_file,
            "plots": plot_file,
        }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DATASET REPORTING UTILITY - EXAMPLE".center(80))
    print("=" * 80 + "\n")

    # Generate synthetic dataset
    print("📊 Creating synthetic dataset...")
    np.random.seed(42)
    n_samples = 200
    n_classes = 3

    # Generate signals with varying lengths
    signals = []
    labels = []
    snr_values = []

    samples_per_class = n_samples // n_classes
    for class_idx in range(n_classes):
        for _ in range(samples_per_class):
            # Random signal length between 100 and 500 samples
            length = np.random.randint(100, 501)
            signal = np.random.randn(length)  # Gaussian noise
            signals.append(signal)
            labels.append(class_idx)

            # Random SNR between -10 dB and 30 dB
            snr = np.random.uniform(-10, 30)
            snr_values.append(snr)

    labels = np.array(labels)
    snr_values = np.array(snr_values)

    print(f"✓ Generated {len(signals)} signals, {n_classes} classes")

    # Create reporter
    reporter = DatasetReporter(
        signals=signals,
        labels=labels,
        class_names=["Target A", "Target B", "Clutter"],
        snr_values=snr_values,
    )

    # Generate all reports
    output_paths = reporter.generate_all_reports(
        output_dir="outputs/reports", plot_dir="outputs/plots"
    )

    print("\n" + "─" * 80)
    print("TEXT REPORT PREVIEW:")
    print("─" * 80)
    print(reporter.generate_text_report())

    print("\n" + "=" * 80)
    print("✅ ALL REPORTS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print("\n📁 Output Files:")
    for file_type, path in output_paths.items():
        print(f"   {file_type}: {path}")
    print("\n" + "=" * 80 + "\n")
