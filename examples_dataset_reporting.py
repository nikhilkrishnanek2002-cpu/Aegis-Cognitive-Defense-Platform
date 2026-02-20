"""
Dataset Reporting Utility - Integration Examples

Demonstrates how to use the dataset reporting module with:
  1. Synthetic radar signal datasets
  2. Real data from PyTorch dataloaders
  3. Integration with training pipelines
  4. Custom analysis workflows
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def example_1_basic_usage():
    """
    Example 1: Basic dataset analysis
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: BASIC DATASET ANALYSIS")
    print("=" * 80)

    from src.dataset_reporting import DatasetReporter

    # Create synthetic dataset
    print("\n📊 Creating synthetic radar dataset...")
    np.random.seed(42)
    n_samples = 300
    n_classes = 3

    signals = []
    labels = []
    snr_values = []

    for class_idx in range(n_classes):
        n_per_class = n_samples // n_classes
        for _ in range(n_per_class):
            # Variable length signals
            length = np.random.randint(80, 600)
            signal = np.random.randn(length)
            signals.append(signal)
            labels.append(class_idx)

            # SNR distribution
            snr = np.random.uniform(-5, 35)
            snr_values.append(snr)

    labels = np.array(labels)
    snr_values = np.array(snr_values)

    print(f"✓ Generated {len(signals)} signals with {n_classes} classes")

    # Create reporter
    reporter = DatasetReporter(
        signals=signals,
        labels=labels,
        class_names=["Fighter Jet", "Helicopter", "Clutter"],
        snr_values=snr_values,
    )

    # Generate reports
    print("\n📝 Generating reports...")
    output_paths = reporter.generate_all_reports(
        output_dir="outputs/reports",
        plot_dir="outputs/plots"
    )

    print("\n✓ Reports generated to:")
    for file_type, path in output_paths.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   {file_type}: {path} ({size:,} bytes)")

    return reporter, output_paths


def example_2_statistics_only():
    """
    Example 2: Generate statistics without plots
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: COMPUTE STATISTICS FOR CUSTOM ANALYSIS")
    print("=" * 80)

    from src.dataset_reporting import DatasetReporter

    np.random.seed(123)
    n_samples = 150
    n_classes = 2

    # Binary classification dataset
    signals = [np.random.randn(np.random.randint(100, 400)) for _ in range(n_samples)]
    labels = np.repeat([0, 1], n_samples // 2)
    snr_values = np.random.uniform(0, 25, n_samples)

    reporter = DatasetReporter(
        signals=signals,
        labels=labels,
        class_names=["Signal", "Noise"],
        snr_values=snr_values,
    )

    # Get statistics
    stats = reporter.compute_statistics()

    print("\n📊 Dataset Statistics:")
    print(f"   Total samples: {stats['class_distribution']}")
    print(f"   Signal lengths: min={stats['signal_lengths']['min']}, "
          f"max={stats['signal_lengths']['max']}, "
          f"mean={stats['signal_lengths']['mean']:.1f}")
    print(f"   SNR: min={stats['snr_stats']['min']:.2f} dB, "
          f"mean={stats['snr_stats']['mean']:.2f} dB, "
          f"max={stats['snr_stats']['max']:.2f} dB")

    # Per-class analysis
    print("\n📊 Per-Class Breakdown:")
    for class_name, class_stat in stats["per_class"].items():
        print(f"\n   {class_name}:")
        print(f"      Samples: {class_stat['count']} ({class_stat['percentage']:.1f}%)")
        print(f"      Signal length: {class_stat['signal_lengths']['mean']:.1f} avg")
        if "snr" in class_stat:
            print(f"      SNR: {class_stat['snr']['mean']:.2f} dB avg")

    return stats


def example_3_just_plots():
    """
    Example 3: Generate only visualization
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: GENERATE VISUALIZATION ONLY")
    print("=" * 80)

    from src.dataset_reporting import DatasetReporter

    np.random.seed(456)
    n_samples = 250
    n_classes = 4

    # 4-class problem
    signals = [np.random.randn(np.random.randint(150, 500)) for _ in range(n_samples)]
    labels = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
    snr_values = np.random.uniform(-10, 30, n_samples)

    reporter = DatasetReporter(
        signals=signals,
        labels=labels,
        class_names=["Class A", "Class B", "Class C", "Class D"],
        snr_values=snr_values,
    )

    print("\n📊 Generating distribution plots...")
    plot_file = "outputs/plots/dataset_distribution_example3.png"
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    reporter.plot_distributions(output_file=plot_file, figsize=(14, 9), dpi=200)
    print(f"✓ Plots saved to: {plot_file}")
    print(f"✓ File size: {os.path.getsize(plot_file):,} bytes")


def example_4_text_report_only():
    """
    Example 4: Generate text report for inspection
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: TEXT REPORT FOR QUICK REVIEW")
    print("=" * 80)

    from src.dataset_reporting import DatasetReporter

    np.random.seed(789)
    n_samples = 100
    n_classes = 3

    signals = [np.random.randn(np.random.randint(100, 400)) for _ in range(n_samples)]
    labels = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
    snr_values = np.random.uniform(-5, 30, n_samples)

    reporter = DatasetReporter(
        signals=signals,
        labels=labels,
        class_names=["Radar A", "Radar B", "Radar C"],
        snr_values=snr_values,
    )

    # Generate and print text report
    print("\n📄 Dataset Report:\n")
    report_text = reporter.generate_text_report()
    print(report_text)


def example_5_without_snr():
    """
    Example 5: Analysis without SNR data
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: ANALYSIS WITHOUT SNR DATA")
    print("=" * 80)

    from src.dataset_reporting import DatasetReporter

    np.random.seed(999)
    n_samples = 200
    n_classes = 2

    signals = [np.random.randn(np.random.randint(100, 500)) for _ in range(n_samples)]
    labels = np.repeat([0, 1], n_samples // 2)

    print("\n📊 Creating reporter without SNR data...")
    reporter = DatasetReporter(
        signals=signals,
        labels=labels,
        class_names=["Positive", "Negative"],
        snr_values=None  # No SNR data
    )

    stats = reporter.compute_statistics()

    print(f"✓ Dataset created: {stats['class_distribution']}")
    print(f"✓ Signal lengths: min={stats['signal_lengths']['min']}, "
          f"max={stats['signal_lengths']['max']}")
    print(f"✓ SNR available: {bool('snr_stats' in stats)}")

    # Generate reports (SNR plots will show "Not Available")
    print("\n📝 Generating reports...")
    output_paths = reporter.generate_all_reports(
        output_dir="outputs/reports",
        plot_dir="outputs/plots"
    )
    print(f"✓ Reports generated successfully")


def example_6_code_template():
    """
    Example 6: Code template for integration
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: INTEGRATION CODE TEMPLATES")
    print("=" * 80)

    print("\n📌 Template 1: Basic Usage with NumPy Arrays")
    print("""
from src.dataset_reporting import DatasetReporter

# Assuming you have signals and labels
# signals = [array, array, ...]  # List of signal arrays
# labels = np.array([0, 1, 2, ...])
# snr_values = np.array([10.5, 12.3, ...])

reporter = DatasetReporter(
    signals=signals,
    labels=labels,
    class_names=['Class A', 'Class B', 'Class C'],
    snr_values=snr_values
)

# Generate all reports
reporter.generate_all_reports(
    output_dir="outputs/reports",
    plot_dir="outputs/plots"
)
    """)

    print("\n📌 Template 2: Integration with PyTorch DataLoader")
    print("""
from src.dataset_reporting import DatasetReporter

# Extract data from DataLoader
signals = []
labels = []
snr_vals = []

for batch in dataloader:
    if len(batch) == 3:
        signal, label, snr = batch
    else:
        signal, label = batch
        snr = None
    
    signals.extend(signal.cpu().numpy())
    labels.extend(label.cpu().numpy())
    if snr is not None:
        snr_vals.extend(snr.cpu().numpy())

# Create reporter
reporter = DatasetReporter(
    signals=signals,
    labels=labels,
    snr_values=np.array(snr_vals) if snr_vals else None
)

reporter.generate_all_reports()
    """)

    print("\n📌 Template 3: Custom Statistics Analysis")
    print("""
from src.dataset_reporting import DatasetReporter

reporter = DatasetReporter(signals, labels, snr_values=snr_values)
stats = reporter.compute_statistics()

# Find problematic classes
for class_name, stat in stats['per_class'].items():
    if stat['count'] < 10:
        print(f"Warning: {class_name} has only {stat['count']} samples")
    
    if stat['signal_lengths']['std'] > 200:
        print(f"Note: {class_name} has high signal length variance")

# Export specific statistics
import json
with open('dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
    """)

    print("\n📌 Template 4: Skip Plots, Generate Text Report")
    print("""
from src.dataset_reporting import DatasetReporter

reporter = DatasetReporter(signals, labels, snr_values=snr_values)

# Just generate text and CSV (faster)
reporter.save_text_report("outputs/reports/dataset_summary.txt")
reporter.save_csv_summary("outputs/reports/dataset_summary.csv")

# Or read the text report
report = reporter.generate_text_report()
print(report)
    """)


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "DATASET REPORTING UTILITY - INTEGRATION EXAMPLES".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # Create output directories
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    # Run examples
    print("\n" + "─" * 80)
    example_1_basic_usage()

    print("\n" + "─" * 80)
    example_2_statistics_only()

    print("\n" + "─" * 80)
    example_3_just_plots()

    print("\n" + "─" * 80)
    example_4_text_report_only()

    print("\n" + "─" * 80)
    example_5_without_snr()

    print("\n" + "─" * 80)
    example_6_code_template()

    print("\n" + "=" * 80)
    print("✅ ALL EXAMPLES COMPLETED".center(80))
    print("=" * 80)
    print("\n📁 Output Files Generated:")
    print("   outputs/reports/dataset_summary.txt")
    print("   outputs/reports/dataset_summary.csv")
    print("   outputs/plots/dataset_distribution.png")
    print("   outputs/plots/dataset_distribution_example3.png")
    print("\n📖 For more information, see: DATASET_REPORTING_GUIDE.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
