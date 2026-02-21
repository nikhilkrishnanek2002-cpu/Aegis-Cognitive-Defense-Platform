#!/usr/bin/env python3
"""Unified experiment runner for the Aegis Cognitive Defense Platform."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    print("⚠️  PyTorch not installed. This experiment runner requires PyTorch.")
    print("Install with: pip install torch torchvision torchaudio")
    sys.exit(1)

import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from src.reporting import (
    plot_confusion_matrix,
    plot_precision_recall,
    plot_roc_curve,
    plot_training_history,
)
from src.train_pytorch import create_pytorch_dataset, train_pytorch_model
from src.dataset_report import generate_dataset_report


@dataclass
class ExperimentPaths:
    """Container for output directories."""

    base: Path
    models: Path
    logs: Path
    plots: Path
    reports: Path


class ExperimentPipeline:
    """End-to-end experiment orchestrator."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.seed = int(self.config.get("experiment", {}).get("seed", 42))
        self.paths = self._prepare_output_dirs()
        self.logger = self._configure_logging()
        self.device = self._resolve_device()
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.history: Optional[Dict[str, Any]] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.dataset_report = None
        self.eval_targets: Optional[np.ndarray] = None
        self.eval_predictions: Optional[np.ndarray] = None
        self.eval_probabilities: Optional[np.ndarray] = None
        self.num_classes = len(self.config.get("model", {}).get("labels", [])) or 6

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}

    def _set_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _prepare_output_dirs(self) -> ExperimentPaths:
        exp_cfg = self.config.get("experiment", {})
        root = Path(exp_cfg.get("output_root", "outputs"))
        name = exp_cfg.get("name", "experiment")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = root / "experiments" / f"{name}_{timestamp}"
        models = base / "models"
        logs = base / "logs"
        plots = base / "plots"
        reports = base / "reports"
        for path in (base, models, logs, plots, reports):
            path.mkdir(parents=True, exist_ok=True)
        return ExperimentPaths(base, models, logs, plots, reports)

    def _configure_logging(self) -> logging.Logger:
        logger = logging.getLogger("experiment")
        logger.setLevel(getattr(logging, self.config.get("logging", {}).get("level", "INFO")))
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        log_file = self.paths.logs / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info("Experiment initialized")
        logger.info("Config path: %s", self.config_path)
        return logger

    def _resolve_device(self) -> torch.device:
        train_cfg = self.config.get("training", {})
        requested = train_cfg.get("device", "auto")
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    def preprocess(self) -> None:
        data_cfg = self.config.get("data", {})
        samples_per_class = int(data_cfg.get("samples_per_class", 50))
        self.logger.info("Generating synthetic dataset with %s samples/class", samples_per_class)
        rd, spec, meta, labels = create_pytorch_dataset(samples_per_class=samples_per_class)
        dataset = TensorDataset(rd, spec, meta, labels)
        self.num_classes = int(rd.shape[0] / samples_per_class) if samples_per_class else rd.shape[0]

        self.dataset_report = generate_dataset_report(
            signals=[arr for arr in rd.cpu().numpy()],
            labels=labels.cpu().numpy(),
            output_root=self.config.get("experiment", {}).get("output_root", "outputs"),
        )
        self.logger.info("Dataset summary saved to %s", self.dataset_report.text_summary)

        train_ratio = float(data_cfg.get("train_split", 0.8))
        total_samples = len(dataset)
        train_size = max(1, int(total_samples * train_ratio))
        eval_size = total_samples - train_size
        if eval_size <= 0:
            eval_size = 1
            train_size = max(1, total_samples - eval_size)
        generator = torch.Generator().manual_seed(self.seed)
        train_dataset_full, eval_dataset = random_split(dataset, [train_size, eval_size], generator=generator)

        val_ratio = float(self.config.get("training", {}).get("val_split", 0.1))
        if val_ratio > 0 and len(train_dataset_full) > 1:
            val_size = max(1, int(len(train_dataset_full) * val_ratio))
            core_size = max(1, len(train_dataset_full) - val_size)
            generator_val = torch.Generator().manual_seed(self.seed + 1)
            train_dataset, val_dataset = random_split(train_dataset_full, [core_size, val_size], generator=generator_val)
        else:
            train_dataset, val_dataset = train_dataset_full, None

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.eval_dataset = eval_dataset
        self.logger.info("Dataset split -> train: %d, val: %s, test: %d",
                         len(self.train_dataset),
                         len(self.val_dataset) if self.val_dataset else "-",
                         len(self.eval_dataset))

    def train(self) -> torch.nn.Module:
        if self.train_dataset is None:
            raise RuntimeError("Call preprocess() before train().")
        train_cfg = self.config.get("training", {})
        model, history = train_pytorch_model(
            epochs=int(train_cfg.get("epochs", 20)),
            batch_size=int(train_cfg.get("batch_size", 16)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
            samples_per_class=int(self.config.get("data", {}).get("samples_per_class", 50)),
            output_dir=str(self.paths.models),
            seed=self.seed,
            device=str(self.device),
            dataset=self.train_dataset,
            val_dataset=self.val_dataset,
        )
        self.history = history
        # Reload best checkpoint to ensure evaluation uses top-performing weights
        best_path = self.paths.models / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info("Loaded best checkpoint from epoch %s for evaluation", checkpoint.get("epoch"))
        return model

    def evaluate(self, model: torch.nn.Module) -> Dict[str, Any]:
        if self.eval_dataset is None:
            raise RuntimeError("Call preprocess() before evaluate().")
        eval_cfg = self.config.get("evaluation", {})
        batch_size = int(eval_cfg.get("batch_size", self.config.get("training", {}).get("batch_size", 16)))
        loader = DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        y_true, y_pred = [], []
        prob_chunks = []
        with torch.no_grad():
            for b_rd, b_spec, b_meta, b_y in loader:
                b_rd = b_rd.to(self.device)
                b_spec = b_spec.to(self.device)
                b_meta = b_meta.to(self.device)
                outputs = model(b_rd, b_spec, b_meta)
                probs = torch.softmax(outputs, dim=1)
                prob_chunks.append(probs.cpu().numpy())
                predictions = torch.argmax(probs, dim=1).cpu().numpy()
                y_pred.extend(predictions.tolist())
                y_true.extend(b_y.numpy().tolist())
        probabilities = np.concatenate(prob_chunks, axis=0)
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        macro_avg = report_dict.get("macro avg", {})
        weighted_avg = report_dict.get("weighted avg", {})
        cm_normalized = cm.astype(float)
        row_sums = cm_normalized.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm_normalized, row_sums, out=np.zeros_like(cm_normalized), where=row_sums != 0)

        try:
            roc_auc_macro = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
            roc_auc_weighted = roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted")
        except ValueError:
            roc_auc_macro = None
            roc_auc_weighted = None

        self.metrics = {
            "accuracy": accuracy,
            "macro_avg": {
                "precision": macro_avg.get("precision"),
                "recall": macro_avg.get("recall"),
                "f1": macro_avg.get("f1-score"),
            },
            "weighted_avg": {
                "precision": weighted_avg.get("precision"),
                "recall": weighted_avg.get("recall"),
                "f1": weighted_avg.get("f1-score"),
            },
            "cm": cm.tolist(),
            "cm_normalized": cm_normalized.tolist(),
            "roc_auc_macro": roc_auc_macro,
            "roc_auc_weighted": roc_auc_weighted,
            "classification_report": report_dict,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_samples": len(y_true),
                "n_classes": len(np.unique(y_true)),
                "config": str(self.config_path),
            },
        }
        self.eval_targets = np.array(y_true)
        self.eval_predictions = np.array(y_pred)
        self.eval_probabilities = probabilities
        self.logger.info("Evaluation accuracy: %.4f", accuracy)
        return self.metrics

    # ------------------------------------------------------------------
    # Artifact management
    # ------------------------------------------------------------------
    def _save_artifacts(self) -> None:
        if self.history is None or self.metrics is None:
            raise RuntimeError("History and metrics must be available before saving artifacts.")

        # Save structured report
        report_path = self.paths.reports / "metrics.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(self.metrics, handle, indent=2)
        history_path = self.paths.reports / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

        # Generate plots
        plots = {
            "confusion_matrix": self.paths.plots / "confusion_matrix.png",
            "roc_curve": self.paths.plots / "roc_curve.png",
            "precision_recall": self.paths.plots / "precision_recall.png",
            "training_history": self.paths.plots / "training_history.png",
        }

        if self.eval_targets is None or self.eval_predictions is None:
            raise RuntimeError("Evaluation labels missing; ensure evaluate() ran successfully.")

        plot_confusion_matrix(self.eval_targets, self.eval_predictions, str(plots["confusion_matrix"]))

        if self.history.get("val_loss"):
            history_for_plot = self.history
        else:
            history_for_plot = dict(self.history)
            history_for_plot["val_loss"] = history_for_plot["loss"][:]
            accuracy_series = history_for_plot.get("accuracy", [])
            history_for_plot["val_accuracy"] = accuracy_series[:] if accuracy_series else []
        plot_training_history(history_for_plot, str(plots["training_history"]))

        # Prepare flattened labels for ROC/PR plots
        if self.eval_probabilities is not None and self.eval_targets is not None:
            unique_classes = np.unique(self.eval_targets)
            if len(unique_classes) > 1 and self.eval_probabilities.shape[1] > 1:
                binarized = label_binarize(self.eval_targets, classes=np.arange(self.eval_probabilities.shape[1]))
                flat_true = binarized.ravel()
                flat_scores = self.eval_probabilities.ravel()
                plot_roc_curve(flat_true, flat_scores, str(plots["roc_curve"]))
                plot_precision_recall(flat_true, flat_scores, str(plots["precision_recall"]))
            else:
                self.logger.warning("Skipping ROC/PR plots due to insufficient class diversity")

        self._sync_latest_artifacts(report_path, history_path, plots)

    def _sync_latest_artifacts(self, metrics_path: Path, history_path: Path, plots: Dict[str, Path]) -> None:
        canonical_root = Path("outputs")
        canonical_reports = canonical_root / "reports"
        canonical_plots = canonical_root / "plots"
        canonical_models = canonical_root / "models"
        for path in (canonical_reports, canonical_plots, canonical_models):
            path.mkdir(parents=True, exist_ok=True)

        shutil.copy(metrics_path, canonical_reports / "metrics.json")
        shutil.copy(history_path, canonical_reports / "training_history.json")
        for name, plot_path in plots.items():
            dest = canonical_plots / f"{name}.png"
            if plot_path.exists():
                shutil.copy(plot_path, dest)

        for checkpoint in ("best_model.pt", "last_model.pt"):
            src = self.paths.models / checkpoint
            if src.exists():
                shutil.copy(src, canonical_models / checkpoint)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._set_seeds()
        self.preprocess()
        model = self.train()
        self.evaluate(model)
        self._save_artifacts()
        self.logger.info("Experiment completed. Artifacts stored in %s", self.paths.base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible photonic radar experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/baseline.yaml"),
        help="Path to experiment YAML configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = ExperimentPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
