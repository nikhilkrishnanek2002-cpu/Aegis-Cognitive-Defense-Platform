#!/usr/bin/env python3
"""
Optimized training script for Cognitive Photonic Radar AI
Demonstrates all fixes: metadata normalization, batch norm, gradient clipping
Expected accuracy: >90% on 6-class classification
"""

import os
import sys
import yaml
import argparse
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import cv2
from sklearn.preprocessing import StandardScaler

from src.train_pytorch import train_pytorch_model
from src.signal_generator import generate_radar_signal
from src.feature_extractor import get_all_features
from src.evaluation_enhanced import compute_comprehensive_metrics, get_metrics_summary


def _augment_radar_images(rd: np.ndarray, spec: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    # Small spatial shifts and Gaussian noise to improve generalization
    shift_h = rng.randint(-2, 3)
    shift_w = rng.randint(-2, 3)
    rd_aug = np.roll(rd, shift=(shift_h, shift_w), axis=(0, 1))
    spec_aug = np.roll(spec, shift=(shift_h, shift_w), axis=(0, 1))
    if rng.rand() < 0.3:
        rd_aug = np.flipud(rd_aug)
        spec_aug = np.flipud(spec_aug)
    if rng.rand() < 0.3:
        rd_aug = np.fliplr(rd_aug)
        spec_aug = np.fliplr(spec_aug)
    rd_aug = np.clip(rd_aug + rng.normal(0.0, 0.01, rd_aug.shape), 0.0, 1.0)
    spec_aug = np.clip(spec_aug + rng.normal(0.0, 0.01, spec_aug.shape), 0.0, 1.0)
    return np.ascontiguousarray(rd_aug), np.ascontiguousarray(spec_aug)


def build_train_val_datasets(
    samples_per_class: int,
    train_split: float,
    seed: int,
    augmentation_factor: int = 1,
    sim_params: dict | None = None,
    meta_noise_std: float = 0.0,
) -> Tuple[TensorDataset, TensorDataset]:
    classes = ["drone", "aircraft", "bird", "helicopter", "missile", "clutter"]
    rd_list, spec_list, meta_list, y_list = [], [], [], []

    print("Generating simulated photonic radar dataset...")
    for label, cls in enumerate(classes):
        for _ in range(samples_per_class):
            if sim_params:
                sig = generate_radar_signal(cls, **sim_params)
            else:
                sig = generate_radar_signal(cls)
            rd, spec, meta, _ = get_all_features(sig)

            # Resize to match model input
            rd = cv2.resize(rd, (128, 128))
            spec = cv2.resize(spec, (128, 128))

            # Normalize images to [0, 1]
            rd = rd / (np.max(rd) + 1e-8)
            spec = spec / (np.max(spec) + 1e-8)

            rd_list.append(rd)
            spec_list.append(spec)
            meta_list.append(meta)
            y_list.append(label)

    rd_array = np.array(rd_list, dtype=np.float32)
    spec_array = np.array(spec_list, dtype=np.float32)
    meta_array = np.array(meta_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.int64)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(y_array))
    train_size = int(len(indices) * train_split)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    # Fit scaler on train metadata only, then apply to both splits
    scaler = StandardScaler()
    meta_train = scaler.fit_transform(meta_array[train_idx]).astype(np.float32)
    meta_val = scaler.transform(meta_array[val_idx]).astype(np.float32)

    rd_train = rd_array[train_idx]
    spec_train = spec_array[train_idx]
    y_train = y_array[train_idx]

    if augmentation_factor > 1:
        rng = np.random.RandomState(seed)
        aug_rd = [rd_train]
        aug_spec = [spec_train]
        aug_meta = [meta_train]
        aug_y = [y_train]
        for _ in range(augmentation_factor - 1):
            rd_aug_list = []
            spec_aug_list = []
            for i in range(len(rd_train)):
                rd_aug, spec_aug = _augment_radar_images(rd_train[i], spec_train[i], rng)
                rd_aug_list.append(rd_aug)
                spec_aug_list.append(spec_aug)
            aug_rd.append(np.array(rd_aug_list, dtype=np.float32))
            aug_spec.append(np.array(spec_aug_list, dtype=np.float32))
            aug_meta.append(meta_train)
            aug_y.append(y_train)

        rd_train = np.concatenate(aug_rd, axis=0)
        spec_train = np.concatenate(aug_spec, axis=0)
        meta_train = np.concatenate(aug_meta, axis=0)
        y_train = np.concatenate(aug_y, axis=0)

    if meta_noise_std > 0.0:
        rng = np.random.RandomState(seed + 1)
        meta_train = meta_train + rng.normal(0.0, meta_noise_std, meta_train.shape).astype(np.float32)

    train_dataset = TensorDataset(
        torch.tensor(rd_train, dtype=torch.float32),
        torch.tensor(spec_train, dtype=torch.float32),
        torch.tensor(meta_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(rd_array[val_idx], dtype=torch.float32),
        torch.tensor(spec_array[val_idx], dtype=torch.float32),
        torch.tensor(meta_val, dtype=torch.float32),
        torch.tensor(y_array[val_idx], dtype=torch.long),
    )

    return train_dataset, val_dataset


def evaluate_on_loader(model, loader, device: str, output_dir: str) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for b_rd, b_spec, b_meta, b_y in loader:
            b_rd = b_rd.to(device)
            b_spec = b_spec.to(device)
            b_meta = b_meta.to(device)
            outputs = model(b_rd, b_spec, b_meta)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(b_y.numpy())
            all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)

    metrics = compute_comprehensive_metrics(
        predictions=y_pred,
        labels=y_true,
        probabilities=y_probs,
        output_dir=output_dir,
        model_name="optimized_training",
        num_classes=6,
    )
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train Photonic Radar AI with optimizations")
    parser.add_argument("--config", type=str, default="experiments/optimized.yaml",
                        help="Path to experiment config")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda or cpu (auto for auto-detect)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Override samples per class from config")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract training params
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})
    exp_cfg = config.get('experiment', {})
    
    samples_per_class = args.samples_per_class or data_cfg.get('samples_per_class', 200)
    train_split = data_cfg.get('train_split', 0.8)
    augmentation_factor = int(data_cfg.get('augmentation_factor', 1))
    sim_params = data_cfg.get('simulation', {})
    meta_noise_std = float(sim_params.pop('meta_noise_std', 0.0))
    epochs = args.epochs or train_cfg.get('epochs', 50)
    batch_size = args.batch_size or train_cfg.get('batch_size', 32)
    learning_rate = train_cfg.get('learning_rate', 0.0005)
    weight_decay = float(train_cfg.get('weight_decay', 0.0))
    lr_schedule = train_cfg.get('lr_schedule')
    label_smoothing = float(train_cfg.get('label_smoothing', 0.0))
    early_stopping = train_cfg.get('early_stopping', {})
    early_stopping_patience = int(early_stopping.get('patience', 0))
    early_stopping_min_delta = float(early_stopping.get('min_delta', 0.0))
    restore_best = bool(early_stopping.get('restore_best', True))
    mixup_cfg = train_cfg.get('mixup', {})
    mixup_alpha = float(mixup_cfg.get('alpha', 0.0))
    cutmix_cfg = train_cfg.get('cutmix', {})
    cutmix_alpha = float(cutmix_cfg.get('alpha', 0.0))
    cutmix_prob = float(cutmix_cfg.get('prob', 0.5))
    seed = exp_cfg.get('seed', 42)
    output_dir = os.path.join(exp_cfg.get('output_root', 'outputs'), 'training')
    reports_dir = os.path.join(exp_cfg.get('output_root', 'outputs'), 'reports')
    
    # Auto-detect device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*70)
    print("[*] OPTIMIZED PHOTONIC RADAR AI TRAINING")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"  Samples/class: {samples_per_class}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Label smoothing: {label_smoothing}")
    if lr_schedule:
        print(f"  LR schedule: {lr_schedule}")
    if early_stopping_patience > 0:
        print(f"  Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    if mixup_alpha > 0.0:
        print(f"  MixUp: alpha={mixup_alpha}")
    if cutmix_alpha > 0.0:
        print(f"  CutMix: alpha={cutmix_alpha}, prob={cutmix_prob}")
    print(f"  Train split: {train_split}")
    print(f"  Augmentation factor: {augmentation_factor}")
    if sim_params:
        print(f"  Simulation params: {sim_params}")
    if meta_noise_std > 0.0:
        print(f"  Metadata noise std: {meta_noise_std}")
    print(f"  Device: {device}")
    print(f"  Seed: {seed}")
    print("\n[+] Improvements applied:")
    print("  [OK] Metadata normalization (StandardScaler)")
    print("  [OK] Batch normalization in CNN branches")
    print("  [OK] Layer normalization in metadata branch")
    print("  [OK] Gradient clipping (norm=1.0)")
    print("  [OK] Realistic clutter signal generation")
    print("  [OK] Increased training data (config-driven)")
    print("  [OK] Class-weighted loss and label smoothing")
    print("  [OK] Augmentation and regularization")
    print("="*70 + "\n")
    
    # Build train/validation datasets
    train_dataset, val_dataset = build_train_val_datasets(
        samples_per_class=samples_per_class,
        train_split=train_split,
        seed=seed,
        augmentation_factor=augmentation_factor,
        sim_params=sim_params,
        meta_noise_std=meta_noise_std,
    )

    # Compute class weights from training labels to reduce class bias
    train_labels = train_dataset.tensors[3].numpy()
    class_counts = np.bincount(train_labels, minlength=6).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Train model
    model, history = train_pytorch_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        samples_per_class=samples_per_class,
        output_dir=output_dir,
        seed=seed,
        device=device,
        dataset=train_dataset,
        val_dataset=val_dataset,
        class_weights=class_weights,
        weight_decay=weight_decay,
        lr_schedule=lr_schedule,
        label_smoothing=label_smoothing,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        restore_best=restore_best,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        cutmix_prob=cutmix_prob,
    )
    
    # Print final results
    print("\n" + "="*70)
    print("[*] FINAL RESULTS")
    print("="*70)
    final_acc = history['accuracy'][-1] if history['accuracy'] else 0
    best_loss = history.get('best_loss', float('inf'))
    best_epoch = history.get('best_epoch', 0)
    
    print(f"Final Training Accuracy: {final_acc:.2%}")
    print(f"Best Loss: {best_loss:.4f} (Epoch {best_epoch})")
    
    if history.get('val_accuracy'):
        final_val_acc = history['val_accuracy'][-1]
        print(f"Final Validation Accuracy: {final_val_acc:.2%}")

    # Evaluate on validation set and save metrics
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    metrics = evaluate_on_loader(model, val_loader, device=device, output_dir=reports_dir)
    print("\nValidation Metrics Summary")
    print(get_metrics_summary(metrics))
    
    print("="*70)
    print(f"[OK] Model saved to: {output_dir}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
