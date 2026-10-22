"""Load local TON_IoT and CICIoT2023 traffic CSVs for Flower."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset


DATASETS = ("ton_iot", "ciciot2023")
DROP_COLUMNS = {"label", "type", "attack", "attack_type", "category", "subcategory",
                "timestamp", "ts", "date", "time", "id", "flow_id", "src_ip", "dst_ip",
                "src_port", "dst_port", "srcip", "dstip", "srcport", "dstport"}


@dataclass
class TrafficData:
    clients: dict[int, DataLoader]
    test: DataLoader
    input_features: int
    train_size: int
    test_size: int
    feature_names: list[str]


def _labels(frame: pd.DataFrame, dataset: str) -> np.ndarray:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    if "label" not in lookup:
        raise ValueError(f"{dataset}: missing label column")
    raw = frame[lookup["label"]]
    if raw.isna().any():
        raise ValueError(f"{dataset}: label cannot be missing")
    values = raw.astype(str).str.strip().str.lower()
    if dataset == "ton_iot":
        binary = values.map({"0": 0, "1": 1, "benign": 0, "normal": 0, "attack": 1})
        if binary.isna().any():
            raise ValueError("TON_IoT label must be binary 0/1 or benign/attack")
        return binary.to_numpy(dtype=np.int64)
    if values.eq("").any():
        raise ValueError("CICIoT2023 label cannot be empty")
    return values.eq("benigntraffic").map({True: 0, False: 1}).to_numpy(dtype=np.int64)


def _sample_csvs(files: list[Path], dataset: str, cap: int, seed: int,
                 expected_features: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample up to cap rows per class uniformly while streaming each CSV."""
    rng = np.random.default_rng(seed)
    seen = np.zeros(2, dtype=np.int64)
    reservoirs: list[list[np.ndarray]] = [[], []]
    feature_names = expected_features
    for path in files:
        for chunk in pd.read_csv(path, chunksize=10000, low_memory=False):
            labels = _labels(chunk, dataset)
            numeric = chunk.select_dtypes(include="number")
            columns = [c for c in numeric.columns if str(c).strip().lower() not in DROP_COLUMNS]
            if feature_names is None:
                feature_names = columns
            if columns != feature_names:
                raise ValueError(f"{path}: numeric feature columns differ from earlier CSV files")
            if not columns:
                raise ValueError(f"{dataset}: no numeric traffic features")
            matrix = numeric[columns].to_numpy(dtype=np.float32)
            for row, label in zip(matrix, labels):
                label = int(label)
                seen[label] += 1
                bucket = reservoirs[label]
                if len(bucket) < cap:
                    bucket.append(row)
                else:
                    index = rng.integers(seen[label])
                    if index < cap:
                        bucket[int(index)] = row
    if min(map(len, reservoirs)) < 10:
        raise ValueError(f"{dataset}: need at least 10 benign and 10 attack rows")
    X = np.stack([row for bucket in reservoirs for row in bucket])
    y = np.concatenate([np.full(len(bucket), i, dtype=np.int64) for i, bucket in enumerate(reservoirs)])
    return X, y, feature_names or []


def load_traffic_data(dataset: str, root: str | Path, *, max_rows_per_class: int = 5000,
                      batch_size: int = 128, seed: int = 42) -> TrafficData:
    """Prepare five disjoint client partitions and a held-out balanced test sample.

    CICIoT2023's provided train/test CSV split is respected when present. TON_IoT
    uses a stratified 80/20 split. Imputation and scaling fit training rows only.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Supported datasets: {DATASETS}")
    directory = Path(root) / dataset
    files = sorted(directory.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {directory}; see README")
    if max_rows_per_class < 10:
        raise ValueError("max_rows_per_class must be at least 10")
    official_train = directory / "train" / "train.csv"
    official_test = directory / "test" / "test.csv"
    if dataset == "ciciot2023" and official_train.is_file() and official_test.is_file():
        train_X, train_y, names = _sample_csvs([official_train], dataset,
                                                max_rows_per_class, seed)
        test_X, test_y, test_names = _sample_csvs([official_test], dataset,
                                                   max(100, max_rows_per_class // 5),
                                                   seed + 1, names)
        assert names == test_names
    else:
        X, y, names = _sample_csvs(files, dataset, max_rows_per_class, seed)
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed)
    train_X[~np.isfinite(train_X)] = np.nan
    test_X[~np.isfinite(test_X)] = np.nan
    median = np.nan_to_num(np.nanmedian(train_X, axis=0), nan=0.0)
    train_X = np.where(np.isfinite(train_X), train_X, median)
    test_X = np.where(np.isfinite(test_X), test_X, median)
    scaler = QuantileTransformer(n_quantiles=min(1000, len(train_X)),
                                 output_distribution="normal", random_state=seed)
    train_X = scaler.fit_transform(train_X).astype(np.float32)
    test_X = scaler.transform(test_X).astype(np.float32)
    indices = np.arange(len(train_y))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    partitions = np.array_split(indices, 5)
    clients = {i: DataLoader(TensorDataset(torch.from_numpy(train_X[part]),
                                                torch.from_numpy(train_y[part])),
                             batch_size=batch_size, shuffle=True)
               for i, part in enumerate(partitions)}
    test = DataLoader(TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y)),
                      batch_size=batch_size)
    return TrafficData(clients, test, train_X.shape[1], len(train_y), len(test_y), names)
