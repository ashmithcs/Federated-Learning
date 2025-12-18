"""task.py
Local CSV-based dataset loader with IID and Dirichlet Non-IID partitioning
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# preprocessing
def _load_and_preprocess():
    csv_path = "/Users/ashmithc/Desktop/FL/CODE/Breast Cancer/data.csv"
    df = pd.read_csv(csv_path)

    # Drop unused columns
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    # Encode diagnosis
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

    # Split features and labels
    x = df.drop("diagnosis", axis=1).values
    y = df["diagnosis"].values

    # Normalize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Train / Test split (same test set for all clients)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return x_train, y_train, x_test, y_test

# IID PARTITIONING (SAME AS YOUR CURRENT CODE)
def load_data_iid(partition_id: int, num_partitions: int):
    x_train, y_train, x_test, y_test = _load_and_preprocess()

    total = len(x_train)
    size = total // num_partitions

    start = partition_id * size
    end = start + size

    x_train_p = x_train[start:end]
    y_train_p = y_train[start:end]

    return x_train_p, y_train_p, x_test, y_test

# NON-IID PARTITIONING (DIRICHLET)
def load_data_dirichlet(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    min_samples: int = 1,
):
    x_train, y_train, x_test, y_test = _load_and_preprocess()

    rng = np.random.default_rng(42)

    num_classes = len(np.unique(y_train))
    class_indices = [
        np.where(y_train == c)[0].tolist()
        for c in range(num_classes)
    ]

    client_indices = [[] for _ in range(num_partitions)]

    for c in range(num_classes):
        idx = class_indices[c]
        rng.shuffle(idx)

        proportions = rng.dirichlet(alpha * np.ones(num_partitions))
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)

        splits = np.split(idx, proportions[:-1])
        for i, split in enumerate(splits):
            client_indices[i].extend(split)

    for i in range(num_partitions):
        if len(client_indices[i]) < min_samples:
            donor = np.argmax([len(ci) for ci in client_indices])
            needed = min_samples - len(client_indices[i])
            transfer = client_indices[donor][:needed]
            client_indices[i].extend(transfer)
            del client_indices[donor][:needed]

    client_idx = client_indices[partition_id]

    return (
        x_train[client_idx],
        y_train[client_idx],
        x_test,
        y_test,
    )
