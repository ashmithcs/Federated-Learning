"""CFL: A Flower / TensorFlow app."""

import os
import pandas as pd
import numpy as np
import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flwr.app import Context

DATASET_TYPE = "dirichlet"   # "iid" or "dirichlet"
DIRICHLET_ALPHA = 0.01


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int):

    csv_path = "/Users/ashmithc/Desktop/FL/CODE/Breast Cancer/data.csv"
    df = pd.read_csv(csv_path)

    # DROP COLUMNS (
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    # DIAGNOSIS (0/1)
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

    # SPLIT FEATURES + LABEL
    X = df.drop("diagnosis", axis=1).values
    y = df["diagnosis"].values

    # NORMALIZE FEATURES
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # TRAIN/TEST SPLIT 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # CLIENT PARTITIONING 
    if DATASET_TYPE == "iid":
        return load_data_iid(
            X_train, y_train, X_test, y_test,
            partition_id, num_partitions
        )

    elif DATASET_TYPE == "dirichlet":
        return load_data_dirichlet(
            X_train, y_train, X_test, y_test,
            partition_id, num_partitions,
            alpha=DIRICHLET_ALPHA,
        )

    else:
        raise ValueError(
            f"Invalid dataset-type '{DATASET_TYPE}'. "
            f"Use 'iid' or 'dirichlet'."
        )

def load_data_iid(
    X_train, y_train, X_test, y_test,
    partition_id: int,
    num_partitions: int,
):
    total = len(X_train)
    size = total // num_partitions

    start = partition_id * size
    end = start + size

    return (
        X_train[start:end],
        y_train[start:end],
        X_test,
        y_test,
    )


def load_data_dirichlet(
    X_train, y_train, X_test, y_test,
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    min_samples: int = 1,   
):
    rng = np.random.default_rng(42)

    num_classes = len(np.unique(y_train))
    class_indices = [np.where(y_train == c)[0].tolist() for c in range(num_classes)]

    client_indices = [[] for _ in range(num_partitions)]

    # Dirichlet allocation 
    for c in range(num_classes):
        idx = class_indices[c]
        rng.shuffle(idx)

        proportions = rng.dirichlet(alpha * np.ones(num_partitions))
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)

        splits = np.split(idx, proportions[:-1])
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    # clients with 0 samples
    for i in range(num_partitions):
        if len(client_indices[i]) < min_samples:
            # find client with most samples
            donor = np.argmax([len(ci) for ci in client_indices])

            needed = min_samples - len(client_indices[i]) #1 sample
            transfer = client_indices[donor][:needed]

            client_indices[i].extend(transfer)
            del client_indices[donor][:needed]

    client_idx = client_indices[partition_id]


    assert len(client_idx) >= min_samples, \
        f"Client {partition_id} has 0 sample"

    return (
        X_train[client_idx],
        y_train[client_idx],
        X_test,
        y_test,
    )
