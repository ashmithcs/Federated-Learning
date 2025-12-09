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
    total = len(X_train)
    size = total // num_partitions

    start = partition_id * size
    end = start + size

    X_train_p = X_train[start:end]
    y_train_p = y_train[start:end]

    return X_train_p, y_train_p, X_test, y_test

