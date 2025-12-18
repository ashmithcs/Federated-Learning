"""
TRUE Decentralized Federated Learning (DFL)
-------------------------------------------

✓ No server
✓ Every client trains locally
✓ Neighbor-based FedAvg (Decentralized FedAvg)
✓ Models converge via local consensus
✓ DP-SGD supported
"""

import random
import numpy as np
from copy import deepcopy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

from model import load_model
from task import load_data_iid, load_data_dirichlet
import tensorflow as tf 
import tomli
#rb = read binary
with open("pyproject.toml", "rb") as py:
    config = tomli.load(py)

config = config["tool"]["flwr"]["app"]["config"]

NUM_ROUNDS = config["num-server-rounds"]
LOCAL_EPOCHS = config["local-epochs"]
BATCH_SIZE = config["batch-size"]
NUM_CLIENTS = config["num-clients"]

# Dataset control
DATASET_TYPE = config.get("dataset-type")      # "iid" or "dirichlet"
DIRICHLET_ALPHA = config.get("dirichlet-alpha")

# Differential Privacy
EPSILON = config["epsilon"]
DELTA = config["delta"]
L2_CLIP = config["l2_clip"]
NOISE_MULTIPLIER = 1.0 / EPSILON

# FedProx
MU = float(config.get("mu", 0.0))

def load_selected_data(cid, num_clients):

    if DATASET_TYPE == "iid":
        return load_data_iid(cid, num_clients)

    elif DATASET_TYPE == "dirichlet":
        return load_data_dirichlet(
            cid,
            num_clients,
            alpha=DIRICHLET_ALPHA,
            min_samples=1,
        )

    else:
        raise ValueError(
            f"Invalid dataset-type '{DATASET_TYPE}'. "
            f"Use 'iid' or 'dirichlet'."
        )
# BUILD RING TOPOLOGY (Each client has 2 neighbors)
def build_ring_topology(num_clients):
    topology = {}
    for i in range(num_clients):
        topology[i] = [(i - 1) % num_clients, (i + 1) % num_clients]
    return topology

# DECENTRALIZED FEDAVG
def decentralized_fedavg(clients, topology):
    """Perform neighbor-based FedAvg (no server)."""
    new_weights = []

    # Collect all client model weights
    all_weights = [c.get_weights() for c in clients]

    for cid, c in enumerate(clients):
        # Neighbor set including self
        neighbors = topology[cid] + [cid]

        # Collect corresponding neighbor models
        neighbor_models = [all_weights[n] for n in neighbors]

        # Layer-wise average
        avg_weights = []
        for layer_tensors in zip(*neighbor_models):
            avg_layer = np.mean(layer_tensors, axis=0)
            avg_weights.append(avg_layer)

        new_weights.append(avg_weights)

    # Update each client's weights
    for cid, c in enumerate(clients):
        c.set_weights(new_weights[cid])


# CLIENT CLASS
class ClientNode:
    def __init__(self, cid, num_clients):
        self.cid = cid
        self.model = load_model()

        x_train, y_train, x_test, y_test = load_selected_data(cid, num_clients)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.ref_weights_tf = None
        self._compile_dp()

    def set_ref_weights(self, ref_weights_np):
        self.ref_weights_tf = [
            tf.convert_to_tensor(w, dtype=tw.dtype)
            for w, tw in zip(ref_weights_np, self.model.trainable_weights)
        ]

    def _compile_dp(self):
        dp_optimizer = DPKerasSGDOptimizer(
            learning_rate=0.02,
            noise_multiplier=NOISE_MULTIPLIER,
            l2_norm_clip=L2_CLIP,
            num_microbatches=1
        )

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")

        def fedprox_loss(y_true, y_pred):
            base = bce(y_true, y_pred)  # shape (batch,)

            if MU <= 0.0 or self.ref_weights_tf is None:
                return base

            prox = tf.add_n([
                tf.reduce_sum(tf.square(w - w_ref))
                for w, w_ref in zip(self.model.trainable_weights, self.ref_weights_tf)
            ])

            return base + (MU / 2.0) * prox


        self.model.compile(
            optimizer=dp_optimizer,
            loss=fedprox_loss,
            metrics=["accuracy"]
        )

    def local_train(self):
        if len(self.x_train) == 0:
         print(f"Client {self.cid} has no data — skipping training")
         return 0.0, 0.0


        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=BATCH_SIZE,
            epochs=LOCAL_EPOCHS,
            verbose=0
        )
        return history.history["loss"][-1], history.history["accuracy"][-1]

    def get_weights(self):
        return deepcopy(self.model.get_weights())

    def set_weights(self, weights):
        self.model.set_weights(deepcopy(weights))

    def evaluate(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, acc

# MAIN DECENTRALIZED FEDAVG PROCESS
def main():
    print(f"Clients        : {NUM_CLIENTS}")
    print(f"Rounds         : {NUM_ROUNDS}")
    print(f"Dataset type   : {DATASET_TYPE}")
    print(f"Epsilon (DP)   : {EPSILON}")
    print(f"DELTA (DP)   : {DELTA}")
    print(f"L2_CLIP (DP)   : {L2_CLIP}")
    print(f"NOISE_MULTIPLIER (DP)   : {NOISE_MULTIPLIER}")
    if DATASET_TYPE == "dirichlet":
        print(f"Dirichlet  : {DIRICHLET_ALPHA}")
    # Initialize topology
    topology = build_ring_topology(NUM_CLIENTS)

    print("\nRing topology neighbors:")
    for cid in topology:
     print(f"Client {cid} neighbors → {topology[cid]}")

    # Initialize clients
    clients = [ClientNode(i, NUM_CLIENTS) for i in range(NUM_CLIENTS)]

    # Start from identical initialization
    initial_weights = clients[0].get_weights()
    for c in clients:
        c.set_weights(initial_weights)

    for rnd in range(NUM_ROUNDS):
        print(f"\n ROUND {rnd + 1}/{NUM_ROUNDS} ")

        # Step 1: Local Training
        for c in clients:
            loss, acc = c.local_train()
            print(f"Client {c.cid} → loss={loss:.4f}  acc={acc:.4f}")

        # Step 2: Decentralized FedAvg
        decentralized_fedavg(clients, topology)
        print("Performed decentralized FedAvg among neighbors")

        # Step 3: Evaluate global consensus
        total_loss, total_acc = 0, 0
        for c in clients:
            loss, acc = c.evaluate()
            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / NUM_CLIENTS
        avg_acc = total_acc / NUM_CLIENTS

        print(f"Global Loss: {avg_loss:.4f}")
        print(f"Global Acc : {avg_acc:.4f}")

    print("\nTraining completed — saving model")
    clients[0].model.save("ring_decentralized_model.keras")

# RUN
if __name__ == "__main__":
    main()
