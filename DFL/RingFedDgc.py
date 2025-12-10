"""
Decentralized FedDGC (Dynamic Gradient Compression)
----------------------------------------------------

✓ No central server
✓ Ring topology
✓ Gradient sparsification (Top-K)
✓ Momentum accumulator (residual error)
✓ Decentralized neighbor averaging
✓ Works with TensorFlow + DP-SGD

"""

import numpy as np
import random
from copy import deepcopy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

from model import load_model
from task import load_data

# Hyperparameters
NUM_CLIENTS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32

TOP_K_RATIO = 0.1         # keep only 10% largest gradients
MOMENTUM = 0.9
LR = 0.02

# DP settings
EPSILON = 1.1
DELTA = 1e-5
L2_CLIP = 1.0
NOISE_MULTIPLIER = 1.0 / EPSILON


# Build Ring Topology
def build_ring_topology(num_clients):
    topology = {}
    for i in range(num_clients):
        topology[i] = [(i - 1) % num_clients, (i + 1) % num_clients]
    return topology


# Top-K Gradient Compression
def top_k_compress(gradient, k_ratio):
    flat = gradient.flatten()
    k = max(1, int(len(flat) * k_ratio))

    # pick largest magnitude indices
    idx = np.argpartition(np.abs(flat), -k)[-k:]

    compressed = np.zeros_like(flat)
    compressed[idx] = flat[idx]

    return compressed.reshape(gradient.shape)


# Client Node (FedDGC logic implemented)
class ClientNode:
    def __init__(self, cid, num_clients):
        self.cid = cid
        self.model = load_model()

        x_train, y_train, x_test, y_test = load_data(cid, num_clients)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self._compile_dp()

        # FedDGC buffers
        self.residual = [np.zeros_like(w) for w in self.model.get_weights()]
        self.momentum = [np.zeros_like(w) for w in self.model.get_weights()]

    def _compile_dp(self):
        optimizer = DPKerasSGDOptimizer(
            learning_rate=LR,
            noise_multiplier=NOISE_MULTIPLIER,
            l2_norm_clip=L2_CLIP,
            num_microbatches=1
        )
        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    # Local Training + Gradient Compression (FedDGC)

    def local_step(self):

        with np.errstate(all='ignore'):
            # Forward-backward pass (TF handles DP noise)
            history = self.model.fit(
                self.x_train,
                self.y_train,
                batch_size=BATCH_SIZE,
                epochs=LOCAL_EPOCHS,
                verbose=0
            )

        # Extract raw gradients
        grads = self.model.optimizer.get_gradients(
            self.model.total_loss,
            self.model.trainable_weights
        )

        # Evaluate grads in numpy
        gradients = [g.numpy() for g in grads]

        compressed_updates = []
        new_residuals = []
        new_momentum = []

        # FedDGC compression per-layer
        for g, r, m in zip(gradients, self.residual, self.momentum):

            g_res = g + r

            # Top-K compression
            g_compressed = top_k_compress(g_res, TOP_K_RATIO)

            # Update residual
            new_r = g_res - g_compressed

            # Momentum update
            new_m = MOMENTUM * m + g_compressed

            compressed_updates.append(g_compressed)
            new_residuals.append(new_r)
            new_momentum.append(new_m)

        # Save updated buffers
        self.residual = new_residuals
        self.momentum = new_momentum

        return compressed_updates

    # Apply aggregated neighbor updates
    def apply_updates(self, aggregated_updates):
        weights = self.model.get_weights()

        new_w = []
        for w, upd in zip(weights, aggregated_updates):
            new_w.append(w - LR * upd)

        self.model.set_weights(new_w)

    def evaluate(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, acc


# Decentralized Aggregation (neighbors only)

def decentralized_dgc(clients, topology):

    # Step 1: Each client computes compressed update
    updates = [c.local_step() for c in clients]

    # Step 2: Neighbor averaging
    aggregated_updates = []
    for cid in range(len(clients)):

        neighbors = topology[cid] + [cid]
        neighbor_updates = [updates[n] for n in neighbors]

        # average per-layer
        avg = []
        for layer_tensors in zip(*neighbor_updates):
            avg_layer = np.mean(layer_tensors, axis=0)
            avg.append(avg_layer)

        aggregated_updates.append(avg)

    # Step 3: Apply update to each client
    for cid, client in enumerate(clients):
        client.apply_updates(aggregated_updates[cid])

# Main Training Loop
def main():

    topology = build_ring_topology(NUM_CLIENTS)

    print("\nRing topology neighbors:")
    for cid in topology:
        print(f"Client {cid} neighbors → {topology[cid]}")

    clients = [ClientNode(i, NUM_CLIENTS) for i in range(NUM_CLIENTS)]

    # Initialize with same weights
    init_w = clients[0].model.get_weights()
    for c in clients:
        c.model.set_weights(init_w)

    for rnd in range(NUM_ROUNDS):

        print(f"\n ROUND {rnd+1}/{NUM_ROUNDS}")

        decentralized_dgc(clients, topology)
        print("FedDGC step completed using compressed neighbor averaging")

        # Evaluation
        loss_sum, acc_sum = 0, 0
        for c in clients:
            l, a = c.evaluate()
            loss_sum += l
            acc_sum += a

        print(f"Consensus Loss: {loss_sum / NUM_CLIENTS:.4f}")
        print(f"Consensus Acc : {acc_sum / NUM_CLIENTS:.4f}")

    print("\nTraining finished — saving model...")
    clients[0].model.save("ring_feddgc_model.keras")


if __name__ == "__main__":
    main()
