"""
TRUE Decentralized Federated Learning (DFL)
-------------------------------------------

✓ No server
✓ Every client trains locally
✓ Every client gossips with a random peer
✓ Models gradually converge without a central aggregator
✓ DP-SGD supported
"""

import random
import numpy as np
from copy import deepcopy

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

from model import load_model
from task import load_data

# Hyperparameters
NUM_CLIENTS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 32

ALPHA = 0.5   # Gossip mixing factor (0.5 = simple averaging)

# Differential Privacy
EPSILON = 1.1
DELTA = 1e-5
L2_CLIP = 1.0
NOISE_MULTIPLIER = 1.0 / EPSILON

# Client Node
class ClientNode:
    def __init__(self, cid, num_clients):
        self.cid = cid
        self.model = load_model()

        # Load private data
        x_train, y_train, x_test, y_test = load_data(cid, num_clients)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self._compile_dp()

    def _compile_dp(self):
        dp_optimizer = DPKerasSGDOptimizer(
            learning_rate=0.02,
            noise_multiplier=NOISE_MULTIPLIER,
            l2_norm_clip=L2_CLIP,
            num_microbatches=1
        )
        self.model.compile(
            optimizer=dp_optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def local_train(self):
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

    def gossip_with(self, peer_model):
        """Perform decentralized gossip averaging"""
        my_weights = self.get_weights()
        new_weights = []

        for w_self, w_peer in zip(my_weights, peer_model):
            new_w = ALPHA * w_self + (1 - ALPHA) * w_peer
            new_weights.append(new_w)

        self.set_weights(new_weights)

    def evaluate(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, acc

# Decentralized FL Process
def main():

    # Initialize nodes
    clients = [ClientNode(i, NUM_CLIENTS) for i in range(NUM_CLIENTS)]

    # Start with identical initialization
    initial_weights = clients[0].get_weights()
    for c in clients:
        c.set_weights(initial_weights)

    for rnd in range(NUM_ROUNDS):
        print(f"\n ROUND {rnd + 1}/{NUM_ROUNDS} ")

        # 1) Local training for ALL clients
        for c in clients:
            loss, acc = c.local_train()
            print(f"Client {c.cid} → loss={loss:.4f}  acc={acc:.4f}")

        # 2) Peer-to-peer model gossip (no server)
        for c in clients:
            peer = random.choice(clients)
            if peer.cid != c.cid:
                c.gossip_with(peer.get_weights())
                print(f"Client {c.cid} gossiped with Client {peer.cid}")

        # 3) Evaluate global consensus
        total_loss, total_acc, total_samples = 0, 0, 0
        for c in clients:
            loss, acc = c.evaluate()
            total_loss += loss
            total_acc += acc
            total_samples += 1

        print(f"Global Consensus Loss: {total_loss / total_samples:.4f}")
        print(f"Global Consensus Acc : {total_acc / total_samples:.4f}")

    print("\nTraining completed — saving model")
    clients[0].model.save("gossip_decentralized_model.keras")


if __name__ == "__main__":
    main()
