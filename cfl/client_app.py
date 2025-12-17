"""CFL: A Flower / TensorFlow app with Differential Privacy (DP)."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

from cfl.model import load_model
from cfl.task import load_data

# Differential Privacy PARAMETERS
EPSILON = 1.9
DELTA = 1e-5
L2_CLIP = 1.0
NOISE_MULTIPLIER = 1.0 / EPSILON


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):

    # Load model 
    model = load_model()
    
    # global weights
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(ndarrays)

    # Read config
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Load client dataset
    cid = context.node_config["partition-id"]
    num_parts = context.node_config["num-partitions"]

    x_train, y_train, _, _ = load_data(cid, num_parts)

    # Differential Privacy Optimizer
    dp_optimizer = DPKerasSGDOptimizer(
        learning_rate=0.02,
        noise_multiplier=NOISE_MULTIPLIER,
        l2_norm_clip=L2_CLIP,
        num_microbatches=1,
    )

    model.compile(
        optimizer=dp_optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )
    partition_id = int(context.node_id)
    print(f"[Client {partition_id}] Samples = {len(x_train)}")

    train_loss = history.history["loss"][-1]
    train_acc = history.history["accuracy"][-1]

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(model.get_weights()),
            "metrics": MetricRecord({
                "num-examples": len(x_train),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "epsilon": EPSILON,
                "delta": DELTA,
            }),
        }),
        reply_to=msg
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):

    # Load model
    model = load_model()
    arr = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(arr)

    cid = context.node_config["partition-id"]
    num_parts = context.node_config["num-partitions"]
    _, _, x_test, y_test = load_data(cid, num_parts)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    return Message(
        content=RecordDict({
            "metrics": MetricRecord({
                "eval_loss": loss,
                "eval_acc": acc,
                "num-examples": len(x_test),
            })
        }),
        reply_to=msg
    )
