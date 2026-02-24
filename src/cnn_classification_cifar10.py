import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_cifar():
    """Load CIFAR-10 and filter it to a binary cat (0) vs dog (1) classification task."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # CIFAR-10 labels: cat=3, dog=5
    train_mask = np.squeeze((y_train == 3) | (y_train == 5))
    test_mask = np.squeeze((y_test == 3) | (y_test == 5))

    x_train = x_train[train_mask].astype(np.float32) / 255.0
    x_test = x_test[test_mask].astype(np.float32) / 255.0

    # binary: dog(5)->1, cat(3)->0
    y_train = (y_train[train_mask] == 5).astype(np.int32).squeeze()
    y_test = (y_test[test_mask] == 5).astype(np.int32).squeeze()

    return (x_train, y_train), (x_test, y_test)


def make_cifar_model() -> tf.keras.Model:
    """Build a simple CNN model for binary image classification (cat vs dog)."""
    return tf.keras.Sequential(
        [
            Conv2D(16, (5, 5), activation="relu", padding="same", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            BatchNormalization(),

            Conv2D(20, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            BatchNormalization(),

            Conv2D(20, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            BatchNormalization(),

            GlobalAveragePooling2D(),
            Dense(1, activation="sigmoid"),
        ]
    )


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute TN, FP, FN, TP counts for binary classification."""
    # 0=cat, 1=dog
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def plot_cnn_pred(
    images: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    title: str,
    n_show: int = 24,
    cols: int = 6,
) -> None:
    """Visualize predictions with probabilities and display confusion statistics."""

    n = min(n_show, images.shape[0])
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(cols * 2.2, rows * 2.2 + 1.2))
    fig.suptitle(title, y=0.98, fontsize=14)

    for i in range(n):
        prob = float(y_prob[i])
        pred = 1 if prob >= 0.5 else 0
        t = int(y_true[i])

        true_str = "dog" if t == 1 else "cat"
        pred_str = "dog" if pred == 1 else "cat"

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(images[i])
        ax.axis("off")

        color = "red" if pred != t else "black"
        ax.set_title(f"T:{true_str}\nP:{pred_str} ({prob:.2f})", fontsize=9, color=color)

    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0.0
    conf_text = (
        "Confusion (cat=0, dog=1)\n"
        f"TN(cat→cat): {tn}   FP(cat→dog): {fp}   FN(dog→cat): {fn}   TP(dog→dog): {tp}\n"
        f"Accuracy: {acc:.4f}"
    )

    fig.text(0.5, 0.02, conf_text, ha="center", va="bottom", fontsize=11)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()


def main():
    set_seed(42)

    (x_train, y_train), (x_test, y_test) = load_cifar()
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test : {x_test.shape}, y_test : {y_test.shape}")

    model = make_cifar_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    probs = model.predict(x_test, batch_size=256, verbose=0).reshape(-1)  # dog 확률
    preds = (probs >= 0.5).astype(np.int32)

    tn, fp, fn, tp = confusion_counts(y_test, preds)

    n_random = 24
    idx = np.random.choice(len(x_test), size=min(n_random, len(x_test)), replace=False)

    plot_cnn_pred(
        images=x_test[idx],
        y_true=y_test[idx],
        y_prob=probs[idx],
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        title="CNN CIFAR-10 Predictions (True vs Pred)",
        n_show=len(idx),
        cols=6,
    )


if __name__ == "__main__":
    main()
