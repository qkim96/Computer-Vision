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


def load_mnist():
    """Load MNIST dataset, normalize it, and prepare both image and CNN-ready formats."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train_img = x_train.astype(np.float32) / 255.0
    x_test_img = x_test.astype(np.float32) / 255.0

    x_train_cnn = x_train_img[..., None]
    x_test_cnn = x_test_img[..., None]

    return (x_train_img, x_train_cnn, y_train), (x_test_img, x_test_cnn, y_test)


def make_mnist_model() -> tf.keras.Model:
    """Build a CNN model for multi-class digit classification (0–9)."""
    return tf.keras.Sequential(
        [
            Conv2D(16, (5, 5), activation="relu", padding="same", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dense(10, activation="softmax"),
        ]
    )


def plot_cnn_pred(
    x_test_img: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_acc: float,
    n_show: int = 66,
) -> None:
    """Visualize MNIST predictions and display overall test accuracy."""

    n = min(n_show, len(x_test_img))

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("CNN MNIST Predictions (True vs Pred)", y=0.99, fontsize=16)

    for i in range(n):
        plt.subplot(6, 11, i + 1)

        plt.imshow(x_test_img[i])
        plt.axis("off")

        pred = int(y_pred[i])
        true = int(y_true[i])

        color = "b" if pred == true else "r"
        plt.title(f"T:{true} P:{pred}", color=color, fontsize="large")

    fig.text(
        0.5,
        0.01,
        f"Test accuracy (full test set): {test_acc:.4f}   |   Title color: blue=correct, red=wrong",
        ha="center",
        va="bottom",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.show()


def main():
    set_seed(42)

    (x_train_img, x_train_cnn, y_train), (x_test_img, x_test_cnn, y_test) = load_mnist()

    model = make_mnist_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train_cnn,
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(x_test_cnn, y_test),
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test_cnn, y_test, verbose=0)
    print(f"[Test] loss={test_loss:.4f}, acc={test_acc:.4f}")

    probs = model.predict(x_test_cnn, batch_size=256, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    plot_cnn_pred(
        x_test_img=x_test_img,
        y_true=y_test,
        y_pred=y_pred,
        test_acc=test_acc,
        n_show=66,
    )


if __name__ == "__main__":
    main()
