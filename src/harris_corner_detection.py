import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.data


def harris(im: np.ndarray, k: int = 3, alpha: float = 0.05) -> np.ndarray:
    """Compute Harris corner response map."""
    dx = cv2.Sobel(im, -1, dx=1, dy=0)
    dy = cv2.Sobel(im, -1, dx=0, dy=1)

    dxx = cv2.GaussianBlur(dx ** 2, (k, k), sigmaX=-1)
    dyy = cv2.GaussianBlur(dy ** 2, (k, k), sigmaX=-1)
    dxy = cv2.GaussianBlur(dx * dy, (k, k), sigmaX=-1)

    return dxx * dyy - dxy ** 2 - alpha * (dxx + dyy) ** 2


def find_local_maxima(im: np.ndarray, threshold: float = 50) -> np.ndarray:
    """Return coordinates of local maxima above threshold."""
    points = np.argwhere(im > threshold)
    points = [(x, y) for y, x in points]

    maxima = []
    for p in points:
        if p[0] == 0 or p[0] == im.shape[1] - 1 or p[1] == 0 or p[1] == im.shape[0] - 1:
            continue

        neighbors = im[p[1] - 1 : p[1] + 2, p[0] - 1 : p[0] + 2]
        if np.all(neighbors <= im[p[1], p[0]]):
            maxima.append(p)

    return np.array(maxima)


def main():
    im = skimage.data.camera()
    har = harris(np.float32(im), k=11, alpha=0.05)
    harris_points = find_local_maxima(har, threshold=2e9)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.02},
    )

    axes[0].imshow(im, cmap="gray")
    hm = axes[0].imshow(har, cmap="jet", alpha=0.75)
    fig.colorbar(hm, ax=axes[0], fraction=0.046, pad=0.02)

    axes[1].imshow(im, cmap="gray")
    axes[1].scatter(harris_points[:, 0], harris_points[:, 1], c="r", s=10)

    plt.show()


if __name__ == "__main__":
    main()
