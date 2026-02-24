import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
import skimage.io
import skimage.color
from io import BytesIO


def arc_length(curve: np.ndarray) -> np.ndarray:
    """Compute normalized cumulative arc-length of a curve."""
    diff_sq_x_and_y = np.diff(curve, axis=0) ** 2
    diff_root = np.sum(np.sqrt(diff_sq_x_and_y), axis=1)
    dist_ = np.cumsum(diff_root)
    dist_ = np.insert(dist_, 0, 0)
    dist_ = dist_ / dist_[-1]
    return dist_


def resample_curve_uniform(curve: np.ndarray, num_points: int | None = None) -> np.ndarray:
    """Resample curve points uniformly along its arc-length."""
    len_crv = curve.shape[0] if num_points is None else num_points
    dist_ = arc_length(curve)
    dist_resampled = np.linspace(dist_[0], dist_[-1], num=len_crv)
    x_resampled = np.interp(dist_resampled, dist_, curve[:, 0])
    y_resampled = np.interp(dist_resampled, dist_, curve[:, 1])
    return np.hstack((np.array([x_resampled]).T, np.array([y_resampled]).T))


def calc_window_energy_function(
    point: np.ndarray,
    next_point: np.ndarray,
    prev_point: np.ndarray,
    gradient_image_: np.ndarray,
    window_size: int = 20,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> np.ndarray:
    """Compute local snake energy (image + elasticity) in a window around a point."""

    win_half = window_size // 2

    patch = gradient_image_[
        int(point[1] - win_half) : int(point[1] + win_half + 1),
        int(point[0] - win_half) : int(point[0] + win_half + 1),
    ]

    patch = cv2.normalize(patch, None, 0.0, 1.0, cv2.NORM_MINMAX)

    elasticity_term_energy = np.zeros_like(patch)
    for y in range(patch.shape[0]):
        for x in range(patch.shape[1]):
            window_point_global_coordinates = (
                np.array([x, y]) - win_half + np.array([point[0], point[1]])
            )
            elasticity_term_energy[y, x] = (
                alpha
                * np.linalg.norm(np.array([next_point[0], next_point[1]]) - window_point_global_coordinates)
                + beta
                * np.linalg.norm(
                    np.array([next_point[0], next_point[1]])
                    - 2 * window_point_global_coordinates
                    + np.array([prev_point[0], prev_point[1]])
                )
            )

    elasticity_term_energy = cv2.normalize(elasticity_term_energy, None, 0, 1, cv2.NORM_MINMAX)
    patch += -elasticity_term_energy
    return patch


def find_max_in_patch(patch: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Find global coordinates of maximum energy within a patch."""
    max_value_in_patch = np.max(patch)
    max_gradient_magnitude_index = np.argwhere(patch == max_value_in_patch)[0]
    max_idx = (
        np.array(max_gradient_magnitude_index) - patch.shape[0] // 2 + np.array([point[0], point[1]])
    )
    return max_idx


def show_energy_function_patches(
    curve_: np.ndarray,
    mag_: np.ndarray,
    window_size: int = 20,
    alpha: float = 0.05,
    beta: float = 0.05,
):
    """Visualize energy patches and movement directions for each curve point."""

    plt.figure(figsize=(15, 15))
    for i in range(curve_.shape[0]):
        patch = calc_window_energy_function(
            curve_[i],
            curve_[(i + 1) % curve_.shape[0]],
            curve_[(i - 1)],
            mag_,
            window_size=window_size,
            alpha=alpha,
            beta=beta,
        )
        plt.subplot(5, 5, i + 1)
        plt.imshow(patch, cmap="gray")
        plt.axis("off")

        w2 = patch.shape[0] // 2

        max_idx = find_max_in_patch(patch, curve_[i]) + w2 - curve_[i]
        max_idx = np.array([max_idx[1], max_idx[0]])

        plt.arrow(
            w2, w2, max_idx[0] - w2, max_idx[1] - w2,
            color="b", head_width=1, head_length=1
        )
        plt.scatter(max_idx[0], max_idx[1], c="r", s=10)
        plt.title("Point %d (%.2f)" % (i, np.linalg.norm(max_idx - w2)))

    plt.tight_layout()
    plt.show()


def update_curve(
    curve: np.ndarray,
    gradient_image_: np.ndarray,
    window_size: int = 20,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> np.ndarray:
    """Update curve points by moving them toward local energy maxima."""

    new_curve = np.zeros_like(curve)
    for i in range(curve.shape[0]):
        patch = calc_window_energy_function(
            curve[i],
            curve[(i + 1) % curve.shape[0]],
            curve[(i - 1)],
            gradient_image_,
            window_size=window_size,
            alpha=alpha,
            beta=beta,
        )
        max_idx = find_max_in_patch(patch, curve[i])

        direction = max_idx - curve[i]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            new_curve[i] = [
                curve[i][0] + 2 * direction[1] / direction_norm,
                curve[i][1] + 2 * direction[0] / direction_norm,
            ]
        else:
            new_curve[i] = curve[i]

    return new_curve


def load_img_and_mag():
    """Load image, convert to grayscale, resize, and compute gradient magnitude."""
    # Load
    url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Gull_portrait_ca_usa.jpg"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    im1_rgb = skimage.io.imread(BytesIO(response.content))

    # Grayscale
    im1 = skimage.color.rgb2gray(im1_rgb)
    im1 = cv2.resize(im1, (0, 0), fx=0.25, fy=0.25)

    # Gradient magnitude
    dx = cv2.Sobel(im1, cv2.CV_64F, dx=1, dy=0, ksize=25)
    dy = cv2.Sobel(im1, cv2.CV_64F, dx=0, dy=1, ksize=25)
    mag = cv2.magnitude(dx, dy)

    return im1, mag


def main():
    im1, mag = load_img_and_mag()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(im1, cmap="gray")
    axes[0].set_title("Grayscale (resized 0.25x)")
    axes[0].set_axis_on()

    axes[1].imshow(mag, cmap="gray")
    axes[1].set_title("Gradient magnitude |∇I| (Sobel ksize=25)")
    axes[1].set_axis_on()
    plt.show()

    curve = np.array([[305, 80], [365, 80], [365, 130], [305, 130], [305, 80]], np.float32)
    curve_resampled = resample_curve_uniform(curve, 25)

    plt.figure(figsize=(5, 5))
    plt.imshow(mag, cmap="gray")
    plt.plot(curve_resampled[:, 0], curve_resampled[:, 1], "r-")
    plt.scatter(curve_resampled[:, 0], curve_resampled[:, 1], c="r", s=5)
    plt.title("Initial curve (25 points)")
    plt.xlim(curve_resampled[:, 0].min() - 20, curve_resampled[:, 0].max() + 20)
    plt.ylim(curve_resampled[:, 1].max() + 20, curve_resampled[:, 1].min() - 20)
    plt.show()

    show_energy_function_patches(curve_resampled, mag, window_size=20, alpha=0.01, beta=0.08)

    new_curve = curve_resampled
    for _ in range(30):
        new_curve[-1] = new_curve[0]  # keep closed
        new_curve = resample_curve_uniform(new_curve)
        new_curve = update_curve(new_curve, mag, window_size=20, alpha=0.01, beta=0.00)

    plt.figure(figsize=(5, 5))
    plt.imshow(mag, cmap="gray")

    plt.plot(new_curve[:, 0], new_curve[:, 1], c="r")
    plt.scatter(new_curve[:, 0], new_curve[:, 1], c="r", s=5)
    for i in range(new_curve.shape[0]):
        plt.text(new_curve[i, 0], new_curve[i, 1], str(i), color="r")

    plt.plot(curve_resampled[:, 0], curve_resampled[:, 1], c="g")
    plt.scatter(curve_resampled[:, 0], curve_resampled[:, 1], c="g", s=5)

    plt.title("Last Iteration")
    plt.xlim(new_curve[:, 0].min() - 20, new_curve[:, 0].max() + 20)
    plt.ylim(new_curve[:, 1].max() + 20, new_curve[:, 1].min() - 20)
    plt.show()


if __name__ == "__main__":
    main()
