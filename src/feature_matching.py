import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests


IM1_URL = "https://drive.google.com/uc?id=1y8eKxsxxQDDxnwZex_qNi_1QtEmr7xai"
IM2_URL = "https://drive.google.com/uc?id=1ZRNAyo9SUeL0BcTJKFzKuEku2-YTkvA9"


def load_gray(url: str) -> np.ndarray:
    """Download image from URL and load it as a grayscale array."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f"Failed to decode image from: {url}")
    return im


def warp_and_blend_images_homography(image1: np.ndarray, image2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp image2 onto image1 using homography H and blend overlapping regions."""
    h_, w_ = image1.shape
    im2warp = cv2.warpPerspective(
        image2,
        H,
        (w_, h_),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    im2mask = cv2.warpPerspective(
        np.ones_like(image2),
        H,
        (w_, h_),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return image1 * (1.0 - im2mask) + im2warp * im2mask


def sift_keypoints(im: np.ndarray):
    """Detect SIFT keypoints and descriptors in a grayscale image."""
    sift = cv2.SIFT_create()
    kpts, desc = sift.detectAndCompute(im, None)
    if desc is None or len(kpts) == 0:
        raise RuntimeError("No SIFT features detected.")
    return kpts, desc


def ratio_test_knn(bf: cv2.BFMatcher, desc1, desc2, ratio=0.75):
    """Apply ratio test to KNN descriptor matches."""
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])
    return good


def reciprocity_filter(good12, desc2, desc1, bf: cv2.BFMatcher, ratio=0.75):
    """Keep only reciprocal (mutual best) matches after ratio test."""
    matches21 = bf.knnMatch(desc2, desc1, k=2)

    recip_pairs = set()
    for m, n in matches21:
        if m.distance < ratio * n.distance:
            recip_pairs.add((m.queryIdx, m.trainIdx))

    good_recip = []
    for g in good12:
        m = g[0]
        if (m.trainIdx, m.queryIdx) in recip_pairs:
            good_recip.append(g)

    return good_recip


def matches_to_pts(kpts1, kpts2, good_recip):
    """Convert matched key points into corresponding point coordinate arrays."""
    pts1 = np.array([[0, 0]] * len(good_recip), dtype=np.float32)
    pts2 = np.array([[0, 0]] * len(good_recip), dtype=np.float32)
    for i, match in enumerate(good_recip):
        m = match[0]
        pts1[i] = kpts1[m.queryIdx].pt
        pts2[i] = kpts2[m.trainIdx].pt
    return pts1, pts2


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB format if it has 3 channels."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    # Original images
    im1 = load_gray(IM1_URL)
    im2 = load_gray(IM2_URL)

    # SIFT + matching (ratio + reciprocity)
    kpts1, desc1 = sift_keypoints(im1)
    kpts2, desc2 = sift_keypoints(im2)

    bf = cv2.BFMatcher_create()
    good12 = ratio_test_knn(bf, desc1, desc2, ratio=0.75)
    good_recip = reciprocity_filter(good12, desc2, desc1, bf, ratio=0.75)

    if len(good_recip) < 8:
        raise RuntimeError(f"Not enough reciprocal matches: {len(good_recip)}")

    # Feature matching lines plot
    match_vis = cv2.drawMatchesKnn(
        im1,
        kpts1,
        im2,
        kpts2,
        good_recip,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    match_vis = bgr_to_rgb(match_vis)

    # Homography + inliers
    im1wide = np.hstack([im1, np.zeros_like(im1)])

    pts1, pts2 = matches_to_pts(kpts1, kpts2, good_recip)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    if H is None or mask is None:
        raise RuntimeError("findHomography failed.")

    H_in = cv2.findHomography(pts1[mask[:, 0] == 1], pts2[mask[:, 0] == 1], cv2.RANSAC)[0]
    if H_in is None:
        raise RuntimeError("findHomography on inliers failed.")

    # Final warp
    lstsqWarp = warp_and_blend_images_homography(im1wide, im2, H_in)

    # Final plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    axes[0, 0].imshow(im1, cmap="gray")
    axes[0, 0].set_title("Original Image 1")
    axes[0, 0].set_axis_on()

    axes[0, 1].imshow(im2, cmap="gray")
    axes[0, 1].set_title("Original Image 2")
    axes[0, 1].set_axis_on()

    axes[1, 0].imshow(match_vis)
    axes[1, 0].set_title("Feature Matching (ratio + reciprocity)")
    axes[1, 0].set_axis_off()

    axes[1, 1].imshow(lstsqWarp, cmap="gray")
    axes[1, 1].set_title("lstsq (warp + blend)")
    axes[1, 1].set_axis_off()

    plt.show()


if __name__ == "__main__":
    main()
