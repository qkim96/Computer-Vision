import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io


img_urls = (
    "https://drive.google.com/uc?export=download&id=1DjiJ60PPxYkM9uzy7OFqxhwyDSqxyv06, "
    "https://drive.google.com/uc?export=download&id=1oiVYZX5F55-ZHfCeUqu5UTwFvh2viXK8, "
    "https://drive.google.com/uc?export=download&id=1gmxmxZ4E6Sq4OrhKgMDYhKaOwvtvcIB6, "
    "https://drive.google.com/uc?export=download&id=1c6eN7YOcf0GzK3zeV0U_f9ih_7_WhtTO"
).split(", ")

imgs = [
    cv2.resize(io.imread(url), (0, 0), fx=0.25, fy=0.25)
    for url in img_urls
]

fig, ax = plt.subplots(1, 4, figsize=(20, 6))
for i, img in enumerate(imgs):
    ax[i].imshow(img)
    ax[i].axis("off")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
plt.show()

detector = cv2.SIFT_create()
matcher = cv2.BFMatcher_create(cv2.NORM_L2, True)


def get_matched_pts(i, j, kpts_, desc_, matcher_):
    """Return matched keypoint coordinates and match objects between two images."""
    matches = matcher_.match(desc_[i], desc_[j])
    mptsi, mptsj = [], []

    for m in matches:
        mptsi.append(kpts_[i][m.queryIdx].pt)
        mptsj.append(kpts_[j][m.trainIdx].pt)

    return np.array(mptsi), np.array(mptsj), matches


def get_transforms_and_inliers(
    kpts_, desc_, matcher_, geometric_model_function=cv2.findHomography
):
    """Compute geometric transforms and inlier masks using RANSAC."""

    transforms_and_inliers = [
        [(), (), (), ()],
        [(), (), (), ()],
        [(), (), (), ()],
    ]

    for i in range(3):
        for j in range(i + 1, 4):
            mptsL, mptsR, matches = get_matched_pts(
                i, j, kpts_, desc_, matcher_
            )

            M, inliers = geometric_model_function(
                np.array(mptsL),
                np.array(mptsR),
                cv2.RANSAC,
            )

            transforms_and_inliers[i][j] = (M, inliers, matches)

    return transforms_and_inliers


def get_transforms_and_inliers_for_img(
    images_,
    geometric_model_function=cv2.findHomography,
    masks=None,
):
    """Detect features and compute transforms with inliers for a list of images."""

    if masks is None:
        masks = [None] * len(images_)

    kpts_, desc_ = zip(
        *[
            detector.detectAndCompute(image, mask)
            for image, mask in zip(images_, masks)
        ]
    )

    return (
        get_transforms_and_inliers(
            kpts_, desc_, matcher, geometric_model_function
        ),
        kpts_,
        desc_,
    )


transforms_and_inliers, kpts, desc = get_transforms_and_inliers_for_img(
    imgs,
    cv2.findHomography,
)


for i in range(3):
    fig, axs = plt.subplots(1, 3 - i, figsize=(20, 6))

    for j in range(i + 1, 4):
        ax_ = axs[j - i - 1] if i < 2 else axs

        matches = np.array(transforms_and_inliers[i][j][2])
        inliers_mask = transforms_and_inliers[i][j][1].squeeze()

        inlier_matches = [
            matches[ii]
            for ii in range(len(inliers_mask))
            if (inliers_mask[ii] == 1)
        ]

        ax_.imshow(
            cv2.drawMatches(
                imgs[i],
                kpts[i],
                imgs[j],
                kpts[j],
                inlier_matches,
                None,
                flags=2,
            )
        )

        total_matches = len(transforms_and_inliers[i][j][2])

        ax_.text(
            0,
            imgs[i].shape[0],
            "Inliers: {}/{} ({:.2f}%)".format(
                len(inlier_matches),
                total_matches,
                len(inlier_matches) / total_matches * 100,
            ),
            color="white",
            fontsize=18,
        )

        ax_.set_title(
            "Image {} -> Image {}".format(i + 1, j + 1),
            # y=1,
            # pad=2
        )
        ax_.axis("off")

    fig.subplots_adjust(
        left=0,
        right=1,
        top=0.92,
        bottom=0.08,
        wspace=0.02,
        hspace=0.02,
    )
    plt.show()


# Spherical warp
w, h2 = imgs[0].shape[1], imgs[0].shape[0] / 2
K = np.array(
    [[1200, 0, w / 2], [0, 1200, h2], [0, 0, 1]],
    dtype=np.float64,
)


def spherical_warp(img, K):
    """Warp an image onto a spherical surface using intrinsic matrix K."""

    h_, w_ = img.shape[:2]

    y_i, x_i = np.indices((h_, w_))
    X = np.stack(
        (x_i, y_i, np.ones_like(x_i)),
        axis=-1,
    ).reshape(h_ * w_, 3)

    Kinv = np.linalg.inv(K)
    X = (Kinv @ X.T).T

    A = np.stack(
        [
            np.sin(X[:, 0]) * np.cos(X[:, 1]),
            np.sin(X[:, 1]),
            np.cos(X[:, 0]) * np.cos(X[:, 1]),
        ],
        axis=-1,
    ).reshape(w_ * h_, 3)

    B = (K @ A.T).T
    B = B[:, :-1] / B[:, [-1]]

    B[
        (B[:, 0] < 0)
        | (B[:, 0] >= w_)
        | (B[:, 1] < 0)
        | (B[:, 1] >= h_)
    ] = -1

    B = B.reshape(h_, w_, -1)

    mask = (B[:, :, 0] != -1).astype(np.float32)

    img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    warped = cv2.remap(
        img_rgba,
        B[:, :, 0].astype(np.float32),
        B[:, :, 1].astype(np.float32),
        cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return warped, mask


imgs_sph_with_masks = [
    spherical_warp(im, K)
    for im in imgs
]

imgs_sph_rgba = [
    imwarp
    for imwarp, m in imgs_sph_with_masks
]

masks_sph_float = [
    m
    for imwarp, m in imgs_sph_with_masks
]


fig = plt.figure(figsize=(20, 8))

for i, img in enumerate(imgs):
    ax1 = plt.subplot(2, 4, i + 1)
    ax1.imshow(img)
    ax1.axis("off")

    ax2 = plt.subplot(2, 4, 4 + i + 1)
    ax2.imshow(imgs_sph_rgba[i])
    ax2.axis("off")

fig.subplots_adjust(
    left=0,
    right=1,
    top=1,
    bottom=0,
    wspace=0.02,
    hspace=0.02,
)
plt.show()


# Final Panorama
def create_distance_transform_mask(im_rgb):
    """Create a normalized distance transform mask for smooth image blending."""
    mask = np.ones(im_rgb.shape[:2], dtype=np.uint8)

    cv2.circle(
        mask,
        (im_rgb.shape[1] // 2, im_rgb.shape[0] // 2),
        3,
        0,
        -1,
    )

    dt = cv2.distanceTransform(
        mask,
        distanceType=cv2.DIST_C,
        maskSize=5,
    )

    dt = 1 - cv2.normalize(
        dt,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    return dt.astype(np.float32)


def affine_to_homography(A):
    """Convert a 2x3 affine transformation matrix into a 3x3 homography matrix."""
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = A
    return H


def stitch_images_simple(
    imgs_H_masks,
    canvas_w_factor=3,
    blend_method=1,
):
    """Warp images to a common canvas and blend them using weighted averaging."""

    center_img = imgs_H_masks[0][0]
    h, w = center_img.shape[:2]

    canvas_w = w * canvas_w_factor
    canvas_h = h

    T = np.array(
        [[1, 0, w], [0, 1, 0], [0, 0, 1]],
        dtype=np.float64,
    )

    acc = np.zeros(
        (canvas_h, canvas_w, 3),
        dtype=np.float32,
    )

    wsum = np.zeros(
        (canvas_h, canvas_w),
        dtype=np.float32,
    )

    for img, H_to_center, m in imgs_H_masks:
        if H_to_center is None:
            H_to_center = np.eye(3, dtype=np.float64)

        H_canvas = T @ H_to_center

        if blend_method == 0:
            weight = np.ones(
                img.shape[:2],
                dtype=np.float32,
            )
        else:
            weight = create_distance_transform_mask(img)

        if m is not None:
            weight = weight * m.astype(np.float32)

        warped_img = cv2.warpPerspective(
            img,
            H_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)

        warped_w = cv2.warpPerspective(
            weight,
            H_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)

        warped_w = np.nan_to_num(
            warped_w,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        warped_img = np.nan_to_num(
            warped_img,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        acc += warped_img * warped_w[..., None]
        wsum += warped_w

    out = np.zeros_like(
        acc,
        dtype=np.uint8,
    )

    valid = wsum > 1e-6

    out[valid] = (
        acc[valid] / wsum[valid, None]
    ).clip(0, 255).astype(np.uint8)

    return out


def crop_black_borders(
    img_rgb: np.ndarray,
    thresh: int = 5,
) -> np.ndarray:
    """Crop black border regions from the stitched panorama image."""

    mask = (
        (img_rgb[:, :, 0] > thresh)
        | (img_rgb[:, :, 1] > thresh)
        | (img_rgb[:, :, 2] > thresh)
    )

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return img_rgb

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    return img_rgb[y0:y1, x0:x1]


imgs_sph_rgb = [
    cv2.cvtColor(im_rgba, cv2.COLOR_RGBA2RGB)
    for im_rgba in imgs_sph_rgba
]

imgs_sph_gray = [
    cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    for im in imgs_sph_rgb
]

imgs_sph_gray = [
    cv2.normalize(
        im,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    for im in imgs_sph_gray
]

masks_sph_u8 = [
    (m * 255).astype(np.uint8)
    for m in masks_sph_float
]


sift = cv2.SIFT_create()
bf = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=True,
)


def estimate_affine(i, j):
    """Estimate affine transformation between two spherical-warped images using SIFT and RANSAC."""
    k1, d1 = sift.detectAndCompute(
        imgs_sph_gray[i],
        masks_sph_u8[i],
    )

    k2, d2 = sift.detectAndCompute(
        imgs_sph_gray[j],
        masks_sph_u8[j],
    )

    if d1 is None or d2 is None:
        raise RuntimeError(
            f"No descriptors for pair {i}->{j}"
        )

    matches = bf.match(d1, d2)
    matches = sorted(
        matches,
        key=lambda m: m.distance,
    )

    pts1 = np.float32(
        [k1[m.queryIdx].pt for m in matches]
    )

    pts2 = np.float32(
        [k2[m.trainIdx].pt for m in matches]
    )

    A, inliers = cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
    )

    if A is None or inliers is None:
        raise RuntimeError(
            f"estimateAffinePartial2D failed for pair {i}->{j}"
        )

    return A


A01 = estimate_affine(0, 1)
A12 = estimate_affine(1, 2)
A23 = estimate_affine(2, 3)

H01 = affine_to_homography(A01)
H12 = affine_to_homography(A12)
H23 = affine_to_homography(A23)

H21 = np.linalg.inv(H12)
H32 = np.linalg.inv(H23)
H31 = H21 @ H32


panorama = stitch_images_simple(
    [
        (imgs_sph_rgb[1], np.eye(3), masks_sph_float[1]),
        (imgs_sph_rgb[0], H01,       masks_sph_float[0]),
        (imgs_sph_rgb[2], H21,       masks_sph_float[2]),
        (imgs_sph_rgb[3], H31,       masks_sph_float[3]),
    ],
    canvas_w_factor=3,
    blend_method=1,
)


panorama_cropped = crop_black_borders(
    panorama,
    thresh=5,
)


fig = plt.figure(figsize=(20, 6))
plt.imshow(panorama_cropped)
plt.axis("off")

fig.subplots_adjust(
    left=0,
    right=1,
    top=1,
    bottom=0,
    wspace=0.02,
    hspace=0.02,
)

plt.show()
