import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.data


def main():
    img = skimage.data.astronaut()
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(np.float32) / 255.

    blur_img0 = cv2.boxFilter(img_gray.copy(), ksize=(11, 11), ddepth=-1)      # Box filter
    blur_img1 = cv2.GaussianBlur(img_gray.copy(), ksize=(11, 11), sigmaX=-1)   # Gaussian filter

    subtracted0 = cv2.subtract(img_gray, blur_img0)
    subtracted1 = cv2.subtract(img_gray, blur_img1)

    fig, axes = plt.subplots(2, 3)
    fig.tight_layout(pad=1.0)

    im = axes[0, 0].imshow(img_gray, cmap="gray")  # original grayscale image
    axes[0, 1].imshow(blur_img0, cmap="gray")      # blur image with Box filter
    axes[0, 2].imshow(subtracted0, cmap="gray")    # original - box blur (edges)
    axes[1, 0].imshow(img_gray, cmap="gray")       # original grayscale image
    axes[1, 1].imshow(blur_img1, cmap="gray")      # blur image with Gaussian filter
    axes[1, 2].imshow(subtracted1, cmap="gray")    # original - gaussian blur (edges)

    axes[0, 0].set_title("Original Grayscale Image", fontsize=8)
    axes[0, 1].set_title("Blur Image with\nBox Filter", fontsize=8)
    axes[0, 2].set_title("Subtracted Image with\nBox Filtered Image", fontsize=8)
    axes[1, 0].set_title("Original Grayscale Image", fontsize=8)
    axes[1, 1].set_title("Blur Image with\nGaussian Filter", fontsize=8)
    axes[1, 2].set_title("Subtracted Image with\nGaussian Filtered Image", fontsize=8)

    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()


if __name__ == "__main__":
    main()
