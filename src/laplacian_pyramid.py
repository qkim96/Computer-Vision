import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io


def build_laplacian_pyramid(G, levels=6):
    """Build Laplacian pyramid from an image using Gaussian downsampling."""
    # Gaussian Pyramid
    gpA = [G]                         # initialize the Gaussian pyramid with the original image, G
    for i in range(levels):           # build the Gaussian pyramid by repeatedly downsampling the image
        G = cv2.pyrDown(G)            # blur and downsample the image to get the next level of the pyramid
        gpA.append(G)                 # append the image to the pyramid

    # Laplacian Pyramid
    lpA = [gpA[levels]]               # initialize with the last (smallest) Gaussian level
    for i in range(levels, 0, -1):    # build Laplacian levels by subtracting upsampled Gaussian
        GE = cv2.pyrUp(gpA[i])        # upsample Gaussian at level i
        L = gpA[i - 1] - GE           # Laplacian = (Gaussian one level up) - (upsampled Gaussian)
        lpA.append(L)

    return lpA


def main():
    url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia.istockphoto.com%2Fphotos%2Fapple-and-orange-difference-picture-id637563258%3Fk%3D6%26m%3D637563258%26s%3D170667a%26w%3D0%26h%3DV3ZWrncLd8Fx8JdDODw0fCryk9-dimP9HS1wgwLsorI%3D&f=1&nofb=1&ipt=10e5cfd56eda60158ba3e0a3bc84a636c0fcda950b874a86ce42beeaf9866c73&ipo=images"
    im = skimage.io.imread(url) / 255.0

    apple = im[60:316, 246:]
    orange = im[63:319, 5:261]

    # Hard-cut image
    plt.figure(figsize=(20, 6))
    plt.subplot(131), plt.imshow(apple), plt.title("Apple " + str(apple.shape) + " " + str(apple.dtype))
    plt.subplot(132), plt.imshow(orange), plt.title("Orange " + str(orange.shape) + " " + str(orange.dtype))
    plt.subplot(133), plt.imshow(np.hstack((apple[:, :125], orange[:, 125:]))), plt.title("Hard-Cut")
    plt.tight_layout()
    plt.show()

    levels = 6
    applePyr = build_laplacian_pyramid(apple, levels=levels)
    orangePyr = build_laplacian_pyramid(orange, levels=levels)

    # Combine apple and orange pyramid to a new pyramid
    newPyr = []
    for i in range(levels + 1):
        half_width_at_scale = applePyr[i].shape[0] // 2

        combined_img = np.hstack(
            (applePyr[i][:, :half_width_at_scale], orangePyr[i][:, half_width_at_scale:])
        )
        newPyr.append(combined_img)

    # Show apple, orange and new pyramids at all levels
    plt.figure(figsize=(20, 6))
    for i in range(levels + 1):
        if i > 0:
            plt.subplot(3, 7, 1 + i), plt.imshow(applePyr[i] + 0.5), plt.title("Apple " + str(i))
            plt.subplot(3, 7, 8 + i), plt.imshow(orangePyr[i] + 0.5), plt.title("Orange " + str(i))
            plt.subplot(3, 7, 15 + i), plt.imshow(newPyr[i] + 0.5), plt.title("New " + str(i))
        else:
            plt.subplot(3, 7, 1 + i), plt.imshow(applePyr[i]), plt.title("Apple " + str(i))
            plt.subplot(3, 7, 8 + i), plt.imshow(orangePyr[i]), plt.title("Orange " + str(i))
            plt.subplot(3, 7, 15 + i), plt.imshow(newPyr[i]), plt.title("New " + str(i))
    plt.tight_layout()
    plt.show()

    # Reconstruct by adding HP to LP at each level
    recon = newPyr[0]
    for i in range(1, levels + 1):
        recon = cv2.pyrUp(recon)           # upsample the reconstructed image
        recon = cv2.add(recon, newPyr[i])  # add the image at this level to the reconstructed image

    plt.figure(figsize=(20, 6))
    plt.subplot(131), plt.imshow(apple), plt.title("Apple")
    plt.subplot(132), plt.imshow(orange), plt.title("Orange")
    plt.subplot(133), plt.imshow(recon), plt.title("Reconstructed (pyramid blend)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
