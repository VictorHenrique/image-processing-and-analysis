""" 
Victor Henrique de Sa Silva
11795759
"""

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

class EnhanceImage:
    def __init__(self, imglow: str, imghigh: str, gamma: float=1.0) -> None:
        self.imglow = np.array([None, None, None, None])
        self.imglow_name = imglow
        self.imghigh_name = imghigh
        self.gamma = gamma
        self.histogram = np.array([None, None, None, None])
        self.joint_histogram = None

        # Reading high resolution image 
        self.imghigh = imageio.imread(f"{self.imghigh_name}.png")

        # Reading low resolution images (0 to 3)
        for i in range(4):
            self.imglow[i] = imageio.imread(f"{self.imglow_name}{i}.png")
        
        self.superres_shape = (len(self.imglow[0]) * 2, len(self.imglow[0][0]) * 2)
        self.superres = None

    def get_cumulative_histogram(self, img_index: int) -> None:
        # Already has the image histogram 
        if self.histogram[img_index]:
            return
        
        # Counting the frequencies of values
        img_histogram = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            img_histogram[i] = np.sum(self.imglow[img_index] == i)
            img_histogram[i] += img_histogram[i-1] if i > 0 else 0
        self.histogram[img_index] = img_histogram

    def joint_cumulative_histogram(self) -> np.array:
        if not self.joint_histogram:
            for i in range(4):
                self.get_cumulative_histogram(i)

            self.joint_histogram = self.histogram.sum(axis=0)
    
    def plt_transformation(self, img_index: int, new_img: np.array) -> None:
        _, ax = plt.subplots(1, 2)
        ax[0].set_xlim([len(new_img), 0])
        ax[0].set_ylim([len(new_img[0]), 0])
        
        # Original Image
        ax[0].set_title(f"{self.imglow_name}{img_index}.png")
        ax[0].imshow(self.imglow[img_index], cmap="gray")

        # Transformed Image
        ax[1].set_title(f"Transformed {self.imglow_name}{img_index}.png")
        ax[1].imshow(new_img, cmap="gray")
        plt.show()

    def histogram_equalization(self, img_index: int, joint_equalization: bool=False, L: np.uint8=256, show_image: bool=False) -> np.array:
        m, n = self.imglow[img_index].shape
        new_image = np.zeros_like(self.imglow[img_index])
        
        # Choosing histogram
        if joint_equalization:
            self.joint_cumulative_histogram()
            histogram = self.joint_histogram
        else:
            self.get_cumulative_histogram(img_index)
            histogram = self.histogram[img_index]

        # Applying transformation
        for i, row in enumerate(self.imglow[img_index]):
            for j, num in enumerate(row):
                new_image[i, j] = ((L - 1) / (m*n)) * histogram[num]
        
        if show_image:
            self.plt_transformation(img_index, new_image)

        return new_image

    def gamma_correction(self, img_index: int, show_image: bool=False) -> np.array:
        new_image = np.zeros_like(self.imglow[img_index], dtype=np.uint8)
        for i, row in enumerate(self.imglow[img_index]):
            for j, _ in enumerate(row):
                new_image[i, j] = 255 * (np.power(self.imglow[img_index][i, j] / 255, 1 / self.gamma))
        
        if show_image:
            self.plt_transformation(img_index, new_image)

        return new_image
    
    def superresolution(self, show_image: bool=False) -> np.array:
        if self.superres:
            return self.superres.copy()
        
        m, n = self.superres_shape
        self.superres = np.zeros(self.superres_shape)
        for i in range(0, m, 2):
            for j in range(0, n, 2):
                low_i, low_j = i // 2, j // 2
                self.superres[i][j] = self.imglow[0][low_i, low_j]
                self.superres[i][j + 1] = self.imglow[1][low_i, low_j]
                self.superres[i + 1][j] = self.imglow[2][low_i, low_j]
                self.superres[i + 1][j + 1] = self.imglow[3][low_i, low_j]
        
        if show_image:
            self.plt_transformation(0, self.superres)
        
        return self.superres.copy()