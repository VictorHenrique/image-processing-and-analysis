""" 
Victor Henrique de Sa Silva
11795759
"""

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

class ImageEnhancer:
    def __init__(self, imglow: str, imghigh: str, gamma: float=1.0) -> None:
        self.imglow = np.array([None, None, None, None])
        self.imglow_name = imglow
        self.imghigh_name = imghigh
        self.gamma = gamma
        self.histogram = np.array([None, None, None, None])
        self.joint_histogram = None
        self.has_joint_histogram = False

        # Reading high resolution image 
        self.imghigh = imageio.imread(f"{self.imghigh_name}")

        # Reading low resolution images (0 to 3)
        for i in range(4):
            self.imglow[i] = imageio.imread(f"{self.imglow_name}{i}.png")
        
        self.superres_shape = self.imghigh.shape
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
        if not self.has_joint_histogram:
            for i in range(4):
                self.get_cumulative_histogram(i)

            self.joint_histogram = self.histogram.sum(axis=0) / 4
            self.has_joint_histogram = True
    
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

        maxValue = (0, 0)
        for i, num in enumerate(histogram):
            if num > maxValue[1]:
                maxValue = (i, num)

        # Applying transformation
        for z in range(L):
            s = ((L - 1) / float(m*n)) * histogram[z]
            new_image[np.where(self.imglow[img_index] == z)] = s
        self.imglow[img_index] = new_image

        if show_image:
            self.plt_transformation(img_index, new_image)

        return new_image.copy()

    def gamma_correction(self, img_index: int, show_image: bool=False) -> np.array:
        new_image = np.zeros_like(self.imglow[img_index], dtype=np.uint8)
        for i, row in enumerate(self.imglow[img_index]):
            for j, _ in enumerate(row):
                new_image[i, j] = 255 * (np.power(self.imglow[img_index][i, j] / 255, 1 / self.gamma))
        
        self.imglow[img_index] = new_image

        if show_image:
            self.plt_transformation(img_index, new_image)

        return new_image.copy()
    
    def superresolution(self, show_image: bool=False) -> np.array:
        if self.superres:
            return self.superres.copy()
        
        m, n = self.superres_shape
        self.superres = np.zeros(self.superres_shape, dtype=np.uint8)
        for i in range(0, m, 2):
            for j in range(0, n, 2):
                low_i, low_j = i // 2, j // 2
                self.superres[i][j] = self.imglow[0][low_i, low_j]
                self.superres[i][j+1] = self.imglow[1][low_i, low_j]
                self.superres[i+1][j] = self.imglow[2][low_i, low_j]
                self.superres[i+1][j+1] = self.imglow[3][low_i, low_j]
        
        if show_image:
            self.plt_transformation(0, self.superres)
        
        return self.superres.copy()

def rmse(h, h_hat):
    return np.sqrt(np.mean((h - h_hat)**2))

def read_input():
    imglow = input()
    imghigh = input()
    op = int(input())
    gamma = float(input())

    return imglow, imghigh, op, gamma

def option_0(enhancer):
    new_image = enhancer.superresolution(show_image=False)
    plt.imshow(enhancer.imghigh - enhancer.superres, cmap="gray")
    plt.show()
    # plt.imshow(img - enhancer.imglow[0], cmap="gray")
    # print(img[0,0], img[0,1], img[1,0], img[1,1])
    return round(rmse(enhancer.imghigh, new_image), 4)

def option_1(enhancer):
    for i in range(4):
        _ = enhancer.histogram_equalization(i)
    
    return option_0(enhancer)

def option_2(enhancer):
    for i in range(4):
        _ = enhancer.histogram_equalization(i, joint_equalization=True)
    
    return option_0(enhancer)

def option_3(enhancer):
    img_rmse = np.array([0, 0, 0, 0])
    for i in range(4):
        new_image = enhancer.gamma_correction(i)
        img_rmse[i] = rmse(enhancer.imglow[i], new_image)

    return option_0(enhancer)

if __name__ == "__main__":
    imglow, imghigh, op, gamma = read_input()
    enhancer = ImageEnhancer(imglow, imghigh, gamma)
    
    operation = {
        0: option_0,
        1: option_1,
        2: option_2,
        3: option_3
    }

    print(operation[op](enhancer))
