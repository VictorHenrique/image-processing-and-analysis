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
        img_histogram, _ = np.histogram(self.imglow[img_index], bins=range(257))
        for i in range(1, 256):
            img_histogram[i] += img_histogram[i-1]
        self.histogram[img_index] = img_histogram

    def joint_cumulative_histogram(self) -> np.array:
        if not self.has_joint_histogram:
            for i in range(4):
                self.get_cumulative_histogram(i)

            self.joint_histogram = self.histogram.sum(axis=0) / 4
            self.has_joint_histogram = True
    
    def plt_transformation(self, new_img: np.array, img_index: int=-1) -> None:
        _, ax = plt.subplots(1, 2)
        ax[0].set_xlim([len(new_img), 0])
        ax[0].set_ylim([len(new_img[0]), 0])
        ax[1].set_xlim([len(new_img), 0])
        ax[1].set_ylim([len(new_img[0]), 0])
        
        # Original Image
        if img_index == -1:
            ax[0].set_title(f"{self.imghigh_name}")
            ax[0].imshow(self.imghigh, cmap="gray")
        else:
            ax[0].set_title(f"{self.imglow_name}{img_index}.png")
            ax[0].imshow(self.imglow[img_index], cmap="gray")

        # Transformed Image
        ax[1].set_title(f"Transformed {self.imglow_name}{img_index}.png")
        ax[1].imshow(new_img, cmap="gray")
        plt.show()

    def histogram_equalization(self, img_index: int, joint_equalization: bool=False, L: np.uint32=256, show_image: bool=False) -> np.array:
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
        for z in range(L):
            s = ((L - 1) / float(m*n)) * histogram[z]
            new_image[np.where(self.imglow[img_index] == z)] = s
        self.imglow[img_index] = new_image

        if show_image:
            self.plt_transformation(new_image, img_index)

        return new_image.copy()

    def gamma_correction(self, img_index: int, show_image: bool=False) -> np.array:
        new_image = np.zeros_like(self.imglow[img_index], dtype=np.uint32)
        for i, row in enumerate(self.imglow[img_index]):
            for j, _ in enumerate(row):
                new_image[i, j] = 255 * (np.power(self.imglow[img_index][i, j] / 255, 1 / self.gamma))
        
        self.imglow[img_index] = new_image.copy()

        if show_image:
            self.plt_transformation(new_image, img_index)

        return new_image.copy()
    
    def superresolution(self, show_image: bool=False) -> np.array:
        if self.superres:
            return self.superres.copy()
        
        m, n = self.superres_shape
        self.superres = np.zeros(self.superres_shape, dtype=np.uint32)
        for i in range(0, m, 2):
            for j in range(0, n, 2):
                low_i, low_j = i // 2, j // 2
                self.superres[i][j] = self.imglow[0][low_i, low_j]
                self.superres[i][j+1] = self.imglow[1][low_i, low_j]
                self.superres[i+1][j] = self.imglow[2][low_i, low_j]
                self.superres[i+1][j+1] = self.imglow[3][low_i, low_j]
        
        if show_image:
            self.plt_transformation(self.superres, 0)
        
        return self.superres.copy()
    
    # Got from: https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
    def gaussian_kernel(self, length: int=3, sigma: float=1.0) -> np.array:
        ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
        gaussian = np.exp(-0.5 * np.square(ax) / sigma**2) 
        kernel = np.outer(gaussian, gaussian)

        return kernel / kernel.sum()

    def apply_convolution(self, conv_kernel: np.array, img_index: int=-1) -> np.array:
        cm, cn = conv_kernel.shape
        cm //= 2
        cn //= 2

        if img_index == -1:
            image = self.superres.copy()
        else:
            image = self.imglow[img_index].copy()
        
        m, n = image.shape 
        new_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(cm, m-cm):
            for j in range(cn, n-cn):
                new_image[i, j] = np.multiply(image[i-cm:i+cm+1, j-cn:j+cn+1], conv_kernel).sum()
        
        return new_image
    
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    def unsharp_mask(self, length: int=3, sigma: float=0.5, k: float=0.5) -> None:
        gaussian_kernel = self.gaussian_kernel(length=length, sigma=sigma)
        smoothed_img = (self.apply_convolution(gaussian_kernel))
        self.superres = self.superres.astype(np.uint8) -  k*(self.superres.astype(np.uint8) - smoothed_img)

def rmse(h, h_hat):
    return "%.4f" % np.sqrt(np.mean((h - h_hat)**2))

def read_input():
    imglow = input()
    imghigh = input()
    op = int(input())
    gamma = float(input())

    return imglow, imghigh, op, gamma

def option_0(enhancer):
    _ = enhancer.superresolution(show_image=False)
    # enhancer.unsharp_mask(length=5,sigma=0.2, k=1)
    return rmse(enhancer.imghigh, enhancer.superres)

def option_1(enhancer):
    for i in range(4):
        _ = enhancer.histogram_equalization(i)
    
    return option_0(enhancer)

def option_2(enhancer):
    for i in range(4):
        _ = enhancer.histogram_equalization(i, joint_equalization=True)
    
    return option_0(enhancer)

def option_3(enhancer):
    for i in range(4):
        _ = enhancer.gamma_correction(i)

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

