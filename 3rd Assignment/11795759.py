""" 
3rd Assignment: image descriptors
2023/1
SCC0251

Name: Victor Henrique de Sa Silva
USP Number: 11795759
"""

import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import convolve
from collections import Counter
import matplotlib.pyplot as plt

class HoGDescriptor:
    def __init__(self, X_0: np.array, X_1: np.array, X_test: np.array) -> None:
        self.X_0_rgb = X_0
        self.X_1_rgb = X_1
        self.X_test_rgb = X_test
        self.X0_length = len(X_0)
        self.X1_length = len(X_1)
        self.Xtest_length = len(X_test)

        self.preprocess_images()

        self.hog_descriptor()

    def preprocess_images(self) -> None:
        self.X_0 = []
        for i in range(self.X0_length):
            self.X_0.append(self.turn_grayscale(self.X_0_rgb[i]))
        np.array(self.X_0)
        
        self.X_1 = []
        for i in range(self.X1_length):
            self.X_1.append(self.turn_grayscale(self.X_1_rgb[i]))
        self.X_1 = np.array(self.X_1)

        self.X_test = []
        for i in range(self.Xtest_length):
            self.X_test.append(self.turn_grayscale(self.X_test_rgb[i]))
        self.X_test = np.array(self.X_test)


    def turn_grayscale(self, img: np.array) -> np.array:
        gs = (.299 * img[:,:,0]) + (.587 * img[:,:,1]) + (.114 * img[:,:,2])
        return gs.astype(np.int32)
    

    def get_gradients(self) -> None:
        horizontal_sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        vertical_sobel = np.transpose(horizontal_sobel)

        self.gradients = {
            "X_0": [[convolve(self.X_0[i], vertical_sobel), convolve(self.X_0[i], horizontal_sobel)] for i in range(self.X0_length)],
            "X_1": [[convolve(self.X_1[i], vertical_sobel), convolve(self.X_1[i], horizontal_sobel)] for i in range(self.X1_length)],
            "X_test": [[convolve(self.X_test[i], vertical_sobel), convolve(self.X_test[i], horizontal_sobel)] for i in range(self.Xtest_length)]
        }

    
    def get_magnitudes(self, img_set: str, idx: int) -> np.array:
        mag = np.sqrt(self.gradients[img_set][idx][0]**2 + self.gradients[img_set][idx][1]**2)
        return mag / mag.sum()
    

    def get_angles(self, img_set: str, idx: int) -> np.array:
        return np.arctan(self.gradients[img_set][idx][0] / self.gradients[img_set][idx][1])


    def hog_descriptor(self) -> None:
        # Silencing the warnings
        np.seterr(divide="ignore", invalid="ignore")
        
        self.get_gradients()
        self.hog = {"X_0": [], "X_1": [], "X_test": []}
        for key in self.hog.keys():
            for i, _ in enumerate(self.gradients[key]):
                mag = self.get_magnitudes(key, i)
                angles = np.degrees(self.get_angles(key, i) + np.pi/2) // 20
                self.hog[key].append(np.array([mag[np.where(angles == j)].sum() for j in range(9)]))
        self.hog["X_0"], self.hog["X_1"], self.hog["X_test"] = np.array(self.hog["X_0"]), np.array(self.hog["X_1"]), np.array(self.hog["X_test"])


    def knn(self, k: int=3) -> list:
        # Distances list structure: (distance, class) 
        classes = []
        for i in range(self.Xtest_length):
            distances_0 = np.array([[np.sqrt(np.sum((self.hog["X_test"][i] - self.hog["X_0"][j])**2)), 0] for j in range(self.X0_length)])
            distances_1 = np.array([[np.sqrt(np.sum((self.hog["X_test"][i] - self.hog["X_1"][j])**2)), 1] for j in range(self.X1_length)])

            # Classes of the k smallest distances
            distances = np.concatenate((distances_0, distances_1))
            k_distances = Counter(np.array(sorted(distances, key=lambda x: x[0]))[:k, 1])
            classes.append("1" if k_distances[1] > k_distances[0] else "0")

        return classes


def read_input() -> tuple:
    X0_paths = input()
    X1_paths = input()
    Xtest_paths = input()
    
    X0 = []
    for path in X0_paths.split():
        X0.append(imageio.imread(f"{path}"))
    
    X1 = []
    for path in X1_paths.split():
        X1.append(imageio.imread(f"{path}"))

    Xtest = []
    for path in Xtest_paths.split():
        Xtest.append(imageio.imread(f"{path}"))

    return (np.array(X0), np.array(X1), np.array(Xtest))

if __name__ == "__main__":
    X0, X1, Xtest = read_input()
    descriptor = HoGDescriptor(X0, X1, Xtest)

    classes = descriptor.knn()
    print(" ".join(classes))
