""" 
Victor Henrique de Sa Silva
11795759
"""

import numpy as np
import imageio

class EnhanceImage:
    def __init__(self, imglow: str, imghigh: str, gamma: float=1.0, debug=False) -> None:
        self.imglow = np.array([None, None, None, None])
        self.imglow_name = imglow
        self.imghigh_name = imghigh
        self.gamma = gamma
        self.histogram = np.array([None, None, None, None])
        self.joint_histogram = None

        if debug:
            print("Filenames:")

        # Reading high resolution image 
        self.imghigh = imageio.imread(f"{self.imghigh_name}.png")

        # Reading low resolution images (0 to 3)
        for i in range(4):
            self.imglow[i] = imageio.imread(f"{self.filename}{i}.png")
            
            if debug: 
                print(f"{self.filename}{i}.png")

    def get_histogram(self, img_index: int) -> None:
        # Already has the image histogram 
        if self.histogram[img_index]:
            return
        
        # Counting the frequencies of values
        img_histogram = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            img_histogram[i] = np.sum(self.imglow[img_index] == i)
        self.histogram[img_index] = img_histogram

    def get_cumulative_histogram(self, histogram: np.array) -> np.array:
        for i in range(1, 256):
            histogram[i] += histogram[i-1]
        return histogram

    def joint_cumulative_histogram(self) -> np.array:
        if not self.joint_histogram:
            for i in range(4):
                self.get_histogram(i)
            
            self.joint_histogram = self.get_cumulative_histogram(self.histogram.sum(axis=0)) 
        
    def transform(self, img_index: int, transformation_function: function) -> np.array:
        
        return np.array((2, 2))  
        
