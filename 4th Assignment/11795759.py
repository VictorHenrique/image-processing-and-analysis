""" 
4th Assignment: mathematical morphology
2023/1
SCC0251

Name: Victor Henrique de Sa Silva
USP Number: 11795759
"""

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class FloodFill:
    def __init__(self, image_path: str, x: int, y: int, conectivity: int) -> None:
        self.first_coord = (x, y)
        self.image = (imageio.imread(image_path) > 127).astype(np.uint8)
        self.numOfRows, self.numOfCols = self.image.shape
        
        # Right, down, left, and up neighbors
        self.neighbors = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        if conectivity == 8:
            # Bottom-right, top-right, top-left, and bottom-left neighbors
            self.neighbors += [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    
    def floodfill(self, display_result: bool) -> list:
        plt.subplot(121).imshow(self.image)
        x, y = self.first_coord
        
        color = not self.image[x, y]
        visited = set()
        q = deque([self.first_coord])
        while q:
            i, j = q.popleft()
            if (i, j) in visited:
                continue
            visited.add((i, j))
            
            self.image[i, j] = color
            for sx, sy in self.neighbors:
                new_x, new_y = i + sx, j + sy
                if 0 <= new_x < self.numOfRows and 0 <= new_y < self.numOfCols and self.image[new_x, new_y] == (not color):
                    q.append((i + sx, j + sy))

        plt.subplot(122).imshow(self.image)
        if display_result:
            plt.show()

        # Visited set is the connected component
        return list(visited) 

        
    def apply(self, display_result: bool=False) -> None:
        connected_components = self.floodfill(display_result)
        connected_components = sorted(connected_components)

        s = ""
        for _, coord in enumerate(connected_components):
            x, y = coord
            s += f"({x} {y}) "
        print(s)


if __name__ == "__main__":
    filename = input()
    x = int(input())
    y = int(input())
    c = int(input())
    
    FF = FloodFill(filename, x, y, c)
    FF.apply(False)