""" 
1st Assignment: enhancement and superresolution
2023/1
SCC0251

Name: Victor Henrique de Sa Silva
USP Number: 11795759
"""

import numpy as np 
import imageio.v2 as imageio
import matplotlib.pyplot as plt

class FrequencyFilter:
    def __init__(self) -> None:
        return
    
    
    def get_frequency_domain(self, image: np.array) -> np.array:
        return np.fft.fftshift(np.fft.fft2(image))
    

    def apply_filter(self, image: np.array, mask: np.array, show_img: bool=False, save_img: bool=False, filename: str="") -> np.array:
        F = self.get_frequency_domain(image)
        g = np.multiply(F, mask)

        result = np.real(np.fft.ifft2(np.fft.ifftshift(g)))
        rmax, rmin = result.max(), result.min()
        result = (255 * (result - rmin) / (rmax - rmin)).astype(np.uint16)

        if show_img:
            plt.imshow(result, cmap="gray")
            plt.show()

        if save_img:
            filename = "out" if not filename else filename
            imageio.imwrite(f"{filename}.png", result)

        return result

    
    def get_distances_from_center(self, P: int, Q: int) -> np.array:
        center_x, center_y = P/2, Q/2
        Y, X = np.ogrid[:P, :Q]
        return np.sqrt((X - center_x)**2 + (Y - center_y)**2)


    def get_ideal_lowpass_filter(self, image: np.array, radius: int, show_mask: bool=False) -> np.array:
        lowpass = np.zeros_like(image, dtype=np.float32)
        P, Q = image.shape

        # Selection of valid indexes
        mask = self.get_distances_from_center(P, Q) <= radius
        lowpass[mask] = 1.0

        if show_mask:
            plt.imshow(lowpass, cmap="gray")
            plt.show()

        return lowpass
    

    def get_ideal_highpass_filter(self, image: np.array, radius: int, show_mask: bool=False) -> np.array:
        highpass = np.ones_like(image, dtype=np.float32)
        P, Q = image.shape

        # Selection of valid indexes
        mask = self.get_distances_from_center(P, Q) <= radius
        highpass[mask] = 0.0

        if show_mask:
            plt.imshow(highpass, cmap="gray")
            plt.show()

        return highpass
    

    def get_ideal_bandpass_filter(self, image: np.array, smaller_radius: int, bigger_radius: int, show_mask: bool=False) -> np.array:
        bandpass = np.zeros_like(image, dtype=np.float32)
        P, Q = image.shape

        # Selection of valid indexes
        distances = self.get_distances_from_center(P, Q) 
        inner_circle = distances >= smaller_radius
        outter_circle = distances <= bigger_radius
        mask = inner_circle == outter_circle
        
        bandpass[mask] = 1.0
        
        if show_mask:
            plt.imshow(bandpass, cmap="gray")
            plt.show()

        return bandpass


    def get_laplacian_highpass_filter(self, image: np.array, show_mask: bool=False) -> np.array:
        P, Q = image.shape
        laplacian = 4*(np.pi**2)* (self.get_distances_from_center(P, Q)**2)
        if show_mask:
            plt.imshow(laplacian, cmap="gray")
            plt.show()

        return laplacian
    
    
    def get_gaussian_lowpass_filter(self, image: np.array, sigma_r: float, sigma_c: float, show_mask: bool=False) -> np.array:
        P, Q = image.shape
        Y, X = np.ogrid[:P, :Q]
        gaussian = np.exp(-((Y - (P/2))**2/(2*sigma_r**2) +  (X - (Q/2))**2/(2*sigma_c**2)))
        if show_mask:
            plt.imshow(gaussian, cmap="gray")
            plt.show()

        return gaussian


    def get_butterworth_lowpass_filter(self, image: np.array, D0: int, n: int, show_mask: bool=False) -> np.array:
        P, Q = image.shape
        distances = self.get_distances_from_center(P, Q)
        H = 1 / (1 + np.power(distances/D0, 2*n))
        if show_mask:
            plt.imshow(H, cmap="gray")
            plt.show()
        
        return H


    def get_butterworth_highpass_filter(self, image: np.array, D0: int, n: int, show_mask: bool=False) -> np.array:
        P, Q = image.shape
        distances = self.get_distances_from_center(P, Q)
        distances[distances == 0.0] = 1.0
        H = 1 / (1 + np.power(distances/D0, -2*n))
        if show_mask:
            plt.imshow(H, cmap="gray")
            plt.show()
        
        return H
    

    def get_butterworth_bandreject_filter(self, image: np.array, D0: int, n0: int, D1: int, n1: int, show_mask: bool=False) -> np.array:
        low1 = self.get_butterworth_lowpass_filter(image, D0, n0)
        low2 = self.get_butterworth_lowpass_filter(image, D1, n1)
        butterworth_bandreject = 1.0 - (low1 - low2)
        if show_mask:
            plt.imshow(butterworth_bandreject, cmap="gray")
            plt.show()
        
        return butterworth_bandreject

    
    def get_butterworth_bandpass_filter(self, image: np.array, D0: int, n0: int, D1: int, n1: int, show_mask: bool=False) -> np.array:
        low1 = self.get_butterworth_lowpass_filter(image, D0, n0)
        low2 = self.get_butterworth_lowpass_filter(image, D1, n1)
        butterworth_bandpass = low1 - low2
        if show_mask:
            plt.imshow(butterworth_bandpass, cmap="gray")
            plt.show()
        
        return butterworth_bandpass


def read_image(fname):
    return imageio.imread(f"{fname}")

def rmse(h, h_hat):
    return "%.4f" % np.sqrt(np.mean((h - h_hat)**2))

def read_inputs():
    img_name = input()
    gt_name = input()
    operation = int(input())
    return (img_name, gt_name, operation)

def read_params(op):
    params = []
    if op <= 1:
        params.append(int(input()))
    elif op == 2:
        params.append(int(input()))
        params.append(int(input()))
    elif op == 4:
        params.append(float(input()))
        params.append(float(input()))
    elif op == 5 or op == 6:
        params.append(int(input()))
        params.append(float(input()))
    elif 7 <= op <= 8:
        params.append(int(input()))
        params.append(int(input()))
        params.append(int(input()))
        params.append(int(input()))
    
    return params
        

if __name__ == "__main__":
    freq_filter = FrequencyFilter()

    img_name, gt_name, op = read_inputs()
    params, filter_mask = read_params(op), None
    img = read_image(img_name) 
    gt = read_image(gt_name) 

    if op == 0:
        filter_mask = freq_filter.get_ideal_lowpass_filter(img, params[0])
    elif op == 1:
        filter_mask = freq_filter.get_ideal_highpass_filter(img, params[0])
    elif op == 2:
        filter_mask = freq_filter.get_ideal_bandpass_filter(img, params[0], params[1])
    elif op == 3:
        filter_mask = freq_filter.get_laplacian_highpass_filter(img)
    elif op == 4:
        filter_mask = freq_filter.get_gaussian_lowpass_filter(img, params[0], params[1])
    elif op == 5:
        filter_mask = freq_filter.get_butterworth_lowpass_filter(img, params[0], params[1])
    elif op == 6:
        filter_mask = freq_filter.get_butterworth_highpass_filter(img, params[0], params[1])
    elif op == 7:
        filter_mask = freq_filter.get_butterworth_bandreject_filter(img, params[0], params[2], params[1], params[3])
    elif op == 8:
        filter_mask = freq_filter.get_butterworth_bandpass_filter(img, params[0], params[2], params[1], params[3])
        
    result = freq_filter.apply_filter(img, filter_mask)
    print(rmse(result, gt))


""" 
x 0) Ideal Low-pass - with radius r;
x 1) Ideal High-pass - with radius r; 
x 2) Ideal Band-pass - with radius r1 and r2;
x 3) Laplacian high-pass (edit: in a previous version we have wrongly asked for
low-pass);
x 4) Gaussian Low-pass - with σ1 and σ2;
x 5) Butterworth low-pass - with D0 and n;
x 6) Butterworth high-pass - with D0 and n;
7) Butterworth band-reject - with D0, D1, n1 and n2;
8) Butterworth band-pass - with D0, D1, n1 and n2;
"""