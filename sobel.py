import skimage
from skimage import filters, io 
import numpy as np


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# open bill-big.jpg as an image in grayscale
img = io.imread('bill-big.jpg', as_gray=True)
sobel_h = filters.sobel_h(img)
sobel_v = filters.sobel_v(img)

sobel_h = NormalizeData(sobel_h)
sobel_v = NormalizeData(sobel_v)

print(sobel_h)
io.imshow(sobel_h)
io.show()

# perform numpy bitwise or on sobel_h and sobel_v
sobel = sobel_h * sobel_v

io.imshow(sobel)
io.show()