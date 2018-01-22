# import cv2
# import tensorflow as tf
# image = cv2.imread("picture.jpg")
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("Over the Clouds", image)
# cv2.imshow("Over the Clouds - gray", gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite("grey.jpg", gray_image)

import os

import cv2
import numpy as np

from util import preprocess

directory = 'dataset/rain/training'

all_pics = []

for filename in os.listdir(directory):

    fullname = directory + '/' + filename
    img = cv2.imread(fullname)
    all_pics.append(np.expand_dims(img, axis=0))

def split_n_slide(pics):

    rainy_pics = []
    sunny_pics = []

    for pic in all_pics:
        sunny = pic[:,:, :pic.shape[2]//2]
        rainy = pic[:,:, pic.shape[2]//2:]
        assert sunny.shape == rainy.shape
        for i in range(0, sunny.shape[1]-255, 256):
            for j in range(0, sunny.shape[2] - 255, 256):
                i_e = i + 256
                j_e = j + 256
                sunny_pics.append( sunny[:,i:i_e, j:j_e])
                rainy_pics.append( rainy[:, i:i_e, j:j_e])

    return np.vstack(rainy_pics), np.vstack(sunny_pics)

rainy, sunny = split_n_slide(all_pics)

# sunny_val = sunny[:500]
# rainy_val = rainy[:500]


rainy = preprocess(rainy)
sunny = preprocess(sunny)

# kernel = np.ones((7, 7), np.float32) / 490
# rainy_l = np.zeros_like(rainy)
# rainy_h = np.zeros_like(rainy)
# def split_hi_low(img):
#
#     lo = cv2.filter2D(img, -1, kernel)
#     hi = img - lo
#     # hi = cv2.Laplacian(img, 5)
#     # lo = img - hi
#
#     return hi, lo
#
# for indx in range(rainy.shape[0]):
#     hi, lo = split_hi_low(rainy[indx])
#     rainy_h[indx] = hi
#     rainy_l[indx] = lo














