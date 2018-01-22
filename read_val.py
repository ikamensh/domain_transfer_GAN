import cv2
import numpy as np
import os
from util import preprocess

rainy_sim = []

test_dir = 'rainy'

for filename in os.listdir(test_dir):

    fullname = test_dir + '/' + filename
    img = cv2.imread(fullname)
    rainy_sim.append(np.expand_dims(img, axis=0))


rainy_sim = np.vstack(rainy_sim)

test_val = rainy_sim[:100]

rainy_sim = preprocess(rainy_sim)



sunny_sim = []

sunny_dir = 'sunny'

for filename in os.listdir(sunny_dir):

    fullname = sunny_dir + '/' + filename
    img = cv2.imread(fullname)
    sunny_sim.append(np.expand_dims(img, axis=0))


sunny_sim = np.vstack(sunny_sim)

sunny_val = sunny_sim[:100]

sunny_sim = preprocess(sunny_sim)

# test_l = np.zeros_like(test_pics)
# test_h = np.zeros_like(test_pics)
#
# for indx in range(test_pics.shape[0]):
#     hi, lo = split_hi_low(test_pics[indx])
#     test_h[indx] = hi
#     test_l[indx] = lo
#
# test_pics = None



