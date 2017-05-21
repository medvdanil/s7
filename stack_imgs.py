import cv2
import numpy as np

imgs = [[], [], []]

names = ['booking', 'DC', "RASK"]

for i_n in range(len(names)):
    image = None
    for k in range(12):
        img = cv2.imread('IBP_graphics_%s_%d.png' % (names[i_n], k))
        if image is None:
            image = np.zeros(tuple(np.array(img.shape[:2]) * (3, 4)) + (3,), dtype = img.dtype)
            print(image.shape)
        i = k % 3
        j = k // 3
        print(i * img.shape[0], (i + 1)* img.shape[0],j * img.shape[1], (j + 1)* img.shape[1])
        image[i * img.shape[0]: (i + 1)* img.shape[0],j * img.shape[1]: (j + 1)* img.shape[1]] = img
        cv2.imwrite('IBP_graphics_%s.png' % names[i_n], image)
