import math

import numpy as np
from scipy.misc import imread, imsave

from PIL import Image, ImageDraw

image = imread("/home/andrew/Downloads/images/12_L.png", mode='L')
image = np.array(image)
print(np.shape(image))

angle = np.pi / 7
transformation_matrix = [[np.cos(angle), 0 - np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]

vectors = np.array([100, 100, 2])
new_image = np.zeros([100, 100])

# Vector image
im = Image.new('RGBA', (100, 100), (0, 255, 0, 0))
draw = ImageDraw.Draw(im)

for i in xrange(0, 100):
    for j in xrange(0, 100):
        current_value = [[i], [j], [1]]
        new_value = np.matmul(transformation_matrix, [[i - 50], [j - 50], [1]])
        new_value[0][0] += 50
        new_value[1][0] += 50
        # print(str(current_value) + " -> " + str(new_value))

        # print("Hi, " + str(image[i][j]))

        new_x = int(math.floor(new_value[0][0]))
        new_y = int(math.floor(new_value[1][0]))

        if int(math.floor(new_value[0][0])) > 99 or int(math.floor(new_value[1][0])) > 99:
            continue


        new_image[int(math.floor(new_value[0][0]))][int(math.floor(new_value[1][0]))] = image[i][j]

        # vectors[i][j] = [new_x][new_y]

        if (i % 5 == 0 and j % 5 == 0):
            if(new_image[new_x][new_y] != 0):
                print("(" + str(i) + ", " + str(j) + ") --> (" + str(new_x) + ", " + str(new_y) + ")")
                blue = int(round(255.0 / 20.0 * math.sqrt((new_x - i) ** 2 + (new_y - j) ** 2)))
                print(int(round(math.sqrt((new_x - i) ** 2 + (new_y - j) ** 2))))
                # draw.point((new_y, new_x), fill=(128,0,0))
                draw.line((j, i, new_y, new_x), fill=(255.0 / 100.0 * i, 255 - 255 / 100 * j, blue))


imsave('fradient.png', image)
imsave('gradient.png', new_image)

print(vectors)

im.save("hradient.png")

