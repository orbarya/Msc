import numpy as np
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
from sol1 import *


def build_gaussian_pyramid(im, max_levels, filter_size):
    filter_vec = generate_row_binomial_coefficients(filter_size)
    reduce (im, filter_vec)
    g_i = im
    i = 0
    pyr = []
    while ((i < max_levels) and (g_i.shape[0] > 32) and (g_i.shape[1] > 32)):
        g_i = reduce(g_i, filter_vec)
        pyr.append(g_i)
        i = i + 1
    return pyr, filter_vec

def expand (im, filter_vec):
    zero_padded_im = zero_pad(im)
    return blur_spatial(zero_padded_im, 2 * filter_vec)


def zero_pad (im):
    a = np.zeros ((2*im.shape[0], 2*im.shape[1]))
    a[::2,::2] = im
    return a


def reduce (im, filter_vec):
    blurred_im = blur_spatial(im, filter_vec)
    return sub_sample(blurred_im)


def sub_sample(im):
    """
    Sub sample an image by taking every second row and every second column only.
    :param im:
    :return:
    """
    return im[::2, ::2]


def blur_spatial(im, filter_vec):
    """
    Blur using convolution.
    :param im:
    :param kernel_size:
    :return:
    """
    x_blurred = convolve(im, filter_vec)
    return convolve(x_blurred, filter_vec.T)


def generate_row_binomial_coefficients(size):
    """
    Return a row of binomial coefficients of the required size.
    :param size: Positive integer >= 2
    :return:
    """
    if size == 1:
        return np.array([1]).reshape((1, 1))
    second_row = np.array([1, 1]).reshape((1, 2))
    a = np.array([1, 1]).reshape((1, 2))
    for i in range(0, size - 2):
        a = convolve2d(a, second_row)
    return a


def build_laplacian_pyramid(im, max_levels, filter_size):
    g_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    l_i = []
    i = 0
    for i in range (0, len(g_pyr) - 1):
        l_i.append(g_pyr[i] - expand(g_pyr[i + 1], filter_vec))
        i = i + 1
    l_i.append(g_pyr[len(g_pyr) - 1])
    return l_i, filter_vec


# a = np.arange(12).reshape((6,2))
# print (a)
# filter_vec = generate_row_binomial_coefficients(3)
# b = sub_sample(a)
# print (b)


im=read_image('./presubmit/presubmit_externals/monkey.jpg', 1)
#imshow(im, 1)
g_pyr, filter_vec = build_gaussian_pyramid(im, 30, 3)
l_pyr, filter_vec = build_laplacian_pyramid(im, 30, 3)

for i in range(0, len(g_pyr)):
    imshow(g_pyr[i], 1)

for i in range(0, len(l_pyr)):
    imshow(l_pyr[i], 1)


#filter_vec = generate_row_binomial_coefficients(3)
#expanded_im = expand(im, filter_vec)
#imshow(expanded_im, 1)
plt.show()

