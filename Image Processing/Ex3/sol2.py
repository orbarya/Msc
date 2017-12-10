import numpy as np
from scipy.signal import convolve2d
from scipy.misc import imread as imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sol1 import *

k = 256


def read_image(filename, representation):
    """Reads an image from file and returns either an RGB or a Grayscale representation of the image.
    Parameters
    ----------
    filename : String
        Path to image file.
    representation : int
        Representation : 1 for grayscale, 2 for RGB.

    Returns
    -------
    np.float64 matrix
        The image normalized between [0,1] in grayscale or RGB representation.
    """
    if representation != 1 and representation != 2:
        raise Exception('Representation must be 1 or 2.')
    im = imread(filename, mode='RGB')
    im_float = im.astype(np.float64)
    im_float /= (k - 1)
    if (representation == 1) & (is_rgb(im_float)):
        im_float = rgb2gray(im_float)
    return im_float


def is_rgb(im):
    """
    Checks if the given image has a pixel dimension of 3 thus concludes that this is an RGB representation image.
    :param im: Image
    :return: Whether this image is in RGB representation.
    """
    return len(im.shape) == 3


def DFT(signal):
    """
    Fourier transform
    :param signal: dtype float64
    :return: dtype complex 128 - Complex Fourier signal
    """
    N = signal.shape[1]
    a = np.arange(start=0, stop=N).reshape((N, 1))
    DFT_MATRIX = np.exp((-2*np.pi*1j * a.dot(a.T) / N))
    return signal.dot(DFT_MATRIX)


def IDFT(fourier_signal):
    """
    Inverse fourier transform
    :param fourier_signal: dtype complex 128
    :return: dtype complex 128 - Complex signal
    """
    N = fourier_signal.shape[1]
    a = np.arange(start=0, stop=N).reshape((N, 1))
    IDFT_MATRIX = np.exp((2*np.pi*1j * a.dot(a.T) / N))
    return (1/N) * fourier_signal.dot(IDFT_MATRIX)


def DFT2(image):
    """
    2D Fourier transform
    :param image: dtype float64 grayscale image
    :return:
    """
    return (DFT((DFT (image)).T)).T


def IDFT2(fourier_image):
    """
    Inverse 2D fourier transform.
    :param fourier_image:
    :return:
    """
    return (IDFT((IDFT(fourier_image)).T)).T


def conv_der(im):
    """
    Image derivative magnitude using simple convolution.
    :param im: dtype float64 Input image
    :return: dtype float64 Image derivative magnitude
    """
    x_derivative_kernel = np.array([1, 0, -1]).reshape((3, 1))
    y_derivative_kernel = x_derivative_kernel.T
    dx = convolve2d(im, x_derivative_kernel, mode='same')
    dy = convolve2d(im, y_derivative_kernel, mode='same')

    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)


def fourier_der(im):
    """
    Calculates the image derivative using fourier transform.
    :param im:
    :return: Image Derivative
    """
    m = im.shape[0]
    n = im.shape[1]
    u = np.arange(start=-m/2, stop=m/2).reshape((m,1))
    v = np.arange(start=-n/2, stop=n/2).reshape((1,n))
    u = np.tile(u, (1,n))
    v = np.tile(v, (m,1))

    im_f = DFT2(im)
    im_f_shifted = np.fft.fftshift(im_f)

    dx = np.real(((2*np.pi*1j)/(m)) * IDFT2(np.fft.ifftshift(np.multiply(u,im_f))))
    dy = np.real(((2*np.pi*1j)/(n)) * IDFT2(np.fft.ifftshift(np.multiply(im_f, v))))
    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)


def blur_spatial(im, kernel_size):
    """
    Blur using convolution.
    :param im:
    :param kernel_size:
    :return:
    """
    kernel = generate_gaussian_kernel(kernel_size)
    return convolve2d(im, kernel, mode='same')


def generate_row_binomial_coefficients (size):
    """
    Return a row of binomial coefficients of the required size.
    :param size: Positive integer >= 2
    :return:
    """
    if size == 1:
        return np.array([1]).reshape((1,1))
    second_row = np.array([1, 1]).reshape((1,2))
    a = np.array([1, 1]).reshape((1,2))
    for i in range(0, size - 2):
        a = convolve2d(a, second_row)
    return a


def generate_gaussian_kernel(size):
    """
    Generate gaussian kernel of the required size.
    :param size:
    :return:
    """
    bin = generate_row_binomial_coefficients(size)
    kernel = (1/size**2) * convolve2d(bin, bin.T)
    return kernel


def blur_fourier(im, kernel_size):
    """
    Blurs an image using fourier transformation.
    :param im:
    :param kernel_size:
    :return:
    """
    m, n = im.shape
    padded_kernel = np.zeros(im.shape)
    kernel = generate_gaussian_kernel(kernel_size)
    row_start, row_end, col_start, col_end = find_kernel_indices(m, n, kernel_size)
    padded_kernel[row_start:row_end, col_start:col_end] = kernel
    shifted_padded_kernel = np.fft.ifftshift(padded_kernel)
    kernel_f = DFT2(shifted_padded_kernel)
    im_f = DFT2(im)
    return np.real(IDFT2(np.multiply(im_f, kernel_f)))


def find_kernel_indices(m, n, kernel_size):
    """
    Finds the indices for placing the kernel in the middle of the image.
    :param m:
    :param n:
    :param kernel_size:
    :return: The row start, row end, col start and col end indices.
    """
    c_m = np.floor(m / 2).astype(np.uint)
    c_n = np.floor(n / 2).astype(np.uint)
    k = np.floor(kernel_size / 2).astype(np.uint)
    row_start = c_m - k
    row_end = (c_m + k + 1).astype(np.uint)
    col_start = c_n - k
    col_end = (c_n + k + 1).astype(np.uint)
    return row_start, row_end, col_start, col_end

im = imread ('./monkey.jpg', 1)
der = conv_der(im)
imshow (der, 1)
plt.show()