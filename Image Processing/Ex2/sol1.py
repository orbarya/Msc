import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

RGB_2_YIQ_MAT = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                          [0.212, -0.523, 0.311]])
YIQ_2_RGB_MAT = np.linalg.inv(RGB_2_YIQ_MAT)
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


def imdisplay(filename, representation):
    """Reads an image from file and displays an either RGB or a Grayscale representation of it.
    Parameters
    ----------
    filename : String
        Path to image file.
    representation : int
        Representation : 1 for gray-scale, 2 for RGB.
    """
    im_float = read_image(filename, representation)
    imshow(im_float, representation)


def imshow(im_float, representation):
    """Creates a new figure and displays the image as either RGB or grayscale.
    Parameters
    ----------
    im_float : np.float64 matrix
        The image to be shown normalized between [0,1] in grayscale or RGB representation.
    representation : int
        Representation : 1 for gray-scale, 2 for RGB.

    Returns
    -------
    """
    #im_float = im_float.clip(0, 1)
    plt.figure()
    if representation == 1:
        plt.imshow(im_float, cmap=plt.cm.gray)
    elif representation == 2:
        plt.imshow(im_float)
    else:
        raise Exception('Representation must be 1 or 2.')
    plt.draw()


# 3.3
def rgb2yiq(imRGB):
    """Converts an image from RGB space to YIQ
    Parameters
    ----------
    imRGB : np.float64 matrix
        The image to be converted normalized between [0,1] in RGB representation.

    Returns
    -------
    np.float64 matrix
        The image normalized between [0,1] in YIQ representation.
    """
    im_yiq = np.empty(imRGB.shape)
    for i in range(0, 3):
        im_yiq[:, :, i] = RGB_2_YIQ_MAT[i, 0] * imRGB[:, :, 0] + RGB_2_YIQ_MAT[i, 1] * imRGB[:, :, 1] + RGB_2_YIQ_MAT[i,
                                                                                                    2] * imRGB[:, :, 2]
    return im_yiq


def yiq2rgb(imYIQ):
    """Converts an image from YIQ space to RGB
    Parameters
    ----------
    imYIQ : np.float64 matrix
        The image to be converted normalized between [0,1] in YIQ representation.

    Returns
    -------
    np.float64 matrix
        The image normalized between [0,1] in RGB representation.
    """
    im_rgb = np.empty(imYIQ.shape)
    for i in range(0, 3):
        im_rgb[:, :, i] = YIQ_2_RGB_MAT[i, 0] * imYIQ[:, :, 0] + YIQ_2_RGB_MAT[i, 1] * imYIQ[:, :, 1] + \
                          YIQ_2_RGB_MAT[i, 2] * imYIQ[:, :, 2]
    return im_rgb


def histogram_equalize(im_orig):
    """
    Takes an RGB or gray-scale image in the [0,1] range and performs histogram equalization on it's intensities or
    Y component in the case of an RGB.
    :param im_orig: Original image.
    :return: Equalized image.
    """
    intensities = im_orig

    # If input is RGB perform equalization on Y component.
    if is_rgb(im_orig):
        yiq = rgb2yiq(im_orig)
        intensities = yiq[:, :, 0]

    # Transform to 0-255 range.
    intensities = (intensities * (k - 1)).round().astype(np.uint8)

    hist_orig, bounds = np.histogram(intensities, bins=k, range=(0, k))
    cum_hist = np.cumsum(hist_orig)

    # Normalize the cumulative histogram to have a codomain between 0-255
    normalized_cum_hist = cum_hist * (k - 1) / intensities.size

    # Linearly strech the normalized cumulative histogram to be between 0-255
    non_zero_indices = np.where(normalized_cum_hist > 0)
    m = non_zero_indices[0][0]
    t = np.round(((normalized_cum_hist - normalized_cum_hist[m]) / (normalized_cum_hist[(k - 1)] -
                                                                    normalized_cum_hist[m])) * (k - 1)).astype(np.uint8)

    # Calculate the equalized histogram.
    im_eq = t[intensities]
    hist_eq, bounds = np.histogram(im_eq, bins=k, range=(0, k))

    # Transform the image to the equalized image between [0,1]
    im_eq = t[intensities] / (k - 1)

    # If the original is RGB take the equalized Y component and transform back to RGB.
    if is_rgb(im_orig):
        yiq[:, :, 0] = im_eq
        im_eq = yiq2rgb(yiq)

    im_eq = im_eq.clip(0, 1)

    return [im_eq, hist_orig, hist_eq]


def is_rgb(im):
    """
    Checks if the given image has a pixel dimension of 3 thus concludes that this is an RGB representation image.
    :param im: Image
    :return: Whether this image is in RGB representation.
    """
    return len(im.shape) == 3


def quantize(im_orig, n_quant, n_iter):
    """
    Takes an image in RGB or grayscale representation, in [0,1] range and quantize it to n_quants.
    :param im_orig: Original image in either grayscale or RGB representation.
    :param n_quant: Number of quants.
    :param n_iter: Number of iterations to perform the error minimization before stopping.
    :return:
    """
    # If original image is RGB quantize the Y component.
    intensities = im_orig
    if is_rgb(im_orig):
        yiq = rgb2yiq(im_orig)
        intensities = yiq[:, :, 0]

    intensities = (intensities * (k - 1)).round().astype(np.uint8)

    hist_orig, bounds = np.histogram(intensities, bins=k, range=(0, k))
    cum_hist = np.cumsum(hist_orig)

    # Compute initial z values
    curr_z = init_z(cum_hist, n_quant, intensities.size)
    done_iter = 0
    converged = False
    error = np.empty(n_iter)

    # Iterating until convergence
    while (done_iter < n_iter) & (not converged):
        q, error[done_iter] = compute_q(hist_orig, curr_z)
        prev_z = curr_z
        curr_z = compute_z(q)
        done_iter += 1
        converged = is_converged(prev_z, curr_z)

    q = np.round(q).astype(np.uint8)

    if converged:
        error = error[:done_iter]

    t = generate_quant_lookup_table(q, curr_z)
    im_quant = t[intensities] / (k - 1)

    # If original image is RGB, take quantized Y component and transform the image back to RGB.
    if is_rgb(im_orig):
        yiq[:, :, 0] = im_quant
        im_quant = yiq2rgb(yiq)

    return [im_quant, error]


def generate_quant_lookup_table(q, z):
    """
    Given the quants, q, and the bounds, z, generates a lookup table for all intensities, which quant they belong to.
    :param q: The quants
    :param z: The bounds
    :return: The intensities to quants lookup table.
    """
    t = np.empty(k, np.uint8)
    for i in range(0, z.size - 1):
        t[z[i]:z[i + 1] + 1] = q[i]
    return t


def is_converged(prev_z, curr_z):
    """
    Checks if the current and prev z arrays are the same
    :param prev_z: Prev z array
    :param curr_z: Current z array
    :return: True iff the two arrays are identical.
    """
    return not (curr_z - prev_z).any()


def init_z(cum_hist, n_quant, num_of_pixels):
    """
    Computes the initial bin bounds for the minimization process.
    :param cum_hist: The image's cumulative histogram of intensities.
    :param n_quant: Number of quantizations required.
    :param num_of_pixels: Total number of pixels in the image.
    :return: Array of initial bin bounds - z.
    """
    z = np.empty(n_quant + 1, dtype=int)
    # pixels per quant
    ppq = num_of_pixels / n_quant
    z[0] = 0
    for i in range(1, n_quant + 1):
        z[i] = np.searchsorted(cum_hist, ppq * i, side='left')
    return z


def compute_z(q):
    """
    Given the array of quantizations q, computes the bin bounds z.
    :param q: Array of quantizations.
    :return: Array of bin bounds z.
    """
    z = np.concatenate((np.array([0]), np.round((q[1:] + q[:-1]) / 2).astype(np.uint16), np.array([k - 1])))
    return z


def compute_q(hist, z):
    """
    Given the current z array - bounds of bins, compute the quantizations.
    :param hist: The cumulative histogram of intensities of an image.
    :param z: Array of bin bounds - starts with 0 and ends with k - 1
    :return: A list holding the array of quantizations as float64 and the error of current q and z.
    """
    q = np.empty(z.size - 1)
    error = 0
    for i in range(0, z.size - 1):

        # Sum for each k=[z_i, z_i + 1] z_k * p(z)
        z_k = np.arange(z[i], z[i + 1])
        q[i] = ((z_k.dot(hist[z[i]: z[i + 1]])) / (np.sum(hist[z[i]: z[i + 1]])))
        error += (np.multiply((q[i] - z_k), (q[i] - z_k))).dot(hist[z[i]: z[i + 1]])
    return [q, error]


def quantize_rgb(im_orig, n_quant, n_iter):
    """
    Takes an image in RGB or grayscale representation, in [0,1] range and quantize it to n_quants.
    :param im_orig: Original image in either grayscale or RGB representation.
    :param n_quant: Number of quants.
    :param n_iter: Number of iterations to perform the error minimization before stopping.
    :return:
    """
    im_array = (im_orig.reshape((im_orig.shape[0]*im_orig.shape[1], im_orig.shape[2])))
    kmeans = KMeans(n_clusters=n_quant, max_iter=n_iter).fit(im_array)
    im_quant_array = kmeans.cluster_centers_[(kmeans.predict(im_array))]
    return [im_quant_array.reshape(im_orig.shape), kmeans.score(im_quant_array)]
