from numpy import diag,zeros, diag
from numpy.linalg import svd as svd, norm as norm, matrix_rank as matrix_rank
from matplotlib import pyplot as plt
import scipy.misc as misc


def display_image(image, title):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


def svd_compression(U, sigma, V, r):
    m_k = [None]*512
    ratios = [None]*512
    distances = [None]*512
    for k in range (0, 512):
        m_k[k], ratios[k], distances[k] = svd_k_compression(U, sigma, V, k, r)
    return m_k, ratios, distances


def svd_k_compression(U, sigma, V, k, r):
    n = sigma.size
    s_k = sigma.copy()
    s_k[k:] = 0
#     print(U.shape)
#     print (s_k.shape)
#     print((V.T).shape)
    m_k = (U.dot(diag(s_k)).dot(V))
    compression_ratio = (2*k*n+k)/(2*n*r+r)
    frobenius = norm(input_image - m_k)
    return m_k, compression_ratio, frobenius


input_image = misc.ascent()

display_image(input_image, "Original Image")

U, sigma, V = svd(input_image)
rank = matrix_rank(input_image)

m_k, ratios, distances = svd_compression(U, sigma, V, rank)
display_image(m_k[255], "Compressed Image - k value"
                        "- 255 - distance - " + str(distances[255]))
display_image(m_k[127], "Compressed Image - k value"
                        "- 127 - distance - " + str(distances[127]))
display_image(m_k[65], "Compressed Image - k value"
                       "- 65 - distance - " + str(distances[65]))
display_image(m_k[31], "Compressed Image - k value"
                       "- 31 - distance - " + str(distances[31]))
display_image(m_k[15], "Compressed Image - k value"
                       "- 15 - distance - " + str(distances[15]))

plt.figure()
plt.xlabel("k Value")
plt.ylabel("Compression Ratio")
plt.plot(ratios)

plt.figure()
plt.xlabel("k Value")
plt.ylabel("Frobenius Distance")
plt.plot(distances)

