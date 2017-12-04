import numpy as np
from scipy.misc import derivative
from sklearn.datasets import make_spd_matrix


def chol(A):
    n = A.shape[0]
    if n == 1:
        alpha = A[0]
        if alpha <= 0:
            return False, np.array([1], ndmin=2)
        return True, np.array([np.sqrt(alpha)], ndmin=2)
    A_hat = A[:-1,:-1]
    found, L_hat = chol(A_hat)
    if (not found):
        print (L_hat.shape)
        print (np.array([0], ndmin=2).shape)
        return False, np.concatenate((L_hat, np.array([0], ndmin=2)), 0)
    a = np.array([A[:-1, -1]], ndmin=2).T
    l = (np.linalg.inv(L_hat)).dot(a)
    alpha = A[-1,-1]
    lambda_sqr = alpha - np.transpose(l).dot(l)
    if lambda_sqr <= 0:
        print (np.linalg.inv(A_hat).shape)
        print (np.array([-1], ndmin=2).shape)
        return False, np.concatenate((np.linalg.inv(A_hat).dot(a), np.array([-1], ndmin=2)), 0)

    else:
        lambda1 = np.sqrt(lambda_sqr)
        lambda_col = np.zeros((n,1))
        lambda_col[-1] = lambda1
        B = np.concatenate((L_hat, l.T), 0)
        L = np.concatenate((B, lambda_col), 1) #np.zeros(np.shape((n-1,0)))))
        return True, L


def non_pd (n):
    A = np.random.random_integers(-200, 200, size=(n,n))
    return (A + A.T) / 2


def gradient(f, x):
    grad = np.empty(x.shape)
    for i in range (0, grad.shape[0]):
        grad[i] = derivative (constantiate_element_i(f, x, i), x[i])
    return grad


def hessian(f, x):
    n = x.shape[0]
    hessian = np.empty((n, n))

    def grad_f(z):
        return gradient(f, z)

    for i in range (0, n):
        def grad_f_element_i(z):
            return (grad_f(z))[i]
        for j in range (0, n):
            #grad_f_const_i = constantiate_element_i(grad_f_element_i, x, i)
            hessian[i,:] = (gradient(grad_f_element_i, x)).T
    return hessian


def constantiate_element_i(f, x, i):
    def scalar_f(x_i):
        y = np.copy(x)
        y[i] = x_i
        return f(y)
    return scalar_f


def func (x):
    A = np.array([[2, 3, 4],
                  [1, 2, 3],
                  [3, 4, 5]])
    return x.T.dot(A).dot(x)

#scalar_func = constantiate_element_i(func, np.array([1, 2, 1]), 2)
#print (func(np.array([1, 2, 3])))
print (hessian(func, np.array([5,3,4])))
# Non positive definite generator
# A = non_pd (4)
# B = make_spd_matrix(4)
# print (A)
# print ("--------------------------------------------")
# found, L = chol(A)
# if not found:
#     print("The matrix is not positive definite. The following vector proves this : ")
#     print(L)
#     print("The x^TAx : ")
#     print(((L.T).dot(A)).dot(L))
# elif found:
#     print("Found : ")
#     print(L)
