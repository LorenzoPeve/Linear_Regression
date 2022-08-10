import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import bokeh
from bokeh.plotting import figure, show
from bokeh.models import tickers, ranges
from bokeh.io import output_notebook
output_notebook()
from typing import Union


def get_p_j_given_x(x, Bj):
    """Calculates P( y = j | x).

    Args:
        x: N x M features
        B: M features x K-1

    Returns:
        np.ndarray: N x K-1 matrix with the probabilities of each observation
        to be classified as a given category. 
    """
    numerator = np.exp(x @ Bj) # Returns N x K-1 Matrix
    # Note: It is critical to sum over the axis because it is
    # only within an observation that the probabilities must add up to 1.
    denominator = (1 + 
                    np.sum(np.exp(x @ Bj), axis = 1)).reshape(-1,1) # N Vector
    
    return numerator / denominator

def get_pK(x, Bj):
    """Calculates P( y = K | x).
    """    
    denominator = (1 + np.sum(np.exp(x @ Bj), axis = 1)).reshape(-1,1)
    return 1 / denominator

def get_L1_vector(x, yi, Bj):
    """
    Computes partial derivatives dL/dBj.

        Assumes Bj is a row vector with K−1 entries and X is a column array.

    Args:
        xi (np.ndarray): Column vector or N observation x M features
            matrix
        yi (np.ndarray): Column vector with categorical data.
        Bj (np.ndarray): Row vector

    """
    dL_dBj = []

    for category, bj in enumerate(Bj, start = 1):

        term_1 = np.sum(x[yi == category])
        denominator = np.ones(shape = x.shape)        
        for i, xi in enumerate(x):
            denominator[i] = (1 + np.sum(np.exp(xi * Bj))) # Vector Scaling
    
        numerator = x * np.exp(x * bj)
        dL_dBj.append(term_1 - np.sum(numerator/denominator))

    return dL_dBj

def get_Lprime2_matrix(x, K, Bj):
    """
    Computes partial second derivatives d2L/dBj dBk.

    Assumes Bj is a row vector with K−1 entries and X is a column array.

    Args:
        xi (np.ndarray): Column vector or N observation x M features
            matrix
        K (int): Number of categories or discrete values y can take
            from 1 to K.
        Bj (np.ndarray): Row vector with regression parameters.

    """
 
    l_prime2 = np.zeros(shape = (K-1, K-1)) # Matrix L2 is (j,k)

    # Approach: 
    #   Since x is a column vector and Bj is a row, vectorized operations make
    #   more sense. 
    #   The only explicit iteration 1:n is for the denominator.
    denominator = np.ones(shape = x.shape)

    for i, xi in enumerate(x):
        denominator[i] = (1 + np.sum(np.exp(xi * Bj))) # Vector Scaling
    
    # Note: symmetric matrix, we are esimating K-1 parameters
    for j in range(K-1):
        for k in range(0, K-1): #

            if j == k:
                f = -1
            else:
                f = 0

            l_prime2[j, k] = f*np.sum(x**2*np.exp(x * Bj[j])/denominator) + \
                    np.sum(x**2*np.exp(x * (Bj[j] + Bj[k]))/(denominator**2))

    return l_prime2

def newton_raphson(xArr, yArr, b_0, tolerance = 0.00001):
    """
    Performs Newton-Raphson root finding.
    
    Args:
        xArr (np.ndarray): Column array with x values.
        yArr (np.ndarray): Column array with y values (discrete).
        b_0 (float): Initial guess for regression parameters.
        tolerance (float): Stops iteration when difference between iterations
            is within tolerance.
    """

    k = len(b_0) + 1
    difference = tolerance * 5
    
    beta_iter = [b_0]
    while abs(difference) > tolerance:
        
        L_1 = get_L1_vector(xArr, yArr, b_0)
        L_2 = get_Lprime2_matrix(xArr, k, b_0)
        beta_1 = b_0 - np.linalg.solve(L_2, L_1)

        # Calculate difference and update iteration state
        difference = max(abs(np.array(beta_1) - np.array(b_0)))
        b_0 = beta_1
        beta_iter.append(b_0)
    
    return beta_1, beta_iter

###############################################################################
# Class illustration

# Initialization
n = 10000
x_i = np.random.normal(0, 1, size = (n,1))
Bj = np.array([-0.2,0,0.2,0.4]).reshape(1,-1)

# Probabilities j = 1 through K-1
p_array_K_minus_one = get_p_j_given_x(x_i, Bj)

# Probabilities j = K
p_K = 1 - np.sum(p_array_K_minus_one, axis = 1).reshape(-1,1)

# Same as doing
p_K_v2 = get_pK(x_i, Bj)
assert max(abs(p_K - p_K_v2)) < 10**-15

# Full array
p_array = np.concatenate([p_array_K_minus_one, p_K], axis = 1)

# Y values generation
y_i = []
for probabilities in p_array:
    y_random = np.random.choice(a = [1,2,3,4,5], 
                                size = 1,
                                p = probabilities)
    y_i.append(y_random[0])
y_i = np.array(y_i)


# Newon Raphson
beta_1, beta_iter = newton_n_iter(x_i, y_i, np.array([-0.5,0.5,0.5,0.5]))
print(beta_1)


# n = np.random.normal(size = (10000,1))
# beta = 

# df = pd.read_csv('data_1.csv')
# beta_1, beta_iter = newton_n_iter(df.x.values, df.y.values, np.array([0.2, 0.3, 0.4]))

# print()








###############################################################################
# Test 1
# x_i = np.ones(shape = (2,4))
# Bj = np.array([[0.1, 0.1, 0.1, 0.1],
#                [0.25, 0.25, 0.25, 0.25],
#                [0.5, 0.5, 0.5, 0.5],
#                [1, 1, 1, 1]])

# expected = [[0.240544371, 0.240544371,	0.240544371, 0.240544371],
#            [0.240544371, 0.240544371,	0.240544371, 0.240544371]]
# assert np.max(abs(get_p_j_given_x(x_i,Bj) - expected)) < 10**-8