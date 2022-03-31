# @Time: 2022.3.29 16:58
# @Author: Bolun Wu

import matplotlib.pyplot as plt
import numpy as np


# util func for figure saving
def save_fig(path):
    plt.savefig(path, format='pdf', bbox_inches='tight')
    
# low dimension
def func0(x):
    return x+x**4+x**5

# high dimension
def func1(x):
    x, y = x[:, 0], x[:, 1]
    return np.cos(x) + np.cos(2*y)

# low frequency
def func2(x):
    return np.cos(x)

# high frequency
def func3(x):
    return np.cos(5*x)
    
# mix frequency
def func4(x):
    return np.cos(x) + np.cos(3*x) + np.cos(5*x)

# funcion type lookup dict
func_dict = { 0: func0, 1: func1, 2: func2, 3: func3, 4: func4 }
