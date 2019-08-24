import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def histogram(data: np.ndarray) -> None:
    """Creates a histogram of a tensorflow tensor. 

    Parameters
    ----------
    data: np.ndarray object

    Returns
    --------


    """

    if data.shape[0] < data.shape[1]: #it is a row vector
        _ = plt.hist(data.T, bins='auto')
        plt.title("Histogram with 'auto' bins")
        plt.show()

    elif data.shape[0] > data.shape[1]: #it is a column vector
        _ = plt.hist(data, bins='auto')
        plt.title("Histogram with 'auto' bins")
        plt.show()

    else:
        ValueError: "you can't get a histogram of a square matrix"


def describe(data: np.ndarray) -> None:
    """


    """

    print("Scipy Stats Describe")
    print(stats.describe(data.T))

    print(f"The Sum of the array is: {np.sum(data)}")
    print(f"The array shape is: {data.shape}")


    return None
    
