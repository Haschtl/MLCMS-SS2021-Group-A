import numpy as np
from matplotlib import pyplot as plt


def load_data(filepath):
    """
    Load the data from text-file to numpy array
    """
    data = np.genfromtxt(filepath, delimiter=' ')
    return data


def sort(data):
    """
    Sort data on x-axis (first axis)
    """
    return data[np.lexsort((data[:, 1], data[:, 0]))]


def show_and_save(name):
    """
    Show and save matplotlib figure. Made for interactive mode (will call plt.pause())0
    """
    plt.show()
    plt.pause(0.2)
    plt.savefig(name+".png", dpi=200)


class InteractiveMode():
    """ 
    Wrapper for matplotlib interactive mode.
    Use it like this:
    with InteractiveMode():
        plt.plot(x,y)
        plt.show()
        plt.figure()
        plt.plot(x,y)
        ...
    """

    def __enter__(self):
        plt.ion()

    def __exit__(self, exception_type, exception_value, traceback):
        plt.ioff()
        plt.show()
