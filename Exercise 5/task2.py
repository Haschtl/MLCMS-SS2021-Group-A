import numpy as np
from matplotlib import pyplot as plt

from helper import InteractiveMode, show_and_save, load_data, sort



def part1(x0,x1):
    pass

def part2(x0,x1):
    pass

def part3(x0,x1):
    pass

def extras(x0,x1):
    pass


if __name__ == "__main__":
    linear_vectorfield_x0 = load_data("linear_vectorfield_data_x0.txt")
    linear_vectorfield_x1 = load_data("linear_vectorfield_data_x1.txt")
    with InteractiveMode():
        part1(linear_vectorfield_x0,linear_vectorfield_x1)
        part2(linear_vectorfield_x0,linear_vectorfield_x1)
        part3(linear_vectorfield_x0,linear_vectorfield_x1)
        extras(linear_vectorfield_x0, linear_vectorfield_x1)
