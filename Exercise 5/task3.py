import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import Rbf
# from scipy.interpolate import RBFInterpolator
# from scipy.spatial.distance import cdist
# from scipy.linalg import solve
from sklearn.metrics import mean_squared_error

import task1
import task2
from helper import InteractiveMode, show_and_save, load_data, sort


def scatter_data(x, y, name="", labels=["x", "y"], title=""):
    """
    Scatter and save 2-D data.

    Parameters
    ----------
    x : (N, 1) array-like
        x coordinates of the data.
    y : (N, 1) array-like
        y coordinates of the data.
    name : string, optional
        Filename addition. Default: ""
    labels : list, optional
        Labels of axes. Default: ["x", "y"]
    title : string, optional
        Title of the plot. Default: ""
    """
    plt.figure()
    plt.scatter(x, y, label=title)
    plt.title(title)
    plt.legend()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    show_and_save("task3_{}".format(name))


def linear_vf(x0, x1, dt, A, t_end=0.1):
    # Solve x_dot=A*x with all x_0^(k) as initial points until
    x = task2.create_trajectory(x0,A,dt,t_end)
    # The resulting points are estimates for x_1^(k).

    plt.figure()
    plt.quiver(x0[:,0], x0[:,1], x[-1][:,0], x[-1][:,1])

    plt.title("Vector field generated with linear operator")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    show_and_save("task3_{}".format("linear_vf"))

    # Compute the mean squared error to x1
    mse = ((x[-1] - x1)**2).mean(axis=0)
    print("Mean squared error: {}".format(mse))

def nonlinear_vf(x0, x1, yi):
    plt.figure()
    plt.quiver(x0[:,0], x0[:,1], x1[:,0], yi)

    plt.title("Vector field generated with RBF")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    show_and_save("task3_{}".format("nonlinear_vf"))

    # Compute the mean squared error
    mse = ((np.array(list(zip(x1[:,0],yi))) - x1)**2).mean(axis=0)
    print("Mean squared error: {}".format(mse))

if __name__ == "__main__":
    # NOTE: This file is a mess mostly because of last day solution trials.
    nonlinear_vectorfield_x0 = load_data("nonlinear_vectorfield_data_x0.txt")
    nonlinear_vectorfield_x1 = load_data("nonlinear_vectorfield_data_x1.txt")

    # Part 1
    dt=0.01
    # task2.plot_vectorfields(
    #     nonlinear_vectorfield_x0, nonlinear_vectorfield_x1, labels=["$x_0$", "$x_1$"], name="inputdata", title="$x_0$ and $x_1$")
    A = task2.part1(nonlinear_vectorfield_x0, nonlinear_vectorfield_x1, dt)
    print(A)
    linear_vf(nonlinear_vectorfield_x0, nonlinear_vectorfield_x1, dt, A)
    
    diff_vector = nonlinear_vectorfield_x1 - nonlinear_vectorfield_x0
    # rbfi = Rbf(diff_vector[:,0], diff_vector[:,1], function='gaussian', epsilon=0.00001)
    rbfi = Rbf(nonlinear_vectorfield_x0[:,0], nonlinear_vectorfield_x0[:,1], function='gaussian', epsilon=0.00001)
    # yi = rbfi(nonlinear_vectorfield_x0)
    yi = rbfi(nonlinear_vectorfield_x1[:,0])
    # print(yi.shape)
    # mse = ((yi - nonlinear_vectorfield_x1)**2).mean(axis=0)
    # print(mse)

    # yi, error = task1.approximate_nonlinear_function(nonlinear_vectorfield_x0[:,0], nonlinear_vectorfield_x0[:,1], nonlinear_vectorfield_x1[:,0], epsilon=0.00001)
    #mse = ((yi - nonlinear_vectorfield_x1[:,1])**2).mean()
    # mse = ((yi - nonlinear_vectorfield_x1[:,1])**2).mean(axis=0)
    # print(mse)

    nonlinear_vf(nonlinear_vectorfield_x0, nonlinear_vectorfield_x1, yi)
    # nonlinear_vf(nonlinear_vectorfield_x0, yi, yi[:,1])
    
    # Solve for a larger time using linear method.
    solutions = []
    for i in range(10):
        solution = task2.create_trajectory(nonlinear_vectorfield_x0,A,dt,i)
        scatter_data(solution[:,0], solution[:,1], name="solution{}".format(i), title="Solution for t={}".format(i))
    
