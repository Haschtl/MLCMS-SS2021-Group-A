import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import Rbf
# from scipy.spatial.distance import cdist
# from scipy.linalg import solve
from sklearn.metrics import mean_squared_error

import task1
import task2
from helper import InteractiveMode, show_and_save, load_data, sort

def linear_vf(x0, x1, dt, A):
    # Solve x_dot=A*x with all x_0^(k) as initial points until
    t_end = 0.1
    x = task2.create_trajectory(x0,A,dt,t_end)
    # The resulting points are estimates for x_1^(k).

    plt.figure()
    plt.quiver(x0[:,0], x0[:,1], x[-1][:,0], x[-1][:,1])

    plt.title("Vector field generated with linear operator")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    show_and_save("task3_{}".format("linear_vf"))

    # Compute the mean squared error to x1
    mse = ((x[-1] - x1)**2).mean(axis=0)/len(x0)
    print("Mean squared error: {}".format(mse))

if __name__ == "__main__":
    nonlinear_vectorfield_x0 = load_data("nonlinear_vectorfield_data_x0.txt")
    nonlinear_vectorfield_x1 = load_data("nonlinear_vectorfield_data_x1.txt")

    # Part 1
    dt=0.01
    # task2.plot_vectorfields(
    #     nonlinear_vectorfield_x0, nonlinear_vectorfield_x1, labels=["$x_0$", "$x_1$"], name="inputdata", title="$x_0$ and $x_1$")
    A = task2.part1(nonlinear_vectorfield_x0, nonlinear_vectorfield_x1, dt)
    print(A)
    linear_vf(nonlinear_vectorfield_x0, nonlinear_vectorfield_x1, dt, A)

    # yi, error = task1.approximate_nonlinear_function(nonlinear_vectorfield_x0[:,0], nonlinear_vectorfield_x1[:,1], nonlinear_vectorfield_x1[:,0])
    # mse = ((yi - nonlinear_vectorfield_x1[:,1])**2).mean()
    # print(mse)
