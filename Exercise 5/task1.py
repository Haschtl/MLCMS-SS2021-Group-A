import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.spatial.distance import cdist

from helper import InteractiveMode, show_and_save, load_data, sort


def approximate_linear_function(data, cond=None, order=1):
    """
    Simple leastsquare approximation for approximation of (nearly) linear functions
    data: x,y data: [x,y]   shape: (N * 2)
    cond: Cut-off ratio for small singular values of a. For the purposes of rank determination, singular values are treated as zero if they are smaller than rcond times the largest singular value of a
    """
    assert order==1 # order should approximate functions of higher order, but this fails and is not part of the exercise
    A = np.vstack([data[:,0], 1*np.ones((order, len(data[:,0])))]).T
    m, residuals, rank, s = np.linalg.lstsq(A, data[:,1], rcond=cond)
    print("Sums of squared residuals: {}".format(residuals))
    return m


def approximate_nonlinear_function(x, y, xi, epsilon=1, smooth=0.0, norm="euclidean"):
    """
    x: Vector with input x-values
    y: Vector with output y-values
    xi: New x-coordinates for inter/extrapolation
    epsilon: parameter for radial basis function
    """
    # if not epsilon:
    #     epsilon = find_epsilon(x)
    #     print("Calculated epsilon: {}".format(epsilon))
    c = calc_radial_coefficients(x, y, epsilon, smooth, norm)
    yi = interpolate_radial_function(xi, x, c, epsilon, norm)
    error = y-interpolate_radial_function(x, x, c, epsilon, norm)
    return yi, error


def calc_radial_coefficients(x, y, epsilon=1, smooth=0.0, norm="euclidean"):
    """
    Approximate the coefficients C
    """
    phi_x = radial_basis(x,x,epsilon,norm)
    phi_x = phi_x - np.eye(x.shape[-1])*smooth
    c = linalg.solve(phi_x, y)
    return c

def interpolate_radial_function(xi, x, c, epsilon=1, norm="euclidean"):
    """
    Calculate yi for xi
    """
    phi = radial_basis(xi, x, epsilon, norm)
    return np.dot(phi, c)


def radial_basis(x_l, x, epsilon, norm="euclidean"):
    """
    Calculates the radial basis for vector x
    """
    r = cdist(np.array([x_l]).T, np.array([x]).T, norm)
    return _radial_basis(r, epsilon)


def _radial_basis(r, epsilon):
    # return np.sqrt((r/epsilon)**2+1) # multiquadric
    return np.exp(-(r/epsilon)**2) # gaussian
    # return r # linear (similar to taylor decomposition)
    # return r**3 # cubic
    # return r**5 # quintic


# def find_epsilon(x):
#     xmax = np.amax([x], axis=1)
#     xmin = np.amin([x], axis=1)
#     xrange = xmax - xmin
#     xrange = xrange[np.nonzero(xrange)]
#     return np.power(np.prod(xrange)/x.shape[-1], 1.0/xrange.size)


def compare_linear_approximation(data, m, name="1", title=""):
    """
    Create a plot to present results of linear approximation 
    """
    x = data[:, 0]
    y = data[:, 1]
    order = len(m)
    approx = 0
    for a in range(order):
        part = m[order-1-a]*x**a
        approx = approx+part
    compare_approxiation(x,y,x,approx, None, name, title)

def compare_approxiation(x, y, x_approx, y_approx, error=None, name="1", title=""):
    """
    Compare the input data with the fitted function
    """
    _, ax = plt.subplots(2,1)
    ax[0].plot(x, y, "o", label="Original data", markersize=2)
    ax[0].plot(x_approx, y_approx, label="Fitted line")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$y$")
    ax[0].legend()
    ax[0].set_title(title)
    if y.shape == y_approx.shape:
        error = y-y_approx
    if error is not None:
        ax[1].plot(x, error, "o", label="Error", markersize=2)
        ax[1].plot(x_approx, np.zeros(y_approx.shape), label="Fitted line")
        ax[1].set_title("Error")
        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$\Delta y$")
    else:
        print("Cannot compare difference, x and x_approx need to be equal")
    plt.tight_layout()
    show_and_save("task1_approx_{}".format(name))



def part1():
    linear_data = sort(load_data("linear_function_data.txt"))
    m1 = approximate_linear_function(linear_data)
    compare_linear_approximation(
        linear_data, m1, name="part1_linear", title="Linear data, Linear approximation")


def part2():
    nonlinear_data = sort(load_data("nonlinear_function_data.txt"))
    m1 = approximate_linear_function(nonlinear_data)
    compare_linear_approximation(nonlinear_data, m1, name="part2_nonlinear", title="Nonlinear data, Linear approximation")


def part3():
    nonlinear_data = sort(load_data("nonlinear_function_data.txt"))
    xi = np.linspace(
        np.min(nonlinear_data[:, 0])*1.0, np.max(nonlinear_data[:, 0])*1.0, 2000)
    for epsilon in [0.005, 0.05, 0.5, 1, 2, 3, 5, 10]:
        yi, error = approximate_nonlinear_function(
            nonlinear_data[:, 0], nonlinear_data[:, 1], xi, epsilon=epsilon)
        compare_approxiation(
            nonlinear_data[:, 0], nonlinear_data[:, 1], xi, yi, error, name="part3_nonlinear_epsilon{}".format(epsilon), title="Nonlinear data, $\epsilon ={}$".format(epsilon))

    # cheat = True
    # if cheat:
    #     from scipy.interpolate import Rbf
    #     rbf = Rbf(nonlinear_data[:, 0], nonlinear_data[:, 1],
    #               function="gaussian", epsilon=epsilon)
    #     y1 = rbf(xi)
    #     compare_approxiation(
    #         nonlinear_data[:, 0], nonlinear_data[:, 1], xi, y1, "Test using scipy")


def extras():
    linear_data = sort(load_data("linear_function_data.txt"))
    xi = np.linspace(
        np.min(linear_data[:, 0])*1.0, np.max(linear_data[:, 0])*1.0, 2000)
    for epsilon in [0.005, 0.05, 0.5, 1, 2, 3, 5, 10]:
        yi, error = approximate_nonlinear_function(
            linear_data[:, 0], linear_data[:, 1], xi, epsilon=epsilon)
        compare_approxiation(
            linear_data[:, 0], linear_data[:, 1], xi, yi, error, name="extra_linear_epsilon{}".format(epsilon), title="Linear data, $\epsilon ={}$".format(epsilon))


if __name__ == "__main__":
    with InteractiveMode():
        part1()
        part2()
        part3()
        extras()
        
# Questions:
# Why is it not a good idea to use the radial basis function for dataset A?
# - A is very linear
# - Extrapolation is wrong
# - One unnecessary parameter, which can destroy the approximation
# 
# Why chose epsilon=3?:
# - Higher epsilon: Unstable behaviour
# - Lower epsilon: more and more ausreisser in approximation
