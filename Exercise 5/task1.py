import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform
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
    return np.sort(data, axis=0)


def approximate_linear_function(data, cond=None, order=1):
    A = np.vstack([data[:,0], 1*np.ones((order, len(data[:,0])))]).T
    m, residuals, rank, s = np.linalg.lstsq(A, data[:,1], rcond=cond)
    print("Sums of squared residuals: {}".format(residuals))
    return m


# def radial_basis(x,x_l,epsilon):
#     r = x_l-x
#     return _radial_basis(r, epsilon)


def _radial_basis(r, epsilon):
    return np.exp(-(r/epsilon)**2)

# def radial_function(x, c_l, x_l, epsilon):
#     answer = np.zeros(x.shape)
#     for l in range(c_l.shape[0]):
#         answer += c_l[l]*radial_basis(x,x_l[l], epsilon)
#     return answer


def find_epsilon(x):
    xmax = np.amax([x], axis=1)
    xmin = np.amin([x], axis=1)
    edges = xmax - xmin
    edges = edges[np.nonzero(edges)]
    return np.power(np.prod(edges)/x.shape[-1], 1.0/edges.size)


def approximate_radial_function(x, y, newx, epsilon=0.001, smooth=0.0, norm="euclidean"):
    """
    x: Vector with input x-values
    y: Vector with output y-values
    n: Number of nodes (randomly selected from xa)
    """
    nodes,x = calc_radial_nodes(x,y,epsilon, smooth, norm)
    # print(nodes)
    return interpolate_radial_function(newx,x,nodes,epsilon,norm)


def radial_basis(x_l, x, epsilon, norm="euclidean"):
    r = cdist(np.array([x_l]).T, np.array([x]).T, norm)
    return _radial_basis(r, epsilon)


def calc_radial_nodes(x, y, epsilon=0.001, smooth=0.0, norm="euclidean"):
    # rand = np.random.choice(len(x), size=100, replace=False)
    # x = x[rand]  # select random nodes
    # y = y[rand]  # select random nodes
    # # x[rand] = 0

    n = x.shape[-1]

    # xi = np.array([x])
    # r = pdist(xi.T, norm)
    # rads = _radial_basis(squareform(r), epsilon)
    rads = radial_basis(x,x.T,epsilon,norm)
    
    A = rads - np.eye(n)*smooth
    nodes = linalg.solve(A, y)
    # nodes = np.linalg.lstsq(A,y)
    
    return nodes, x

def interpolate_radial_function(newx, x, nodes, epsilon=0.001, norm="euclidean"):
    # Now calculate for new x-basis
    # rand = np.random.choice(len(nodes), size=(len(nodes)-100), replace=False)
    # x = x[rand]  # select random nodes
    # nodes = nodes[rand]  # select random nodes
    # # nodes[rand] = 0

    # r = cdist(np.array([newx]).T, np.array([x]).T, norm)
    # phi = _radial_basis(r, epsilon)
    phi = radial_basis(newx,x,epsilon,norm)
    return np.dot(phi, nodes)


def compare_linear_approximation(data, m):
    x = data[:, 0]
    y = data[:, 1]
    order = len(m)
    approx = 0
    for a in range(order):
        part = m[order-1-a]*x**a
        approx = approx+part
    compare_approxiation(x,y,x,approx)

def compare_approxiation(x, y, x_approx, y_approx):
    """
    Compare the input data with the fitted function
    """
    fig, ax = plt.subplots(2,1)
    ax[0].plot(x, y, "o", label="Original data", markersize=2)
    ax[0].plot(x_approx, y_approx, label="Fitted line")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$y$")
    ax[0].legend()
    if y.shape == y_approx.shape:
        ax[1].plot(x, y-y_approx, "o", label="Error", markersize=2)
        ax[1].plot(x_approx, y_approx-y_approx, label="Fitted line")
        ax[1].set_title("Error")
        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$\Delta y$")
    else:
        print("Cannot compare difference, x and x_approx need to be equal")
    plt.tight_layout()
    show_and_save("task1_linearfit")


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

if __name__ == "__main__":
    with InteractiveMode():
        linear_data = sort(load_data("linear_function_data.txt"))
        nonlinear_data = sort(load_data("nonlinear_function_data.txt"))

        # m1 = approximate_linear_function(linear_data)
        # compare_linear_approximation(linear_data, m1)

        # m2 = approximate_linear_function(nonlinear_data, order=1)
        # compare_linear_approximation(nonlinear_data, m2)

        # m2 = approximate_linear_function(nonlinear_data, order=3)
        # compare_linear_approximation(nonlinear_data, m2)

        x1 = np.linspace(
            np.min(nonlinear_data[:, 0]), np.max(nonlinear_data[:, 0]), 100).T
        x1 = nonlinear_data[:, 0]
        # print(find_epsilon(nonlinear_data[:,0]))
        y1 = approximate_radial_function(
            nonlinear_data[:, 0], nonlinear_data[:, 1], x1, epsilon=0.001, smooth=0.0)
        compare_approxiation(nonlinear_data[:, 0], nonlinear_data[:, 1], x1, y1)
        
        # from scipy.interpolate import Rbf
        # rbf = Rbf(nonlinear_data[:, 0], nonlinear_data[:, 1], function="gaussian",epsilon=0.001, smooth=0.0)
        # y1 = rbf(x1)
        # compare_approxiation(nonlinear_data[:, 0], nonlinear_data[:, 1], x1, y1)

        # y1 = approximate_radial_function(linear_data[:,0], linear_data[:,1],linear_data[:,0])
        # compare_approxiation(linear_data[:, 0], linear_data[:, 1], linear_data[:, 0], y1)
        
