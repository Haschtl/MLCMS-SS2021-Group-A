import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

from helper import InteractiveMode, show_and_save, load_data, sort


def finite_diff(x0,x1,dt):
    return (x1-x0)/dt


def plot_vectorfields(*args, labels=[], name="", title=""):
    plt.figure()
    for idx, arg in enumerate(args):
        plt.plot(arg[:, 0], arg[:, 1], "o", label=labels[idx], markersize=2)
    plt.title(title)
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    show_and_save("task2_{}".format(name))


def plot_phase_portrait(x,y,u,v, trajectory=None, title=""):
    plt.figure()
    ax =plt.subplot(1,1,1)
    ax.set_aspect('equal')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_title(title)
    ax.streamplot(x, y, u, v)
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1])


def step(A_alpha, X): # taken from ex3: task1.py
    """
    Calculates A*x; Shape(A_alpha): 2x2, Shape(X): 2xN
    """
    return np.array([A_alpha[0][0]*X[:,0]+A_alpha[0][0]*X[:,1], A_alpha[1][0]*X[:,0]+A_alpha[1][1]*X[:,1]]).T


def interp(x1, x2, v1, v2, x):  # taken from ex3: task1.py
    """
    2D Interpolation for velocity calculation 
    """
    v = np.nan
    f1 = interpolate.interp2d(x1, x2, v1)
    f2 = interpolate.interp2d(x1, x2, v2)
    return np.array([f1(x[0], x[1])[0], f2(x[0], x[1])[0]])


def create_trajectory(x1, x2, v1, v2, x0=None, delta=1, iterations=1000): # taken from ex3: task1.py
    """
    Calculates a trajectory in 2D space starting from x0 using Euler's method
    """
    if x0 is None:
        x0 = 0.00001*np.random.rand(2)-0.00005
    trajectory = [x0]
    np.random.rand(1)
    for _ in range(iterations):
        v = interp(x1, x2, v1, v2, trajectory[-1])
        # print(trajectory[-1], v)
        trajectory.append(trajectory[-1]+delta*v)
    return np.array(trajectory)



def part1(x0, x1, cond=None): 
    dt = 0.1
    # use finite-difference formula from 1.3
    # to estimate the vectors v^(k) at all points x_0^(k).
    # Choose timestep dt that will minimize the error in part 2.
    v = finite_diff(x0,x1,dt=dt)
    # Then approximate the 2x2 matrix A: v^(k)=Ax_0^(k)
    A, residuals, rank, s = np.linalg.lstsq(x0, v, rcond=cond)
    print("Sums of squared residuals: {}".format(residuals))
    return A


def part2(x0, x1, A):
    dt = 0.1
    # Use the estimate of A from part 1
    # Solve x_dot=A*x with all x_0^(k) as initial points until
    # t_end = 0.1
    # t = 0
    v = np.zeros(x0.shape)
    for idx, _x0 in enumerate(x0):
        v[idx] = A@_x0
    # v = step(A,x0)
    x0 += v*dt
    # v = A*x0
    # while t<t_end:
    #     for idx, _x0 in enumerate(x0):
    #         v[idx] = A@_x0
    #     x0 += v*dt
    #     t+=dt
    # The resulting points are estimates for x_1^(k).
    # Compute the mean squared error to x1
    plot_vectorfields(
        x1, x0, labels=["$x_1$", "$x_1$ (approx)"], name="ff", title="Approximated x1")

    # trajectory = None #create_trajectory(x0[:, 0], x0[:, 1], v[:, 0], v[:, 1], [10, 10])
    # plot_phase_portrait(x0[:, 0], x0[:, 1], v[:,0],v[:,1],trajectory=trajectory)
    mse = ((x0 - x1)**2).mean(axis=0)/len(x0)
    print("Mean squared error: {}".format(mse))


def part3(x0, x1):
    point = np.array([10, 10])  # far outside point
    # Again: Solve linear system with your matrix approximation for
    t_end = 100
    # Visualize the trajectory and phase portrait in [-10,10]^2


if __name__ == "__main__":
    linear_vectorfield_x0 = load_data("linear_vectorfield_data_x0.txt")
    linear_vectorfield_x1 = load_data("linear_vectorfield_data_x1.txt")
    with InteractiveMode():
        plot_vectorfields(
            linear_vectorfield_x0, linear_vectorfield_x1, labels=["$x_0$", "$x_1$"], name="inputdata", title="$x_0$ and $x_1$")
        A = part1(linear_vectorfield_x0, linear_vectorfield_x1)
        part2(linear_vectorfield_x0, linear_vectorfield_x1, A)
        part3(linear_vectorfield_x0, linear_vectorfield_x1)
