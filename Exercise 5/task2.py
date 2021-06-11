import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

from helper import InteractiveMode, show_and_save, load_data, sort


def finite_diff(x0,x1,dt):
    return (x1-x0)/dt


def plot_vectorfields(*args, labels=[], name="", title=""):
    marker = ["o","x"]
    plt.figure()
    for idx, arg in enumerate(args):
        if idx<len(labels):
            plt.plot(arg[:, 0], arg[:, 1], marker[idx%len(marker)], label=labels[idx], markersize=3)
        else:
            plt.plot(arg[:, 0], arg[:, 1], marker[idx %
                     len(marker)], markersize=3)
    plt.title(title)
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    show_and_save("task2_{}".format(name))


def plot_phase_portrait(A, range=[[-10,10],[-10,10]], resolution=[100,100], trajectory=None, name="phaseportrait", title=""):
    x = np.linspace(range[0][0], range[0][1], num=resolution[0])
    y = np.linspace(range[1][0], range[1][1], num=resolution[1])
    xv, yv = np.meshgrid(x, y)
    xy = np.stack((np.ravel(xv), np.ravel(yv)), axis=-1)
    uv = xy@A
    u = uv[:, 0].reshape(resolution)
    v = uv[:, 1].reshape(resolution)
    plt.figure()
    ax =plt.subplot(1,1,1)
    ax.set_aspect('equal')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_title(title)
    ax.streamplot(x, y, u, v)
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1])
    show_and_save("task2_{}".format(name))


def interp(x1, x2, v1, v2, x):  # taken from ex3: task1.py
    """
    2D Interpolation for velocity calculation 
    """
    v = np.nan
    f1 = interpolate.interp2d(x1, x2, v1)
    f2 = interpolate.interp2d(x1, x2, v2)
    return np.array([f1(x[0], x[1])[0], f2(x[0], x[1])[0]])


def create_trajectory(x0,A, dt, t_end):
    t = 0
    x = [np.copy(x0)]
    while t < t_end:
        v = x[-1]@A
        xn = x[-1]+v*dt
        t += dt
        x.append(np.copy(xn))
    return np.array(x)


def part1(x0, x1, dt, cond=None): 
    # use finite-difference formula from 1.3
    # to estimate the vectors v^(k) at all points x_0^(k).
    # Choose timestep dt that will minimize the error in part 2.
    v = finite_diff(x0,x1,dt=dt)   # v= (x1-x0)/dt
    # Then approximate the 2x2 matrix A: v^(k)=Ax_0^(k)
    A, residuals, rank, s = np.linalg.lstsq(x0, v, rcond=cond)  # x0 @ A = v
    print("Sums of squared residuals: {}".format(residuals))
    return A


def part2(x0, x1, dt, A):
    # Use the estimate of A from part 1
    # Solve x_dot=A*x with all x_0^(k) as initial points until
    t_end = 0.1
    x = create_trajectory(x0,A,dt,t_end)
    # The resulting points are estimates for x_1^(k).
    plot_vectorfields(
        x1, *x[1:], labels=["$x_1$", "$x_1$ (approx)"], name="part2_approximation", title="Approximated x1")

    # Compute the mean squared error to x1
    mse = ((x[-1] - x1)**2).mean(axis=0)/len(x0)
    print("Mean squared error: {}".format(mse))


def part3(x0, dt, A):
    point = np.array([10.0, 10.0])  # far outside point
    # Again: Solve linear system with your matrix approximation for
    t_end = 100.0
    # Visualize the trajectory and phase portrait in [-10,10]^2
    trajectory = create_trajectory(point, A, dt, t_end)

    # create phase portrait
    plot_phase_portrait(A, [[-10, 10], [-10, 10]], [100, 100], trajectory=trajectory,
                        name="part3_phaseportrait", title="Phase portrait and example trajectory")


if __name__ == "__main__":
    linear_vectorfield_x0 = load_data("linear_vectorfield_data_x0.txt")
    linear_vectorfield_x1 = load_data("linear_vectorfield_data_x1.txt")
    dt=0.1
    with InteractiveMode():
        plot_vectorfields(
            linear_vectorfield_x0, linear_vectorfield_x1, labels=["$x_0$", "$x_1$"], name="inputdata", title="$x_0$ and $x_1$")
        A = part1(linear_vectorfield_x0, linear_vectorfield_x1, dt)
        part2(linear_vectorfield_x0, linear_vectorfield_x1, dt, A)
        part3(linear_vectorfield_x0, dt, A)
