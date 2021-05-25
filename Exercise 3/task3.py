import numpy as np
import matplotlib.pyplot as plt
from task1 import arrowed_spines, create_trajectory


def xdot1(alpha,x1,x2):
    """
    Andronov-Hopf bifurcation x_1
    """
    return alpha*x1-x2-x1*(x1**2+x2**2)


def xdot2(alpha, x1, x2):
    """
    Andronov-Hopf bifurcation x_2
    """
    return x1+alpha*x2-x2*(x1**2+x2**2)


def step(alpha, X):
    """
    Andronov-Hopf bifurcation vector field normal form
    """
    return np.array([xdot1(alpha,X[0],X[1]), xdot2(alpha,X[0],X[1])])
    # return A_alpha.dot(X)


def step2_alpha1(alpha2, x):
    """
    cusp bifurcation == 0: 
    Returns the values for alpha1 given x and alpha2
    a_1+a_2*x-x^3 = 0 
    """
    return -np.multiply(alpha2,x)+x**3


def step2_alpha2(alpha1, x):
    """
    cusp bifurcation == 0: 
    Returns the values for alpha2 given x and alpha1
    a_1+a_2*x-x^3 = 0 
    """
    return np.multiply((x**3-alpha1), (1/x))


def plot(ax: plt.axis, x, y, u, v, trajectory, pause=0.2, title=""):
    """
    Plot function for subtask 1
    """
    ax.set_aspect('equal')
    arrowed_spines(ax)
    ax.set_xlim([min(x)*1,max(x)*1])
    ax.set_ylim([min(y)*1,max(y)*1])
    ax.set_title(title)
    ax.streamplot(x, y, u, v)
    ax.plot(trajectory[:, 0], trajectory[:, 1])


def subtask1(ax, alpha, x0s=[], range=[[-2, 2], [-2, 2]], resolution=[20, 20], title=""):
    """
    Subtask 1: plot Andronov-Hopf bifurcation
    """
    x1 = np.linspace(range[0][0], range[0][1], num=resolution[0])
    x2 = np.linspace(range[1][0], range[1][1], num=resolution[1])

    xv, yv = np.meshgrid(x1, x2)
    xy = np.stack((np.ravel(xv), np.ravel(yv)), axis=-1).T
    v = step(alpha, xy)
    v1 = v[0].reshape(resolution)
    v2 = v[1].reshape(resolution)

    plot(ax, x1, x2, v1, v2, trajectory=np.array(
        [[0, 0], [0, 0]]), title=title)
    for x0 in x0s:
        trajectory = create_trajectory(x1, x2, v1, v2, x0, delta=0.1)
        ax.plot(trajectory[:, 0], trajectory[:, 1])


def subtask2(ax, range=[-100, 100], resolution=100, title=""):
    """
    Subtask 2: plot cusp bifurcation 3d
    """
    x = np.linspace(-10, 10, num=resolution)
    alpha2 = np.linspace(range[0], range[1], num=resolution)

    xv, yv = np.meshgrid(x, alpha2)
    xalpha2 = np.stack((np.ravel(xv), np.ravel(yv)), axis=-1).T
    alpha1 = step2_alpha1(xalpha2[1], xalpha2[0])
    # alpha1[alpha1>resolution] = np.nan
    # alpha1[alpha1<-resolution] = np.nan
    ax.plot_surface(alpha1.reshape([resolution, resolution]), xalpha2[1].reshape([resolution, resolution]), xalpha2[0].reshape(
        [resolution, resolution]))
    ax.set_xlabel("$\\alpha_1$")
    ax.set_ylabel("$\\alpha_2$")
    ax.set_zlabel("$x$")


if __name__ == "__main__":
    alpha = 0.1
    matrices = {
        "$\\alpha=0$, focus, stable":  {"alpha": -1, "x0s": []},
        "$\\alpha=-1$, focus, stable":  {"alpha": 0, "x0s": []},
        "$\\alpha=1$, focus, neutrally stable": {"alpha": 1, "x0s": []},
        "$\\alpha=1$, exemplary trajectories": {"alpha": 1, "x0s": [np.array([2, 0]), np.array([0.5, 0])]},
    }
    if input("Show Animation? [y/N]: ").lower() == "y":
        plt.ion()
        fig, ax = plt.subplots(1)
        lower = -3
        upper = 3
        stepsize = 0.2
        alpha = lower
        while True:
            plt.cla()
            if alpha>upper or alpha<lower:
                stepsize = -stepsize
            alpha = alpha + stepsize
            m = "$\\alpha={}".format(alpha)
            subtask1(ax, alpha, title=m)
            plt.pause(0.1)

    elif input("Show 3 bifuractions of dynamical system (8) [y/N]: ") == "y":
        for m in matrices.keys():
            fig, ax = plt.subplots(1)
            subtask1(ax, matrices[m]["alpha"], x0s=matrices[m]["x0s"], title=m)
            fig.set_size_inches(8, 8)
            fig.savefig(m.replace(", ","_").replace("$\\","").replace("$","").lower()+'.png', dpi=100)
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        subtask2(ax, title="Cusp bifurcation")
        plt.show()
