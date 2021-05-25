import numpy as np
import matplotlib.pyplot as plt
from task1 import arrowed_spines, create_trajectory


def lorenz_step(X,sigma=10,beta=8/3,phi=28):
    x = sigma*(X[1]-X[0])
    y = X[0]*(phi-X[2])-X[1]
    z = X[0] * X[1] - beta*X[2]
    return np.array([x,y,z])


def lorenz_diagram(ax, x_0=np.array([10,10,10]), sigma=10,beta=8/3, phi=28, T_end=1000, T_step=0.01):
    X = [x_0]
    plt.ion()
    for _t in range(int(T_end/T_step)):
        t = _t/T_step
        dX = lorenz_step(X[-1],sigma,beta,phi)
        # print(dX,T_step,X[-1])
        X.append(X[-1]+T_step*dX)
    X = np.array(X)
    ax.plot(X[:, 0], X[:, 1], X[:, 2], linewidth=0.4)
    plt.show()


def lorenz_traj_compare(ax, x_0=np.array([10, 10, 10]), x_1=np.array([10+10**-8,10,10]), sigma=10, beta=8/3, phi=28, T_end=5000, T_step=0.01):
    X_0 = [x_0]
    X_1 = [x_1]
    max_diff = 1
    plt.ion()
    for _t in range(int(T_end/T_step)):
        t = _t*T_step
        dx_0 = lorenz_step(X_0[-1], sigma, beta, phi)
        dx_1 = lorenz_step(X_1[-1], sigma, beta, phi)
        # print(dX,T_step,X[-1])
        X_0.append(X_0[-1]+T_step*dx_0)
        X_1.append(X_1[-1]+T_step*dx_1)
        if np.linalg.norm(X_0[-1]-X_1[-1])>max_diff:
            print("Difference between trajectories >{} at T={} with t_step={}".format(max_diff,t,T_step))
            break
        if any([np.isnan(x) for x in X_0[-1]]):
            print("NaN occurred in x_0")
            break
        if any([np.isnan(x) for x in X_1[-1]]):
            print("NaN occurred in x_1")
            break
    if t == int(T_end/T_step)*T_step:
        print("Difference was never >{} until T={}".format(max_diff,T_end))
    X_0 = np.array(X_0)
    X_1 = np.array(X_1)
    ax.plot(X_0[:, 0], X_0[:, 1], X_0[:, 2])
    ax.plot(X_1[:, 0], X_1[:, 1], X_1[:, 2])
    plt.show()
    
def step(x_n, r):
    """
    discrete logistic map step
    """
    # return np.multiply(np.multiply(r, x_n), (1-x_n))
    return r*x_n*(1-x_n)

def dx(x_n, r):
    return np.multiply(2*r, x_n)-r

def dr(x_n, r):
    return np.multiply(x_n, (1-x_n))

def iterate(x_0, r, iter=20):
    """
    Iterate logistic map
    """
    values = [x_0]
    for _ in range(iter):
        values.append(step(values[-1],r))
    return np.array(values)


def plot(ax: plt.axis, R, data, title=""):
    """
    Plot function for subtask 1
    """
    ax.set_aspect('equal')
    arrowed_spines(ax)
    ax.set_title(title)
    # ax.set_xlim([min(x)*1, max(x)*1])
    # ax.set_ylim([min(y)*1, max(y)*1])
    # ax.streamplot(x, y, u, v)
    # ax.plot(trajectory[:, 0], trajectory[:, 1])


def subtask(ax, x_0=np.linspace(-4,4), range=[-5, 5], resolution=50, title=""):
    """
    Subtask 1: Vary r from 0 to 2
    """
    plt.ion()
    R = np.linspace(range[0], range[1], num=resolution)
    data = []
    u = []
    v = []
    # ax.set_xlim([-5, 5])
    for r in R:
        plt.cla()
        line = iterate(x_0, r)
        plt.plot(x_0, line.T)
        ax.set_ylim([-5, 5])
        ax.set_title("r={}".format(r))
        plt.pause(0.2)
        data.append(step(x_0,r))
        u.append(dx(x_0,r))
        v.append(dr(x_0,r))
    # plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot()
    data = np.array(data)
    u = np.array(u)
    v = np.array(v)
    
    # x_0, R = np.meshgrid(x_0, R)
    R, x_0 = np.meshgrid(R, x_0)
    
    print(x_0.shape, R.shape, data.shape, u.shape, v.shape)
    ax.streamplot(R, x_0, u, v)
    ax.set_ylabel("$x_0$")
    ax.set_xlabel("$r$")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_0, R, data)
    # plot(ax, R, data, title)


def bifurcation_diagram(ax, x=1e-5 * np.ones(10000), r=np.linspace(0, 4.0, 10000), iter=100000):
    plt.ion()
    ax.set_ylabel("$x$")
    ax.set_xlabel("$r$")
    for i in range(iter):
        x = step(x, r)
        if i >= (iter - 100):
            ax.plot(r, x, color=(0, (iter-i) / 100, 0, 1),
                    marker=',', linewidth=0)
            plt.pause(0.2)
            plt.show()
    # plt.ioff()

    ax.set_xlim(min(r), max(r))
    ax.set_title("Bifurcation diagram")
    plt.show()


if __name__ == "__main__":
    for ri in range(40):
        r = ri/10
        x = 0.000001
        for i in range(1000):
            x = step(x,r)
        print("R={}, x --> {}".format(r,x))
    if input("Make some tests on logistic map? [y/N]: ").lower() == "y":
        fig, ax = plt.subplots(1)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        subtask(ax,np.linspace(0,1),[0,4])
        fig, ax = plt.subplots(1)
        subtask(ax)
        plt.show()
    if input("Logistic map: Vary r from 0 to 2? [y/N]: ").lower() == "y":
        fig, ax = plt.subplots(1)
        bifurcation_diagram(ax,x=0.1*np.ones(5000), r=np.linspace(0,2,5000))
    if input("Logistic map: Vary r from 2 to 4? [y/N]: ").lower() == "y":
        fig, ax = plt.subplots(1)
        bifurcation_diagram(ax, x=0.1*np.ones(5000), r=np.linspace(2, 4, 5000))
    if input("Logistic map: Whole bifurcation diagram? [y/N]: ").lower() == "y":
        fig, ax = plt.subplots(1)
        bifurcation_diagram(ax, x=0.1*np.ones(5000), r=np.linspace(0, 4, 5000))
    if input("Lorenz Attractor: Trajectory starting from [10,10,10] ? [y/N]").lower() == "y":
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        lorenz_diagram(ax, x_0=np.array([10,10,10]))
    if input("Lorenz Attractor: Trajectory starting from [10+10^-8,10,10] ? [y/N]").lower() == "y":
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        lorenz_diagram(ax, x_0=np.array([10+10 ** -8, 10, 10]))
    if input("Lorenz Attractor: Difference of [10,10,10] and [10+10^-8,10,10] ? [y/N]").lower() == "y":
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        lorenz_traj_compare(ax, x_0=np.array([10, 10, 10]), x_1=np.array([10+10**-8, 10, 10]))
    if input("Lorenz Attractor: Difference of [10,10,10] and [10+10**-8,10,10] with phi=0.5? [y/N]").lower() == "y":
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        lorenz_traj_compare(ax, x_0=np.array(
            [10, 10, 10]), x_1=np.array([10+10**-8, 10, 10]), phi=0.5)
    plt.ioff()
    plt.show()
