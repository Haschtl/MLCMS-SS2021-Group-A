import numpy as np
import matplotlib.pyplot as plt

from helper import InteractiveMode, show_and_save, load_data


def lorenz_step(X,sigma=10, beta=8/3, phi=28):
    """
    Iteration step function for Lorenz Attractor.

    Parameters
    ----------
    X : (N, 3) ndarray
        Points up until current iteration.
    sigma : float, optional
        Sigma parameter of the Lorenz Attractor. Default: 10
    beta : float, optional
        Beta parameter of the Lorenz Attractor. Default: 8/3
    phi : float, optional
        Phi parameter of the Lorenz Attractor. Default: 28
    
    Returns
    -------
    (1, 3) ndarray
        Point to add in current iteration.
    """
    x = sigma*(X[1]-X[0])
    y = X[0]*(phi-X[2])-X[1]
    z = X[0] * X[1] - beta*X[2]
    return np.array([x,y,z])

def lorenz(x_0=np.array([10,10,10]), sigma=10,beta=8/3, phi=28, T_end=1000, T_step=0.01):
    """
    3D Lorenz Attractor function.

    Parameters
    ----------
    x_0 : (1, 3) ndarray, optional
        Starting point of Lorenz Attractor. Default: 
        np.array([10,10,10])
    sigma : float, optional
        Sigma parameter of the Lorenz Attractor. Default: 10
    beta : float, optional
        Beta parameter of the Lorenz Attractor. Default: 8/3
    phi : float, optional
        Phi parameter of the Lorenz Attractor. Default: 28
    T_end : int, optional
        Ending point of the iteration. Default: 1000
    T_step : float, optional
        Step size. Default: 0.01
    
    Returns
    -------
    X : (N, 3) ndarray
        Points of Lorenz Attractor in 3-D space with given(or default) 
        parameters.
    """
    X = [x_0]
    for _t in range(int(T_end/T_step)):
        dX = lorenz_step(X[-1],sigma,beta,phi)
        X.append(X[-1]+T_step*dX)
    X = np.array(X)
    return X

def plot_data(x, y, name="", labels=["x", "y"], title=""):
    """
    Plot and save 2-D data.

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
    plt.plot(x, y, label=title)
    plt.title(title)
    plt.legend()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    show_and_save("task4_{}".format(name))

def plot_3d_data(x, y, z, name="", labels=["x", "y", "z"], title=""):
    """
    Plot and save 3-D data.

    Parameters
    ----------
    x : (N, 1) array-like
        x coordinates of the data.
    y : (N, 1) array-like
        y coordinates of the data.
    z : (N, 1) array-like
        z coordinates of the data.
    name : string, optional
        Filename addition. Default: ""
    labels : list, optional
        Labels of axes. Default: ["x", "y", "z"]
    title : string, optional
        Title of the plot. Default: ""
    """
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the values
    ax.plot(x, y, z, linewidth=0.4)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    show_and_save("task4_{}".format(name))

def part1(dt=10):
    """
    Create a time-delay embedding of the given 2-D data set.

    Parameters
    ----------
    dt : int, optional
        Time-delay amount. Default: 10
    """
    data = load_data("takens_1.txt")
    x_t = data[:,0]
    y_t = data[:,1]
    # Plot original data
    plot_data(x_t, y_t, name="data", labels=["x", "y"], title="Data set")
    # Plot x values against time
    t = np.arange(x_t.shape[0])
    plot_data(t, x_t, name="x(t)-t", labels=["t", "x(t)"], title="x(t) - t")
    # Shift x(t) to create the time-delayed version of it, x(t+dt). Then plot x(t) against x(t+dt)
    x_t_dt = np.roll(x_t, dt)
    plot_data(x_t, x_t_dt, name="x(t)-x(t+{})".format(dt), labels=["x(t)", "x(t+{})".format(dt)], title="x(t) - x(t+{})".format(dt))
    # Do the same in 3-D using x(t), x(t+dt) and x(t+2dt).
    # x_t_2dt = np.roll(x_t, 2*dt)
    # plot_3d_data(x_t, x_t_dt, x_t_2dt, name="x_t_x_t_dt_x_t_2dt", labels=["x(t)", "x(t+1)", "x(t+2)"], title="x(t) - x(t+1)")

def part2(dt=8):
    """
    Create a time-delay embedding of the Lorenz Attractor.

    Parameters
    ----------
    dt : int, optional
        Time-delay amount. Default: 8
    """
    # Generate and plot Lorenz coordinates with starting point [10,10,10] and parameters sigma=10, beta=8/3, phi=28
    lorenz_coords = lorenz(x_0=np.array([10,10,10]))
    x_t = lorenz_coords[:,0]
    y_t = lorenz_coords[:,1]
    z_t = lorenz_coords[:,2]
    plot_3d_data(x_t, y_t, z_t, name="lorenz", labels=["x", "y", "z"], title="Lorenz attractor")

    # Shift x(t) two times to create time-delays x(t+dt) and x(t+2dt). Then plot them in a 3D graph.
    x_t_dt = np.roll(x_t, dt)
    x_t_2dt = np.roll(x_t, 2*dt)
    plot_3d_data(x_t, x_t_dt, x_t_2dt, name="lorenz_embedding", labels=["x(t)", "x(t+{})".format(dt), "x(t+{})".format(2*dt)], title="Lorenz Embedding")

    # Do the same for z(t)
    z_t_dt = np.roll(z_t, dt)
    z_t_2dt = np.roll(z_t, 2*dt)
    plot_3d_data(z_t, z_t_dt, z_t_2dt, name="lorenz_embedding_z", labels=["z(t)", "z(t+{})".format(dt), "z(t+{})".format(2*dt)], title="Lorenz Embedding")


if __name__ == "__main__":
    with InteractiveMode():
        part1(dt=10)
        part2(dt=8)
