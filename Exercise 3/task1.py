import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True)

def arrowed_spines(ax=None, arrowLength=10, labels=('$x_1$', '$x_2$'), arrowStyle='<|-'):
    xlabel, ylabel = labels

    for i, spine in enumerate(['left', 'bottom']):
        # Set up the annotation parameters
        t = ax.spines[spine].get_transform()
        xy, xycoords = [1, 0], ('axes fraction', t)
        xytext, textcoords = [arrowLength, 0], ('offset points', t)

        # create arrowprops
        arrowprops = dict( arrowstyle=arrowStyle,
                           facecolor=ax.spines[spine].get_facecolor(), 
                           linewidth=ax.spines[spine].get_linewidth(),
                           alpha = ax.spines[spine].get_alpha(),
                           zorder=ax.spines[spine].get_zorder(),
                           linestyle = ax.spines[spine].get_linestyle() )

        if spine == 'bottom':
            ha, va = 'left', 'center'
            xarrow = ax.annotate(xlabel, xy, xycoords=xycoords, xytext=xytext, 
                        textcoords=textcoords, ha=ha, va='center',
                        arrowprops=arrowprops)
        else:
            ha, va = 'center', 'bottom'
            yarrow = ax.annotate(ylabel, xy[::-1], xycoords=xycoords[::-1], 
                        xytext=xytext[::-1], textcoords=textcoords[::-1], 
                        ha='center', va=va, arrowprops=arrowprops)
    return xarrow, yarrow

def A_alpha(alpha:float):
    return np.array([[alpha,alpha],[-0.25,0]])

def step(A_alpha, X):
    return np.array([A_alpha[0][0]*X[0]+A_alpha[0][0]*X[1], A_alpha[1][0]*X[0]+A_alpha[1][1]*X[1]])
    # return A_alpha.dot(X)


def plot(ax: plt.axes, x, y, u, v, eigenvalues, trajectory, pause=0.2, title=""):
    ax[0].spines['left'].set_color('none')
    ax[0].spines['bottom'].set_color('none')
    ax[0].spines['right'].set_color('none')
    ax[0].spines['top'].set_color('none')
    ax[0].xaxis.set_ticks_position('bottom')
    ax[0].yaxis.set_ticks_position('left')
    ax[1].set_aspect('equal')

    arrowed_spines(ax[1])
    ax[1].set_xlim([min(x)*1,max(x)*1])
    ax[1].set_ylim([min(y)*1,max(y)*1])
    ax[0].grid(True, which='both')
    ax[0].axhline(y=0, color='k')
    ax[0].axvline(x=0, color='k')
    ax[0].set_title("Eigenvalues")
    ax[1].set_title(title)

    ax[0].scatter(eigenvalues[0].real, eigenvalues[0].imag)
    ax[0].scatter(eigenvalues[1].real, eigenvalues[1].imag)
    # ax[1].quiver(x,y,u,v)
    ax[1].streamplot(x, y, u, v)

    ax[1].plot(trajectory[:, 0], trajectory[:, 1])

def interp(x1,x2,v1,v2,x):
    v = np.nan
    f1 = interpolate.interp2d(x1,x2,v1)
    f2 = interpolate.interp2d(x1,x2,v2)
    return np.array([f1(x[0], x[1])[0], f2(x[0], x[1])[0]])


def create_trajectory(x1, x2, v1, v2, x0=None, delta=1, iterations=1000):
    if x0 is None:
        x0 = 0.00001*np.random.rand(2)-0.00005
    trajectory = [x0]
    np.random.rand(1)
    for _ in range(iterations):
        v = interp(x1, x2, v1, v2, trajectory[-1])
        # print(trajectory[-1], v)
        trajectory.append(trajectory[-1]+delta*v)
    return np.array(trajectory)

def main(ax, A_alpha, x0, range=[[-1,1],[-1,1]], resolution=[10,10], title=""):
    eig, v = np.linalg.eig(A_alpha)
    print("Eigenvalues of A_alpha: {}".format(eig))
    x1 = np.linspace(range[0][0], range[0][1], num=resolution[0])
    x2 = np.linspace(range[1][0], range[1][1], num=resolution[1])

    xv, yv = np.meshgrid(x1, x2)
    xy = np.stack((np.ravel(xv), np.ravel(yv)), axis=-1).T
    v = step(A_alpha, xy)

    v1 = v[0].reshape(resolution)
    v2 = v[1].reshape(resolution)
    trajectory = create_trajectory(x1, x2, v1, v2, x0)
    # print(trajectory)
    # print(x.shape, y.shape, xy.shape, z.shape)
    plot(ax, x1, x2, v1, v2, eigenvalues=eig, trajectory=trajectory, title=title)


if __name__ == "__main__":
    alpha = 0.1
    matrices = {
        "Node, stable":     {"A": np.array([[-alpha, 0], [0, -2*alpha]]), "x0":np.array([4,4])},
        "Focus, stable":    {"A": np.array([[-alpha, -alpha], [alpha, -alpha]]), "x0": np.array([4, 4])},
        "Saddle, unstable": {"A": np.array([[alpha, 3*alpha], [alpha, -alpha]]), "x0": np.array([0.3, -1])},
        "Node, unstable":   {"A": np.array([[alpha, 0], [0, 2*alpha]]), "x0":np.array([-0.1,0.1])},
        "Focus, unstable":  {"A": np.array([[alpha, alpha], [-alpha, alpha]]), "x0":None},
    }
    if False:
        plt.ion()
        fig, ax = plt.subplots(1, 2)
        m = "Node, stable"
        r = 20
        for i in range(r):
            plt.cla()
            p = i/r*2-1
            a = -p+0.4
            b = 3*(p+0.5)*(p-0.2)
            c = (p+0.5)*(p-0.2)
            d = -2*(p+0.2)
            main(ax, np.array([[a,b],[c,d]]), x0=np.array([1,1]), title=m)
            plt.pause(0.2)
    else:
        for m in matrices.keys():
            fig, ax = plt.subplots(1, 2)
            main(ax, matrices[m]["A"], x0 = matrices[m]["x0"], title=m)
            fig.set_size_inches(8, 3)
            fig.savefig(m.replace(", ","_").lower()+'.png', dpi=100)
            # plt.show()
        plt.show()
