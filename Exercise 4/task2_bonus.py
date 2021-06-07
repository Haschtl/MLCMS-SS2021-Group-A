import numpy as np
import matplotlib.pyplot as plt
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector
from task2_subtask2 import part2dataset, swissroll_color


def bonus_task(n=1000):
    X = part2dataset(n)
    X_color = swissroll_color(X)

    idx_plot = np.random.permutation(n)[0:n]
    # Optimize kernel parameters
    X_pcm = pfold.PCManifold(X)
    X_pcm.optimize_parameters()
    print(f"epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}")

    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon),
        n_eigenpairs=9,
        dist_kwargs=dict(cut_off=X_pcm.cut_off),
    )

    dmap = dmap.fit(X_pcm)
    evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_

    plot_pairwise_eigenvector(
        eigenvectors=dmap.eigenvectors_[idx_plot, :],
        n=1,
        fig_params=dict(figsize=[15, 15]),
        scatter_params=dict(cmap=plt.cm.Spectral, c=X_color[idx_plot]),
    )
    plt.show()
    # plt.savefig("task2_bonus_n{}.png".format(n), dpi=100)


if __name__ == "__main__":
    bonus_task(1000)
    bonus_task(5000)
