import scipy.io as io
import numpy as np
from matplotlib import pyplot as plt
import scipy
from tqdm import tqdm

from data import get_image_paths, write_output


def process_images():
    img_paths = get_image_paths()
    for img_path in tqdm(img_paths):
        process_image(img_path)


def process_image(img_path):
    # Load sparse matrix
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace(
        'images', 'ground_truth').replace('IMG_', 'GT_IMG_'))

    # Read image
    img = plt.imread(img_path)

    # Create a zero matrix of image size
    k = np.zeros((img.shape[0], img.shape[1]))

    gt = mat["image_info"][0, 0][0, 0][0]

    # Generate hot encoded matrix of sparse matrix
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1

    # generate density map
    k = gaussian_filter_density(k)

    # File path to save density map
    write_output(k, img_path)


def gaussian_filter_density(gt):
    # Generates a density map using Gaussian filter transformation

    density = np.zeros(gt.shape, dtype=np.float32)

    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    # FInd out the K nearest neighbours using a KDTree

    pts = np.array(
        list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048

    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point

        # Convolve with the gaussian filter

        density += scipy.ndimage.filters.gaussian_filter(
            pt2d, sigma, mode='constant')

    return density


if __name__ == "__main__":
    process_images()
