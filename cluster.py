import os
import sys
import argparse
import numpy as np
import kaldiio
import itertools
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--out-rttm-dir', type=str, required=True)
parser.add_argument('--ark-file', type=str, required=True)
parser.add_argument('--segments-file', type=str, required=True)
parser.add_argument('--threshold', type=float, required=True)
parser.add_argument('--spk-num', type=int, default=None)

args = parser.parse_args()

def read_xvector_timing_dict(kaldi_segments):
    segs = np.loadtxt(kaldi_segments, dtype=object)
    split_by_filename = np.nonzero(segs[1:, 1] != segs[:-1, 1])[0] + 1
    return {s[0, 1]: (s[:, 0], s[:, 2:].astype(float)) for s in np.split(segs, split_by_filename)}


def merge_adjacent_labels(starts, ends, labels):
    # Merge neighbouring (or overlaping) segments with the same label
    adjacent_or_overlap = np.logical_or(np.isclose(ends[:-1], starts[1:]), ends[:-1] > starts[1:])
    to_split = np.nonzero(np.logical_or(~adjacent_or_overlap, labels[1:] != labels[:-1]))[0]
    starts = starts[np.r_[0, to_split+1]]
    ends = ends[np.r_[to_split, -1]]
    labels = labels[np.r_[0, to_split+1]]

    # Fix starts and ends times for overlapping segments
    overlaping = np.nonzero(starts[1:] < ends[:-1])[0]
    ends[overlaping] = starts[overlaping+1] = (ends[overlaping] + starts[overlaping+1]) / 2.0
    return starts, ends, labels

def write_output(fp, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(f'SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} '
                 f'<NA> <NA> {label + 1} <NA> <NA>{os.linesep}')

def compute_affinity_matrix(X):
    """Compute the affinity matrix from data.
    Note that the range of affinity is [0,1].
    Args:
        X: numpy array of shape (n_samples, n_features)
    Returns:
        affinity: numpy array of shape (n_samples, n_samples)
    """
    # Normalize the data.
    l2_norms = np.linalg.norm(X, axis=1)
    X_normalized = X / l2_norms[:, None]
    # Compute cosine similarities. Range is [-1,1].
    cosine_similarities = np.matmul(X_normalized, np.transpose(X_normalized))
    # Compute the affinity. Range is [0,1].
    # Note that this step is not mentioned in the paper!
    affinity = (cosine_similarities + 1.0) / 2.0
    return affinity

def sim_enhancement(matrix):
    # Symmeterization Y_{i, j} = max(S_{ij}, S_{ji})
    matrix = np.maximum(matrix, matrix.T)
    # Diffusion Y = Y Y^T
    matrix = np.dot(matrix, matrix.T)
    # Row-wise max normalization S_{ij} = Y_{ij} / max_k(Y_{ik})
    maxes = np.max(matrix, axis=1)
    for i in range(matrix.shape[0]):
        matrix[i] = matrix[i] / maxes[i]
    return matrix


def SpectralCluster(affinity, num_spks=None, threshold=1e-2):
    S = sim_enhancement(affinity)
    np.fill_diagonal(S, 0.)
    L_norm = laplacian(S, normed=True)
    eigvals, eigvecs = np.linalg.eig(L_norm)
    if num_spks is None:
        kmask = np.real(eigvals) < threshold
        #import pdb;pdb.set_trace()
        #print(eigvals[kmask])
        P = np.real(eigvecs)[:, kmask]
    else:
        index_array = np.argsort(eigvals)
        P = np.real(eigvecs[:, index_array])[:, :num_spks]
    km = KMeans(n_clusters=P.shape[1])
    return km.fit_predict(P)

if __name__ == "__main__":

    segs_dict = read_xvector_timing_dict(args.segments_file)
    arkit = kaldiio.load_ark(args.ark_file)
    recit = itertools.groupby(arkit, lambda e: e[0].rsplit('-', 3)[0])
    for file_name, segs in recit:
        print(file_name)
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)

        norm = np.linalg.norm(x, axis=1)
        x = x / norm[:, None]
        scr_mx = compute_affinity_matrix(x)
        labels = SpectralCluster(scr_mx, num_spks=args.spk_num, threshold=args.threshold)

        assert(np.all(segs_dict[file_name][0] == np.array(seg_names)))
        start, end = segs_dict[file_name][1].T

        starts, ends, out_labels = merge_adjacent_labels(start, end, labels)
        os.makedirs(args.out_rttm_dir, exist_ok=True)
        with open(os.path.join(args.out_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
            write_output(fp, out_labels, starts, ends)
