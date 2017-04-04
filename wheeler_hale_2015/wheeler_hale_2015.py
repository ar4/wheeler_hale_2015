"""Implementation of Wheeler and Hale's well log aligning using dynamic
warping.
"""
import numpy as np
import pandas
import lasio
import fastdtw
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix


def load_logs(log_files):
    """Load the LAS log files into a list of pandas dataframes.

    Args:
        log_files: A list of paths to log files.

    Returns: A list of pandas dataframes, one for each log. This requires
        that pandas is installed.
    """
    logs = []
    for log_file in log_files:
        l = lasio.read(log_file)
        logs.append(l.df())
    return logs


def prepare_logs(logs, normalize=True, fillna=True):
    """Prepare the input log files.

    Args:
        logs: A list of pandas dataframes, one for each log.
        normalize: An optional boolean indicating whether to normalize the
            data so that each type of measurement in each log has zero
            median, and has an interquartile range of 1. Default True.
        fillna: An optional boolean indicating whether to fill missing data
            with 0. Default True.
    """
    if normalize:
        _normalize(logs)
    if fillna:
        _fillna(logs)


def _normalize(logs):
    """Shift and scale each measurement of each log to have zero median
    and an interquartile range (range between the first and third quartiles)
    of 1.
    """
    for l in logs:
        for col in l.columns:
            cq1 = np.nanpercentile(l[col].values, 25)
            cq2 = np.nanmedian(l[col].values)
            cq3 = np.nanpercentile(l[col].values, 75)
            l[col].values[:] = (l[col].values[:] - cq2) / (cq3 - cq1)


#def _fillna(logs):
#    """Fill missing data with zeros."""
#    for l in logs:
#        l.fillna(0, inplace=True)


def _fillna(logs):
    """Fill missing data with random values."""
    for l in logs:
        for col in l.columns:
            for i, v in enumerate(l[col].values):
                if pandas.isnull(v):
                    # Shift and scale so random values have median 0 and
                    # interquartile range 1
                    l[col].values[i] = (np.random.random() - 0.5) * 2


def _mydist(p):
    """Return the p-norm of the difference between x and y."""
    return lambda x, y: np.linalg.norm(x - y, ord=p)


def get_rgt(logs, p=1/8.0, its=None, path_multiplier=1.5, row_multiplier=2):
    """Find the Relative Geologic Time (RGT) of each depth in each log and
    save this in a new 'RGT' column of each log's dataframe.

    Args:
        logs: A list of pandas dataframes, one for each log.
        p: A positive int or float specifying the norm to use to measure
            distance between logs. Default 1/8.
        its: An optional int specifying how many iterations of the linear
            solver to run. Default None, which will use the default of the
            solver (100).
        path_multiplier: An optional scalar used to multiply the length of
            the longest log to estimate the maximum path length. Default 1.5.
        row_multiplier: An optional scalar used to multiply the length of
            the longest log to estimate the average number of nonzeros per
            row of the A matrix, for preallocation. Default 2.
    """

    if isinstance(p, int):
        distfunc = p
    else:
        distfunc = _mydist(p)
    dist, path, path_len = _get_path(logs, distfunc, path_multiplier)
    A = _build_A(logs, path, path_len, row_multiplier)
    _solve(A, logs, its)


def _get_path(logs, distfunc, path_multiplier=1.5):
    """Using dynamic warping to determine the path that aligns each pair of
    logs, and its distance.

    Args:
        logs: A list of pandas dataframes, one for each log.
        distfunc: An int or function to use for calculating the distance
            between two vectors when finding the minimum distance path for
            dynamic warping. If given an int p, the pth norm will be used.
            If given a function, the function should accept two vectors
            (corresponding to the measurements at a certain depths for a
            pair of logs) and return a scalar distance.
        path_multiplier: An float used to multiply the length of
            the longest log to estimate the maximum path length.

    Returns:
        (With n_logs being the number of logs, and max_path_len being the
            maximum length of the paths determined by dynamic warping.)
        dist: A (n_logs, n_logs) array of distances between pairs of logs,
            calculated by dynamic warping. Only the upper triangle, excluding
            the main diagonal, is filled (it is symmetric).
        path: A (n_logs, n_logs, max_path_len) integer array containing the
            depth indices that align pairs of logs. Log i at depth indices
            path[i, j, :] should be aligned with log j at depth indices
            path[j, i, :].
        path_len: A (n_logs, n_logs) integer array containing the length
            of the path that aligns pairs of logs. Only the upper triangle,
            excluding the main diagonal, is filled (it is symmetric).
    """
    max_len_logs = _get_max_len_logs(logs)
    est_max_path_len = _get_est_max_path_len(max_len_logs, path_multiplier)
    n_logs = len(logs)

    dist = np.zeros([n_logs, n_logs])
    path_len = np.zeros([n_logs, n_logs], dtype=np.int)
    path = np.zeros([n_logs, n_logs, est_max_path_len], dtype=np.int)
    for i, l1 in enumerate(logs[:-1]):
        l1_cols = set(l1.columns)
        for j in range(i + 1, len(logs)):
            l2 = logs[j]
            l2_cols = set(l2.columns)
            intersect_cols = list(l1_cols & l2_cols)
            dist[i, j], path_tmp = fastdtw.fastdtw(l1[intersect_cols].values,
                                                   l2[intersect_cols].values,
                                                   dist=distfunc)
            path[i, j, :len(path_tmp)] = [p[0] for p in path_tmp]
            path[j, i, :len(path_tmp)] = [p[1] for p in path_tmp]
            path_len[i, j] = len(path_tmp)
    path = path[:, :, :np.max(path_len)]
    return dist, path, path_len


def _get_max_len_logs(logs):
    """Return the maximum length of the logs."""
    max_len_logs = 0
    for l in logs:
        if len(l) > max_len_logs:
            max_len_logs = len(l)
    return max_len_logs


def _get_est_max_path_len(max_len_logs, path_multiplier):
    """Estimate the maximum path length from the maximum log length, by
    approximating that it is 'path_multiplier' times the length.
    """
    return int(path_multiplier * max_len_logs)


def _build_A(logs, path, path_len, row_multiplier):
    """Construct the A matrix for the system of equations that will be solved
    to give dRGT (the depth derivative of RGT) for each log.
    """

    A_nonzeros, indices, indptr = _allocate_A(path_len, logs, row_multiplier)
    cumulative_log_len = _get_cumulative_log_len(logs)

    # initialise number of nonzeros to 0
    num_nonzero_rows = 0
    num_nonzero_indices = 0

    for i in range(path.shape[0]-1):
        for j in range(i + 1, path.shape[0]):
            for k in range(path_len[i, j]):
                num_nonzero_indices, num_nonzero_rows = \
                        _add_row(A_nonzeros, indices, indptr, num_nonzero_rows,
                                 num_nonzero_indices, cumulative_log_len,
                                 i, j, k, path, 1.0)

    indptr[num_nonzero_rows] = num_nonzero_indices
    # chop off the overestimated space at the end
    A_nonzeros = A_nonzeros[:num_nonzero_indices]
    indices = indices[:num_nonzero_indices]
    indptr = indptr[:num_nonzero_rows + 1]
    A = csr_matrix((A_nonzeros, indices, indptr), dtype=np.float,
                   shape=(num_nonzero_rows, cumulative_log_len[-1]))
    return A


def _allocate_A(path_len, logs, row_multiplier):
    """Allocate the arrays necessary to build the sparse A matrix."""
    num_pairs = np.sum(path_len)
    num_rows = num_pairs
    max_len_logs = _get_max_len_logs(logs)
    # Estimate the average number of nonzeros in a row of the matrix to be
    # the maximum log length times a multiplier.
    est_av_num_nonzero_in_row = int(row_multiplier * max_len_logs)
    est_num_nonzero = num_rows * est_av_num_nonzero_in_row

    A_nonzeros = np.zeros(est_num_nonzero)
    indices = np.zeros(est_num_nonzero, dtype=np.int)
    indptr = np.zeros(num_rows + 1, dtype=np.int)

    return A_nonzeros, indices, indptr


def _get_cumulative_log_len(logs):
    """Return a list containing a cumulative sum of log length, starting
    from 0."""
    log_len = np.zeros(len(logs), dtype=np.int)
    for i, log in enumerate(logs):
        log_len[i] = len(log)
    return np.append([0], np.cumsum(log_len))


def _add_row(A_nonzeros, indices, indptr, num_nonzero_rows, num_nonzero_indices,
             cumulative_log_len, i, j, k, path, weight):
    """Add a row to the A matrix corresponding to one path pair.

    This sums the dRGT (depth derivative of RGT / change in RGT from previous
    depth) entries of log i up to the index in the path for that log, and
    subtracts the sum of the dRGT entries of log j up to the index in the
    path for that log. This should sum to zero, as they are supposed to
    have the same RGT, so b=0 when we solve Ax=b.
    """

    indptr[num_nonzero_rows] = num_nonzero_indices
    num_nonzero_rows += 1

    # + sum (from p = 0 to path[i,j,k]) dRGT[i][p]
    _add_shift_sum(A_nonzeros, indices, num_nonzero_indices,
                   cumulative_log_len, i, path[i, j, k], +weight)
    num_nonzero_indices += path[i, j, k] + 1

    # - sum (from p = 0 to path[j,i,k]) dRGT[j][p]
    _add_shift_sum(A_nonzeros, indices, num_nonzero_indices,
                   cumulative_log_len, j, path[j, i, k], -weight)
    num_nonzero_indices += path[j, i, k] + 1

    return num_nonzero_indices, num_nonzero_rows


def _add_shift_sum(A_nonzeros, indices, num_nonzero_indices,
                   cumulative_log_len, logidx, sampleidx, value):
    """Insert 'value' times the sum of the dRGT values of log 'logidx' up to
    'sampleidx', into the matrix A.
    """
    col0 = cumulative_log_len[logidx]
    col1 = cumulative_log_len[logidx] + sampleidx
    idx_range = range(num_nonzero_indices,
                      num_nonzero_indices + col1 - col0 + 1)
    indices[idx_range] = np.arange(col0, col1 + 1)
    A_nonzeros[idx_range] = value


def _solve(A, logs, its=None):
    """Solve the system Ax = 0, convert x from dRGT to RGT, and save into
    a new RGT column in each log.

    Using a solver constrained to only find values of dRGT bigger than 1, we
    ensure that RGT is always increasing.
    """
    b = np.zeros(A.shape[0])
    res = lsq_linear(A, b, bounds=(1.0, np.inf), verbose=2,
                     lsmr_tol='auto', max_iter=its)
    _copy_rgt_to_logs(res.x, logs)


def _copy_rgt_to_logs(x, logs):
    """Convert dRGT to RGT and save into a new RGT column of each log.

    RGT[i] = sum from p = 0 to p = i of dRGT.
    """
    k = 0
    for l in logs:
        log_len = len(l)
        rgt = np.cumsum(x[k : k + log_len])
        l['RGT'] = rgt
        k += log_len
