import numpy as np
from functools import reduce


def get_outliers(data, results_number, dimensionality, p, p1=0.5, p2=0.5, f=10):
    """
    Find results_number outliers in data.

    Args:
        ----------
        data (numpy.array) : 2D array.
        results_number (int) : Number of outlier points to return.
        dimensionality (int) : Dimensionality of the projection which is used in order to determine the outliers.
        p (int) : number of solutions to work with.
        p1 (float) : probability of first type mutation.
        p2 (float) : probability of second type mutation.
        f (int) : number of splits for each dimension.

    Returns:
        ----------
        np.array: 2D array of results_number of outlier points.
    """
    records_num, features_num = data.shape
    best_set = []
    s = get_initial_state(data, dimensionality, p, f)
    print(s)
    return

    while not convergence(s):
        s = selection(s)
        s = cross_over(s)
        s = mutation(s, p1, p2)
        best_set = get_best(best_set, s, records_num)

    return get_solutions(best_set)


def get_initial_state(data, dimensionality, p, f):
    records_num, features_num = data.shape
    s = np.full((p, features_num), fill_value=-1)

    for solution in s:
        positions = np.random.choice(range(features_num), replace=False, size=dimensionality)
        solution[positions] = np.random.randint(low=0, high=f, size=dimensionality)

    return s


def selection(s):
    return


def convergence(s):
    return


def cross_over(s):
    return


def mutation(s, p1, p2):
    return


def get_best(best_set, s, records_num):
    return


def get_solutions(best_set):
    return


def sparcity_coefficient(partitions, solution, dimensionality, f, records_num):
    """
            Compute sparsity coefficient of a cube represented by a solution.

            Args:
                ----------
                partitions (numpy.array) : 3D array of index partitions according to each feature.
                solution (numpy.array) : 1d array representing a solution.
                dimensionality (int) : Dimensionality of the projection.
                f (int) : number of splits for each dimension.
                records_num (int) : Number of point in a data set.
            Returns:
                ----------
                np.array: 3D array of index partitions according to each feature
    """
    points = query_partions(partitions, solution)
    prob = f ** dimensionality
    return (len(points) - records_num * prob) / np.sqrt(records_num * prob * (1 - prob))


def query_partions(partitions, solution):
    """
            Query points corresponding to solution.

            Args:
                ----------
                partitions (numpy.array) : 3D array of index partitions according to each feature.
                solution (numpy.array) : 1d array representing a solution.

            Returns:
                ----------
                np.array: 1D array of indexes of points represented by solution
        """
    axes = np.nonzero(solution != -1)[0]
    points = partitions[axes, solution[axes]]
    intersection = reduce(np.intersect1d, points)
    return intersection[intersection != -1]


def partition_data(data, f):
    """
        Partition data into f equidepth ranges featurewise.

        Args:
            ----------
            data (numpy.array) : 2D array.
            f (int) : number of splits for each dimension.

        Returns:
            ----------
            np.array: 3D array of index partitions according to each feature
    """
    records_num, features_num = data.shape
    partitions = np.insert(
        np.argsort(data, axis=0),
        range((records_num // f + 1) * (records_num % f) if records_num % f else records_num,
              records_num, records_num // f),
        values=-1, axis=0
    )
    return partitions.T.reshape((features_num, f, -1))
