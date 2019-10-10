import numpy as np
from sklearn.base import TransformerMixin
from functools import reduce
from utils import cartesian


class EvolutionaryOutliersSearch(TransformerMixin):
    """
        Class for performing outliers search for high dimensional data using
        evolutionary algorithm.
    """

    def __init__(self, results_number, dimensionality, p, p1=0.5, p2=0.5, f=10, random_state=42):
        """
            Initialize EvolutionaryOutliersSearch instance

            Args:
                ----------
                results_number (int) : Number of outlier points to find.
                dimensionality (int) : Dimensionality of the projection which is used to determine the outliers.
                p (int) : number of solutions to work with.
                p1 (float) : probability of first type mutation.
                p2 (float) : probability of second type mutation.
                f (int) : number of splits for each dimension.
                random_state (int) : seed for random numbers generator.

        """
        self.result_number_ = results_number
        self.dimensionality = dimensionality
        self.p = p
        self.p1 = p1
        self.p2 = p2
        self.f = f
        np.random.seed(random_state)
        #: int: Number of instances in fitted data.
        self.records_number = None
        #: int: Number of features in fitted data.
        self.features_number = None
        # numpy.array: 3D array of index partitions according to each feature.
        self.partitions = None
        # list of numpy.array: List with best solutions found.
        self.best_set = np.empty(0)
        self.best_set_coef = np.empty(0)

    def fit(self, data, y=None):
        """
                Fit the data by partitioning it into self.f equidepth ranges featurewise.

                Args:
                    ----------
                    data (numpy.array) : 2D array of data to fit.

                Returns:
                    ----------
                    self: Follow sklearn transformers interface.
        """
        if len(data.shape) != 2:
            raise ValueError("Data must be two dimensional")
        self.records_number, self.features_number = data.shape
        partitions = np.insert(
            np.argsort(data, axis=0),
            range((self.records_number // self.f + 1) * (self.records_number % self.f)
                  if self.features_number % self.f else self.records_number,
                  self.records_number, self.records_number // self.f),
            values=-1, axis=0
        )
        self.partitions = partitions.T.reshape((self.features_number, self.f, -1))
        return self

    def transform(self, data):
        """
                Find points in the data that are marked to be outliers.

                Args:
                    ----------
                    data (numpy.array) : 2D array of data.

                Returns:
                    ----------
                    np.array: 1D array of indexes of ouliers in data
        """
        s = self.get_initial_state()
        while not self.convergence(s):
            s = self.selection(s)
            s = self.cross_over(s)
            s = self.mutation(s)
            self.save_best_solutions()

        return np.unique(np.concatenate([
            self.query_partions(solution) for solution in self.best_set
        ]))

    def query_partions(self, solution):
        """
                Query points corresponding to solution from self.partitions.

                Args:
                    ----------
                    solution (numpy.array) : 1d array representing a solution.

                Returns:
                    ----------
                    np.array: 1D array of indexes of points represented by solution
        """
        axes = np.nonzero(solution != -1)[0]
        points = self.partitions[axes, solution[axes]]
        intersection = reduce(np.intersect1d, points)
        return intersection[intersection != -1]

    def sparcity_coefficient(self, solution, dimensionality=None):
        """
                Compute sparsity coefficient of a cube represented by a solution.

                Args:
                    ----------
                    solution (numpy.array) : 1d array representing a solution.
                    dimensionality (int, optional) : dimensionality of solution
                Returns:
                    ----------
                    float: sparsity coefficient of given solution
        """
        if dimensionality is None:
            dimensionality = self.dimensionality

        points = self.query_partions(solution)
        prob = 1 / self.f ** dimensionality
        return (len(points) - self.records_number * prob) / \
               np.sqrt(self.records_number * prob * (1 - prob))

    def get_initial_state(self):
        """
                Generate array of initial solutions.

                Returns:
                    ----------
                    np.array: 2D array of possible solutions
        """
        s = np.full((self.p, self.features_number), fill_value=-1)

        for solution in s:
            positions = np.random.choice(range(self.features_number),
                                         replace=False, size=self.dimensionality)
            solution[positions] = np.random.randint(low=0, high=self.f, size=self.dimensionality)

        return s

    @staticmethod
    def convergence(s):
        """
                Check if convergence criteria for current set of solutions is met

                Args:
                    ----------
                    s (numpy.array) : 2d array representing a set of current solutions.

                Returns:
                    ----------
                    bool: boolean value whether the condition is met.
        """
        convergence_criteria = 0.95
        p, features_num = s.shape
        for j in range(features_num):
            _, count = np.unique(s[:, j], return_counts=True)
            if np.all(count < convergence_criteria * count.sum()):
                return False
        return True

    def selection(self, s):
        """
                Perform selection stage of evolutionary algorithm

                Args:
                    ----------
                    s (numpy.array) : 2d array representing a set of current solutions.

                Returns:
                    ----------
                    np.array: 2d array representing a set of solutions after selection.
        """
        p, features_num = s.shape
        coefficients = np.empty(shape=(self.p,))
        for i, solution in enumerate(s):
            coefficients[i] = self.sparcity_coefficient(solution)
        ranks = coefficients.argsort().argsort()
        return s[np.random.choice(np.arange(p), size=p, p=ranks / np.sum(ranks))]

    def cross_over(self, s):
        """
                Perform cross-over stage of evolutionary algorithm

                Args:
                    ----------
                    s (numpy.array) : 2d array representing a set of current solutions.

                Returns:
                    ----------
                    np.array: 2d array representing a set of solutions after cross-over.
        """

        def recombination(s1, s2):
            """
                    Recombine genes of two solutions. If s2 is None, simply return s1

                    Args:
                        ----------
                        s1 (numpy.array) : 1d array representing a solution.
                        s2 (numpy.array) : 1d array representing a solution.
                    Returns:
                        ----------
                        tuple of np.array: recombined solutions.
            """
            # create children
            c1, c2 = np.full_like(s1, fill_value=-1), np.full_like(s1, fill_value=-1)
            # positions where both are valid
            r = np.where(
                np.logical_and(s1 != -1, s2 != -1)
            )[0]
            # positions where exactly one is valid
            q = np.where(
                np.logical_xor(s1 != -1, s2 != -1)
            )[0]
            q_sol = s1[q] + s2[q] + 1  # valus at position q different from -1
            # find best solution among r
            r_size = r.shape[0]
            best_coef, best_sol = np.inf, None
            for sol in cartesian(np.vstack([s1[r], s2[r]]).T):
                c1[r] = sol
                coef = self.sparcity_coefficient(c1, r_size)
                if coef < best_coef:
                    best_coef, best_sol = coef, sol
            c1[r] = best_sol
            # greedily find best solution among q
            q_size = q.shape[0]
            dimensions_inserted = r_size
            taken_q = np.zeros_like(q)
            while dimensions_inserted != r_size + q_size // 2:
                dimensions_inserted += 1

                best_coef, best_index = np.inf, None
                for taken, q_index in zip(taken_q, q):
                    if not taken:
                        c1[q_index] = q_sol[q_index]
                        coef = self.sparcity_coefficient(c1, dimensions_inserted)
                        if coef < best_coef:
                            best_coef, best_index = coef, q_index
                        c1[q_index] = -1

                c1[best_index] = q_sol[best_index]
                taken_q[best_index] = 1

            # make c2 complemtary to c1
            crossover_positions = np.concatenate([r, q])
            c2[crossover_positions] = s1[crossover_positions] + s2[crossover_positions] - c1[crossover_positions]
            return c1, c2

        for i, (s1, s2) in enumerate(zip(s[::2], s[1::2])):
            s[2 * i], s[2 * i + 1] = recombination(s1, s2)
        return s

    def mutation(self, s):
        """
                Perform mutation stage of evolutionary algorithm

                Args:
                    ----------
                    s (numpy.array) : 2d array representing a set of current solutions.

                Returns:
                    ----------
                    np.array: 2d array representing a set of solutions after mutation.
        """
        for i, solution in enumerate(s):
            #         first type mutation -1 -> N, M -> -1
            if np.random.random() < self.p1:
                remove_position = np.random.choice(
                    np.where(solution != -1)[0]
                )
                insert_position = np.random.choice(
                    np.where(solution == -1)[0]
                )
                solution[i, remove_position] = -1
                solution[i, insert_position] = np.random.randint(self.f)
            # second type mutation N -> M
            if np.random.random() < self.p2:
                mutation_position = np.random.choice(
                    np.where(solution != -1)[0]
                )
                solution[i, mutation_position] = np.random.randint(self.f)
        return s

    def save_best_solutions(self, s):
        """
                Save best results_number solutions from current best solutions and s

                Args:
                    ----------
                    s (numpy.array) : 2d array representing a set of solutions.
        """
        s_coef = [self.sparcity_coefficient(sol) for sol in s]
        coefs = np.concatenate([self.best_set_coef, s_coef])
        best_indexes = np.argpartition(coefs, self.result_number_)[:self.result_number_]
        self.best_set = np.concatenate([self.best_set, s])[best_indexes]
        self.best_set_coef = np.concatenate([self.best_set_coef, s])[best_indexes]
