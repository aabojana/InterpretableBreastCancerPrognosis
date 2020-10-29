import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import LeaveOneOut


class ReliabilityEstimation(object):
    pi = math.pi
    one_over_sqrt_2 = 1 / math.sqrt(2)

    def __init__(self):
        return

    def o_ref(self, predicted_proba):
        reliability = 2 * predicted_proba * (1 - predicted_proba)
        return reliability

    def hellinger_distance(self, trivial, predicted_probas):
        """
            Received two double vectors and calculates the Hellinger distance.
            Parameters
            ----------
            trivial : list
                List of trivial probabilities (for binary classification its' length is 2).
            predicted_probas : list
                List of predicted probabilities.
            Returns
                The Hellinger's distance.
            ------
        """
        if len(trivial) != len(predicted_probas):
            print("Vectors of trivial and predicted must have equal size")
            return
        sum = 0
        for i in range(len(trivial)):
            sum += math.pow(math.sqrt(trivial[i]) - math.sqrt(predicted_probas[i]), 2)

        return math.sqrt(sum) * self.one_over_sqrt_2

    def gaussian_kernel(self, standard_deviation, distance):
        gamma = 1 / (2 * math.pow(standard_deviation, 2))
        kernel = math.sqrt(gamma / self.pi) * math.exp(-gamma * math.pow(distance, 2))
        return kernel

    def individual_density(self, data, sample):
        """
            Find individual density for a given example.
            Individual density is estimated probability density function for a given unlabeled example.
            The density is estimated using Parzen window with Gaussian Kernel.
            Parameters
            ----------
            data : numpay array [number of samples x number of features]
                Data set
            sample : umpay array [1 x number of features]
                Sample for which we want to estimate reliability od prediction.
            Returns
                The individual density (float).
            ------
        """

        individualDens = 0

        # Get Euclidean distances from other instances in data set
        n_neighbors = np.shape(data)[0]
        neighbours = NearestNeighbors(n_neighbors=n_neighbors)
        neighbours.fit(data)
        distances, indices = neighbours.kneighbors(sample.reshape(1, -1), return_distance=True)

        dist_std = np.std(distances)
        for i in range(n_neighbors):
            individualDens += self.gaussian_kernel(standard_deviation=dist_std, distance=distances[0, i])

        individualDens = individualDens / n_neighbors
        return individualDens

    def DENS(self, data, sample):
        """
            Calculates DENS reliability estimation for a given example.
            Parameters
            ----------
            data : numpay array [number of samples x number of features]
                Data set
            sample : numpay array [1 x number of features]
                Sample for which we want to estimate reliability od prediction.
            ----------
            Returns
                DENS reliability estimate (float).
                Returns a value that estimates a density of problem space around the instance being predicted.
                ----------
                """
        n_examples = np.shape(data)[0]
        individual_dens = self.individual_density(data, sample)

        dens = np.zeros(n_examples)
        for i in range(n_examples):
            dens[i] = self.individual_density(data, data[i, :])

        reliability = np.max(dens) - individual_dens
        return reliability

    def CNK(self, data_input, data_output, sample, predicted_proba, n_kNN=50):
        """
        Parameters
            ----------
            data_input: numpay array [number of samples x number of features]
                Data set
            data_output: numpay array [number of samples x 1]
                Labels
            sample: numpay array [1 x number of features]
                Sample for which we want to estimate reliability od prediction.
            n_kNN: int
                Number of neighbors (default is 50)
            predicted_proba: list of predicted probabilities for a given sample
        Returns:
        For classification, CNK is equal to 1 minus the average distance between predicted class distribution and (trivial)
         class distributions of the $k$ nearest neighbours from the learning set. A greater value implies better prediction.
        """
        predicted_proba = predicted_proba[0]
        nn = NearestNeighbors(n_neighbors=n_kNN)
        nn.fit(data_input)

        # get indexes of nearest neighbours
        sample_neighbours_ind = nn.kneighbors(sample.reshape(1, -1), return_distance=False)[0]

        # calculate Hellinger's distances between sample predicted probabilities and
        # its' nearest neighbors trivial probabilities
        sum = 0
        sample_neighbours_input = np.array([])
        sample_neighbours_output = np.array([])
        for ind in sample_neighbours_ind:
            sample_neighbours_input = np.append(sample_neighbours_input, data_input[ind])
            sample_neighbours_output = np.append(sample_neighbours_output, data_output[ind])
            # for problem of binary classification
            trivial_probas = list(map(lambda x: abs(1 - x - data_output[ind]),
                                      [0, 1]))  # determine trivial probabilities for Ki nearest neighbour
            sum += self.hellinger_distance(trivial=trivial_probas, predicted_probas=predicted_proba)

        reliability = 1 - (sum / n_kNN)
        return reliability

    def LCV(self, data_input, data_output, sample, n_kNN, classifier):
        sum = 0
        # define neighborhood
        flag = False
        while not flag:
            nn = NearestNeighbors(n_neighbors=n_kNN)
            nn.fit(data_input)
            # get indexes of nearest neighbours
            sample_neighbours_ind = nn.kneighbors(sample.reshape(1, -1), return_distance=False)
            sample_neighbours_input = np.empty((n_kNN, np.shape(data_input)[1]))
            sample_neighbours_output = np.array([])
            i = 0
            for ind in sample_neighbours_ind.ravel():
                sample_neighbours_input[i] = data_input[ind]
                sample_neighbours_output = np.concatenate((sample_neighbours_output, [data_output[ind]]))
                i += 1

            clas1 = np.sum(sample_neighbours_output)
            # clas0 = n_kNN - clas1 # valid "if" code for survival problem
            # if clas0 > 4:
            if clas1 > 4:  # valid "if" code for relapse problem
                flag = True
            else:
                n_kNN += 1

        loocv = LeaveOneOut()
        for train_index, test_index in loocv.split(sample_neighbours_input, sample_neighbours_output):
            X_train, X_test = sample_neighbours_input[train_index], sample_neighbours_input[test_index]
            y_train, y_test = sample_neighbours_output[train_index], sample_neighbours_output[test_index]

            classifier.fit(X_train, y_train)
            probas = classifier.predict_proba(X_test)
            # for problem of binary classification
            trivial_probas = list(
                map(lambda x: abs(1 - x - y_test), [0, 1]))  # determine trivial probabilities for Ki nearest neighbour

            # distance between test example and sample
            nn1 = NearestNeighbors(n_neighbors=1)
            nn1.fit(np.concatenate((X_test, [sample]), axis=0))
            distance, _ = nn1.kneighbors(sample.reshape(1, -1), return_distance=True)
            sum += distance.ravel()[0] * self.hellinger_distance(trivial=trivial_probas,
                                                                 predicted_probas=probas.ravel())
            reliability = sum / n_kNN

        return reliability
