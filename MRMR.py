import pandas as pd
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()
mr = importr('mRMRe')

class MRMR(object):
    """
        Implements R-MRMRe algorithm and provides basic functionality for MRMR in python manner.
        Parameters
        ----------
        n_features : int
            Number of features to select.
        filter : string
            "mRMRe.Filter" by default
        selected_features: np.array
            An array of selected indices
    """

    def __init__(self, n_features):
        self.n_features = n_features
        self.filter = "mRMRe.Filter"
        self.selected_features = np.array([], dtype=np.integer)

    def fit(self, X, y):
        """
            Fits the filter.

            Parameters
            ----------
            X : array-like, shape (n_features,n_samples)
                The training input samples.
            y : array-like, shape (n_features,n_samples)
                The target values.
            Returns
            ------
        """
        # translate data from python DataFrame to R DataFrame
        data = pd.DataFrame(X)
        data['Class'] = y
        target_indices = data.shape[1]
        r_df = pandas2ri.py2ri(data)
        mrmrData = mr.mRMR_data(data=r_df)  # Create mRMR Data

        # Call mRMR method
        selection = mr.mRMR_classic(self.filter, data=mrmrData, target_indices=target_indices, feature_count=self.n_features)
        result = mr.solutions(selection)
        result_py = pandas2ri.ri2py(result)
        result_py = result_py[0]
        result_py = result_py[0:len(result_py)]
        result_final = list(map(lambda x: x - 1, result_py))
        self.selected_features = np.append(self.selected_features, np.array(result_final))

    def transform(self, X):
        return X[:, self.selected_features]

    def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        self.fit(X, y)
        return self.transform(X)
