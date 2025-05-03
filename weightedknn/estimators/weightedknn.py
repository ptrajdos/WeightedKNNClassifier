import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
import threading as th

class WeightedKNNClassifier(KNeighborsClassifier):
    """
    A simple KNN classifier that uses the KNeighborsClassifier from sklearn.
    """

    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="brute",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
       
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        """
        Fit the model using the training data.
        """
        if self.algorithm != "brute":
            raise ValueError(
                "The algorithm must be 'brute' for the WeightedKNNClassifier."
            )
        
        if self.metric  == 'precomputed':
            raise ValueError(
                "The metric cannot be 'precomputed' for the WeightedKNNClassifier."
            )
        super().fit(X, y)
        self._predict_lock = th.RLock()
        self._pred_weights  = None
        return self
    
    @property
    def _fit_X(self):
        """
        Return the training data.
        """
        if self._pred_weights is not None:
            return self._pred_weights * self._fit_X_internal
        
        return self._fit_X_internal
    
    @_fit_X.setter
    def _fit_X(self, value):
        """
        Set the training data.
        """
        self._fit_X_internal = value
    
    def _check_fit(self):
        if not hasattr(self, "_predict_lock"):
            raise NotFittedError(
                "The model must be fitted before calling predict."
            )
        
    def __getstate__(self):
        state = self.__dict__.copy()
        if '_predict_lock' in state:
            del state['_predict_lock']
        return state

    def __setstate__(self, state):

        self.__dict__.update(state)
        
        if hasattr(self, "_fit_X_internal"):
            self._predict_lock = th.RLock()

    def predict(self, X, weights=None):
        self._check_fit()
        with self._predict_lock:
            if weights is None:
                return super().predict(X)

            if weights.ndim == 1:
                if len(weights) != self.n_features_in_:
                    raise ValueError(
                        f"weights must be of shape ({self.n_features_in_},), "
                        f"but got {weights.shape}."
                    )
                self._pred_weights = weights
                return super().predict(self._pred_weights * X)

            if weights.ndim == 2:
                if weights.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"weights must be of shape ({X.shape[0]}, {self.n_features_in_}), "
                        f"but got {weights.shape}."
                    )
                if weights.shape[1] != self.n_features_in_:
                    raise ValueError(
                        f"weights must be of shape ({X.shape[0]}, {self.n_features_in_}), "
                        f"but got {weights.shape}."
                    )
                predicted = []
                for w,x_idx in zip(weights, range(X.shape[0])):
                    self._pred_weights = w
                    predicted.append(super().predict(w * X[x_idx:x_idx+1,:]))
                    
                return np.asanyarray(predicted)
            
            raise ValueError(
                f"weights must be of shape ({X.shape[0]}, {self.n_features_in_}) or "
                f"({self.n_features_in_},), but got {weights.shape}."
            )
    
    def predict_proba(self, X, weights=None):
        self._check_fit()
        with self._predict_lock:
            if weights is None:
                return super().predict_proba(X)
            if weights.ndim == 1:
                if len(weights) != self.n_features_in_:
                    raise ValueError(
                        f"weights must be of shape ({self.n_features_in_},), "
                        f"but got {weights.shape}."
                    )
                self._pred_weights = weights
                return super().predict_proba(self._pred_weights * X)

            if weights.ndim == 2:
                if weights.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"weights must be of shape ({X.shape[0]}, {self.n_features_in_}), "
                        f"but got {weights.shape}."
                    )
                if weights.shape[1] != self.n_features_in_:
                    raise ValueError(
                        f"weights must be of shape ({X.shape[0]}, {self.n_features_in_}), "
                        f"but got {weights.shape}."
                    )
                predicted_proba = []
                for w, x_idx in zip(weights, range(X.shape[0])):
                    self._pred_weights = w
                    predicted_proba.append(super().predict_proba(w * X[x_idx:x_idx+1,:]))
                    
                return np.asanyarray(predicted_proba)[:,0]
            
            raise ValueError(
                f"weights must be of shape ({X.shape[0]}, {self.n_features_in_}) or "
                f"({self.n_features_in_},), but got {weights.shape}."
            )
