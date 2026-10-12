from unittest import TestCase
import pickle
import unittest

import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from weightedknn.estimators.weightedknn_regressor import WeightedKNNRegressor
from sklearn.utils.estimator_checks import check_estimator


class TestWeightedKNNRegressor(TestCase):

    def get_estimators(self):
        return {
            "WeightedKNNRegressor1": WeightedKNNRegressor(),
            "WeightedKNNRegressorDistance": WeightedKNNRegressor(weights="distance"),
        }

    def check_predictions(self, y_pred, n_samples, clf_name, n_outputs=None):
        expected_shape = (n_samples,) if n_outputs is None else (n_samples, n_outputs)
        self.assertEqual(
            y_pred.shape, expected_shape, f"Wrong prediction shape. For {clf_name}"
        )
        self.assertFalse(np.isnan(y_pred).any(), f"NaNs in predictions. For {clf_name}")
        self.assertFalse(np.isinf(y_pred).any(), f"Inf in predictions. For {clf_name}")

    def test_sklearn(self):

        for name, clf in self.get_estimators().items():
            check_estimator(clf)

    def test_diabetes(self):
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            self.check_predictions(y_pred, X_test.shape[0], clf_name)
            self.assertTrue(
                r2_score(y_test, y_pred) > 0,
                f"Regressor should be better than the mean! For {clf_name}",
            )

    def test_multioutput(self):
        X, y = make_regression(
            n_samples=100, n_features=4, n_targets=3, random_state=0
        )
        weights_list = [
            None,
            np.random.random(X.shape[1]),
            np.random.random((X.shape[0], X.shape[1])),
        ]

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X, y)
            for weights in weights_list:
                y_pred = clf.predict(X, weights=weights)
                self.check_predictions(y_pred, X.shape[0], clf_name, n_outputs=3)

    def test_weighted_1D_diabetes(self):
        X, y = load_diabetes(return_X_y=True)
        weights = np.arange(1, X.shape[1] + 1)

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X, y)
            y_pred = clf.predict(X, weights=weights)
            self.check_predictions(y_pred, X.shape[0], clf_name)
            self.assertTrue(
                r2_score(y, y_pred) > 0,
                f"Regressor should be better than the mean! For {clf_name}",
            )

    def test_weighted_2D_diabetes(self):
        X, y = load_diabetes(return_X_y=True)
        weights = np.random.random((X.shape[0], X.shape[1]))

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X, y)
            y_pred = clf.predict(X, weights=weights)
            self.check_predictions(y_pred, X.shape[0], clf_name)

    def test_weighted_2D_zeros(self):
        X, y = load_diabetes(return_X_y=True)
        weights = np.zeros((X.shape[0], X.shape[1]))

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X, y)
            y_pred = clf.predict(X, weights=weights)
            self.check_predictions(y_pred, X.shape[0], clf_name)

    def test_weighted_ones_equal_unweighted(self):
        X, y = load_diabetes(return_X_y=True)

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X, y)
            y_pred = clf.predict(X)
            self.assertTrue(
                np.allclose(y_pred, clf.predict(X, weights=np.ones(X.shape[1]))),
                f"1D unit weights should not change predictions. For {clf_name}",
            )
            self.assertTrue(
                np.allclose(y_pred, clf.predict(X, weights=np.ones(X.shape))),
                f"2D unit weights should not change predictions. For {clf_name}",
            )

    def test_weights_match_scaled_features(self):
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )
        rng = np.random.default_rng(0)
        w = rng.random(X.shape[1])
        weights_2d = rng.random(X_test.shape)

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X_train, y_train)

            reference = KNeighborsRegressor(**clf.get_params())
            reference.fit(w * X_train, y_train)
            self.assertTrue(
                np.allclose(clf.predict(X_test, weights=w), reference.predict(w * X_test)),
                f"1D weights should equal scaling the features. For {clf_name}",
            )

            y_pred = clf.predict(X_test, weights=weights_2d)
            for i, w_row in enumerate(weights_2d):
                reference.fit(w_row * X_train, y_train)
                self.assertTrue(
                    np.allclose(y_pred[i], reference.predict(w_row * X_test[i : i + 1])),
                    f"2D weights should equal per-sample scaling. For {clf_name}",
                )

    def test_weights_do_not_persist(self):
        X, y = load_diabetes(return_X_y=True)

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X, y)
            y_pred = clf.predict(X)
            clf.predict(X, weights=np.random.random(X.shape[1]))
            clf.predict(X, weights=np.random.random(X.shape))
            self.assertTrue(
                np.allclose(y_pred, clf.predict(X)),
                f"Weights leaked into a later unweighted predict. For {clf_name}",
            )

    def test_pickle(self):
        X, y = load_diabetes(return_X_y=True)
        weights = np.random.random(X.shape[1])

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X, y)
            clf_restored = pickle.loads(pickle.dumps(clf))
            self.assertTrue(
                np.allclose(
                    clf.predict(X, weights=weights),
                    clf_restored.predict(X, weights=weights),
                ),
                f"Pickled model predicts differently. For {clf_name}",
            )

    def test_wrong_weights(self):
        X, y = load_diabetes(return_X_y=True)

        weights = [
            np.random.random((X.shape[0], X.shape[1] + 1)),
            np.random.random((X.shape[0] + 1, X.shape[1])),
            np.random.random((X.shape[0] + 1, X.shape[1] + 1)),
            np.random.random((X.shape[0] + 1,)),
            np.random.random((X.shape[0], X.shape[1], 1)),
        ]
        for w in weights:
            for clf_name, clf in self.get_estimators().items():
                clf.fit(X, y)
                with self.assertRaises(
                    ValueError, msg=f"{clf_name} should have raised an error"
                ):
                    clf.predict(X, weights=w)

    def test_wrong_fit(self):
        X, y = load_diabetes(return_X_y=True)
        wrong_arg_dicts = [
            {"metric": "wrong"},
            {"metric": "precomputed"},
            {"algorithm": "wrong"},
            {"algorithm": "kd_tree"},
            {"algorithm": "ball_tree"},
        ]
        for wrong_args in wrong_arg_dicts:
            for clf_name, clf in self.get_estimators().items():
                with self.assertRaises(
                    ValueError, msg=f"{clf_name} should have raised an error"
                ):
                    clf.set_params(**wrong_args)
                    clf.fit(X, y)


if __name__ == "__main__":
    unittest.main()
