from unittest import TestCase
import unittest

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from weightedknn.estimators.weightedknn import WeightedKNNClassifier
from sklearn.utils.estimator_checks import check_estimator


class TestWeightedKNN(TestCase):

    def get_estimators(self):
        return {
            "WeightedKNNClassifier1": WeightedKNNClassifier(),
        }

    def test_sklearn(self):

        for name, clf in self.get_estimators().items():
            check_estimator(clf)

    def general_checker(self, X, y, better_than_random=True):
        n_classes = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            metric_val = cohen_kappa_score(y_test, y_pred)
            if better_than_random:
                self.assertTrue(
                    metric_val > 0,
                    f"Classifier should be better than random! For {clf_name}",
                )

            probas = clf.predict_proba(X)

            self.assertIsNotNone(probas, f"Probabilites are None. For {clf_name}")
            self.assertFalse(
                np.isnan(probas).any(),
                f"NaNs in probability predictions. For {clf_name}",
            )
            self.assertFalse(
                np.isinf(probas).any(),
                f"Inf in probability predictions. For {clf_name}",
            )

            self.assertTrue(
                probas.shape[0] == X.shape[0],
                f"Different number of objects in prediction. For {clf_name}",
            )
            self.assertTrue(
                probas.shape[1] == n_classes,
                f"Wrong number of classes in proba prediction. For {clf_name}",
            )

            self.assertTrue(
                np.all(probas >= 0), f"Negative probabilities. For {clf_name}"
            )
            self.assertTrue(
                np.all(probas <= 1), f"Probabilities bigger than one. For {clf_name}"
            )

            prob_sums = np.sum(probas, axis=1)
            self.assertTrue(
                np.allclose(prob_sums, np.asanyarray([1 for _ in range(X.shape[0])])),
                f"Not all sums close to one. For {clf_name}",
            )

    def test_iris(self):
        X, y = load_iris(return_X_y=True)
        self.general_checker(X, y)

    def test_digits(self):
        X, y = load_digits(return_X_y=True)
        self.general_checker(X, y)

    def test_one_val(self):

        R = 200

        X = np.zeros((R, 2))
        X[: R // 2, 0] = 1
        X[R // 2 :, 1] = 1

        y = [1 if n >= R // 2 else 0 for n in range(R)]
        n_classes = 2

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            metric_val = cohen_kappa_score(y_test, y_pred)

            self.assertTrue(
                metric_val > 0,
                f"Classifier should be better than random! For {clf_name}",
            )

            probas = clf.predict_proba(X_test)

            self.assertIsNotNone(probas, f"Probabilites are None. For {clf_name}")
            self.assertFalse(
                np.isnan(probas).any(),
                f"NaNs in probability predictions. For {clf_name}",
            )
            self.assertFalse(
                np.isinf(probas).any(),
                f"Inf in probability predictions. For {clf_name}",
            )
            self.assertTrue(
                probas.shape[0] == X_test.shape[0],
                f"Different number of objects in prediction. For {clf_name}",
            )
            self.assertTrue(
                probas.shape[1] == n_classes,
                f"Wrong number of classes in proba prediction. For {clf_name}",
            )

            self.assertTrue(
                np.all(probas >= 0), f"Negative probabilities. For {clf_name}"
            )
            self.assertTrue(
                np.all(probas <= 1), f"Probabilities bigger than one. For {clf_name}"
            )

            prob_sums = np.sum(probas, axis=1)
            self.assertTrue(
                np.allclose(
                    prob_sums, np.asanyarray([1 for _ in range(X_test.shape[0])])
                ),
                f"Not all sums close to one. For {clf_name}",
            )

            X_n = np.random.random((100, 2)) * 2 - 1

            probas = clf.predict_proba(X_n)

            self.assertIsNotNone(probas, f"Probabilites are None. For {clf_name}")
            self.assertFalse(
                np.isnan(probas).any(),
                f"NaNs in probability predictions. For {clf_name}",
            )
            self.assertFalse(
                np.isinf(probas).any(),
                f"Inf in probability predictions. For {clf_name}",
            )
            self.assertTrue(
                probas.shape[0] == X_n.shape[0],
                f"Different number of objects in prediction. For {clf_name}",
            )
            self.assertTrue(
                probas.shape[1] == n_classes,
                f"Wrong number of classes in proba prediction. For {clf_name}",
            )

            self.assertTrue(
                np.all(probas >= 0), f"Negative probabilities. For {clf_name}"
            )
            self.assertTrue(
                np.all(probas <= 1), f"Probabilities bigger than one. For {clf_name}"
            )

            prob_sums = np.sum(probas, axis=1)
            self.assertTrue(
                np.allclose(prob_sums, np.asanyarray([1 for _ in range(X_n.shape[0])])),
                f"Not all sums close to one. For {clf_name}",
            )

    def test_weighted_1D_iris(self):
        X, y = load_iris(return_X_y=True)
        weights = np.array([1, 2, 3, 4])

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X, y)
            y_pred = clf.predict(X, weights=weights)

            metric_val = cohen_kappa_score(y, y_pred)
            self.assertTrue(
                metric_val > 0,
                f"Classifier should be better than random! For {clf_name}",
            )
            probas = clf.predict_proba(X, weights=weights)
            self.assertIsNotNone(probas, f"Probabilites are None. For {clf_name}")
            self.assertFalse(
                np.isnan(probas).any(),
                f"NaNs in probability predictions. For {clf_name}",
            )
            self.assertFalse(
                np.isinf(probas).any(),
                f"Inf in probability predictions. For {clf_name}",
            )
            self.assertTrue(
                probas.shape[0] == X.shape[0],
                f"Different number of objects in prediction. For {clf_name}",
            )
            self.assertTrue(
                probas.shape[1] == len(np.unique(y)),
                f"Wrong number of classes in proba prediction. For {clf_name}",
            )
            self.assertTrue(
                np.all(probas >= 0), f"Negative probabilities. For {clf_name}"
            )
            self.assertTrue(
                np.all(probas <= 1), f"Probabilities bigger than one. For {clf_name}"
            )
            prob_sums = np.sum(probas, axis=1)
            self.assertTrue(
                np.allclose(prob_sums, np.asanyarray([1 for _ in range(X.shape[0])])),
                f"Not all sums close to one. For {clf_name}",
            )

    def test_weighted_2D_iris(self):
        X, y = load_iris(return_X_y=True)
        weights = np.random.random((X.shape[0], X.shape[1]))

        for clf_name, clf in self.get_estimators().items():
            clf.fit(X, y)
            y_pred = clf.predict(X, weights=weights)

            metric_val = cohen_kappa_score(y, y_pred)
            self.assertTrue(
                metric_val > 0,
                f"Classifier should be better than random! For {clf_name}",
            )
            probas = clf.predict_proba(X, weights=weights)
            self.assertIsNotNone(probas, f"Probabilites are None. For {clf_name}")
            self.assertFalse(
                np.isnan(probas).any(),
                f"NaNs in probability predictions. For {clf_name}",
            )
            self.assertFalse(
                np.isinf(probas).any(),
                f"Inf in probability predictions. For {clf_name}",
            )
            self.assertTrue(
                probas.shape[0] == X.shape[0],
                f"Different number of objects in prediction. For {clf_name}",
            )
            self.assertTrue(
                probas.shape[1] == len(np.unique(y)),
                f"Wrong number of classes in proba prediction. For {clf_name}",
            )
            self.assertTrue(
                np.all(probas >= 0), f"Negative probabilities. For {clf_name}"
            )
            self.assertTrue(
                np.all(probas <= 1), f"Probabilities bigger than one. For {clf_name}"
            )
            prob_sums = np.sum(probas, axis=1)
            self.assertTrue(
                np.allclose(prob_sums, np.asanyarray([1 for _ in range(X.shape[0])])),
                f"Not all sums close to one. For {clf_name}",
            )

    def test_wrong_weights(self):
        X, y = load_iris(return_X_y=True)

        weights = [
            np.random.random((X.shape[0], X.shape[1] + 1)),
            np.random.random((X.shape[0] + 1, X.shape[1])),
            np.random.random((X.shape[0] + 1, X.shape[1] + 1)),
            np.random.random((X.shape[0] + 1,)),
        ]
        for w in weights:
            for clf_name, clf in self.get_estimators().items():
                clf.fit(X, y)
                with self.assertRaises(
                    ValueError, msg=f"{clf_name} should have raised an error"
                ):
                    clf.predict(X, weights=w)

                with self.assertRaises(
                    ValueError, msg=f"{clf_name} should have raised an error"
                ):
                    clf.predict_proba(X, weights=w)

    def test_wrong_fit(self):
        X, y = load_iris(return_X_y=True)
        wrong_arg_dicts = [
            {"metric": "wrong"},
            {"metric":"precomputed"},
            {"algorithm": "wrong"},
            {"algorithm": "kdtree"},
            {"algorithm": "balltree"},
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
