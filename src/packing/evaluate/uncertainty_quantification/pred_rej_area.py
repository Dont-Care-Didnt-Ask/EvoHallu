import numpy as np

from typing import List
from abc import ABC, abstractmethod


def normalize(target: List[float]):
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    target = (np.array(target) - min_t) / (max_t - min_t)
    return target


def skip_target_nans(target, estimator):
    newt, newe = [], []
    for t, e in zip(target, estimator):
        if np.isnan(t):
            continue
        newt.append(t)
        newe.append(e)
    return newt, newe


class UEMetric(ABC):
    """
    Abstract class, which measures the quality of uncertainty estimations from some Estimator using
    ground-truth uncertainty estimations calculated from some GenerationMetric.
    """

    @abstractmethod
    def __str__(self):
        """
        Abstract method. Returns unique name of the UEMetric.
        Class parameters which affect generation metric estimates should also be included in the unique name
        to diversify between UEMetric's.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Abstract method. Measures the quality of uncertainty estimations `estimator`
        by comparing them to the ground-truth uncertainty estimations `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: a quality measure of `estimator` estimations.
                Higher values can indicate either better or lower qualities,
                which depends on a particular implementation.
        """
        raise Exception("Not implemented")


def get_random_scores(function, metrics, num_iter=1000, seed=42):
    np.random.seed(seed)
    rand_scores = np.arange(len(metrics))

    value = []
    for i in range(num_iter):
        np.random.shuffle(rand_scores)
        rand_val = function(rand_scores, metrics)
        value.append(rand_val)
    return np.mean(value)


def normalize_metric(target_score, oracle_score, random_score):
    if not (oracle_score == random_score):
        target_score = (target_score - random_score) / (oracle_score - random_score)
    return target_score


class PredictionRejectionArea(UEMetric):
    """
    Calculates area under Prediction-Rejection curve.
    """

    def __init__(self, max_rejection: float = 1.0):
        """
        Parameters:
            max_rejection (float): a maximum proportion of instances that will be rejected.
                1.0 indicates entire set, 0.5 - half of the set
        """
        super().__init__()
        self.max_rejection = max_rejection

    def __str__(self):
        if self.max_rejection == 1:
            return "prr"
        return f"prr_{self.max_rejection}"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Measures the area under the Prediction-Rejection curve between `estimator` and `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: area under the Prediction-Rejection curve.
                Higher values indicate better uncertainty estimations.
        """
        target = normalize(target)
        # ue: greater is more uncertain
        ue = np.array(estimator)
        num_obs = len(ue)
        num_rej = int(self.max_rejection * num_obs)
        # Sort in ascending order: the least uncertain come first
        ue_argsort = np.argsort(ue)
        # want sorted_metrics to be increasing => smaller scores is better
        sorted_metrics = np.array(target)[ue_argsort]
        # Since we want all plots to coincide when all the data is discarded
        cumsum = np.cumsum(sorted_metrics)[-num_rej:]
        scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
        prr_score = np.sum(scores) / num_rej
        return prr_score