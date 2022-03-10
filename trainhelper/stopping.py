### -------------------
### --- Third-Party ---
### -------------------
import optuna
import operator
import numpy as np


class EarlyStoppingCallback_torch:
    """
    Early stops the training if test loss doesn't improve after a given Patience
    """
    def __init__(self, patience):
        self.patience = patience
        self.count = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, current_loss):
        val = current_loss
        if self.best_loss is None:
            self.best_loss = val
        elif current_loss > self.best_loss:
            self.count += 1
        else:
            self.best_loss = current_loss
            self.count = 0

    def should_stop(self):
        if self.count >= self.patience:
            self.early_stop = True
        else:
            self.early_stop = False

class EarlyStoppingCallback:
    """Early stopping callback for Optuna."""
    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
            self._best_score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
            self._best_score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        """multi-object, but looks only on Accuracy """
        for trial in study.best_trials[-5:]:
            if self._operator(trial.values[0], self._best_score):
                self._best_score = trial.values[0]

        if self._operator(self._best_score, self._score):
            self._iter = 0
            self._score = self._best_score
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()

#class EarlyStoppingCallback(object):
#    """Early stopping callback for Optuna."""

#    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
#        self.early_stopping_rounds = early_stopping_rounds

#        self._iter = 0

#        if direction == "minimize":
#            self._operator = operator.lt
#            self._score = np.inf
#        elif direction == "maximize":
#            self._operator = operator.gt
#            self._score = -np.inf
#        else:
#            ValueError(f"invalid direction: {direction}")

    #def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
    #    """Do early stopping."""
    #    if self._operator(study.best_value, self._score):
    #        self._iter = 0
    #        self._score = study.best_value
    #    else:
    #        self._iter += 1

    #    if self._iter >= self.early_stopping_rounds:
    #        study.stop()
