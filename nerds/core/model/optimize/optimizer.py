from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.base import clone

from nerds.core.model.optimize.params import ExactListParam, RangeParam
from nerds.core.model.evaluate.validation import KFoldCV
from nerds.util.logging import get_logger

log = get_logger()


class Optimizer(object):
    """ This class wraps around the popular hyperopt optimization library.
        It accepts a `NERModel`, a dictionary corresponding to a parameter grid
        to perform search on, and the named entity label to be optimized for,
        and optimizes for F-score using Tree-structured Parzen Estimators. It
        returns only the `NERModel` corresponding to the best performing
        configuration.

        The parameter configuration is accepted as a list of `RangeParam`s
        or `ExactListParam`s. So if a neural network has two parameters to
        optimize, say learning rate and number of neurons, we make a param_grid
        as:
        {
            "learning_rate": RangeParam(0.1, 0.5),
            "number_of_neurons": ExactListParam(range(10, 100))
        }
        This means that the param_grid has two parameters to be optimized:
        1) learning_rate: it is a float parameter, varying between 0.1 and 0.5
        on a continuous domain. Hence, `RangeParam`.
        2) number_of_neurons: it is an integer parameter, varying between 10
        and 100. Hence we explicitly make a list of numbers between 10 and 100,
        and feed it in as an `ExactListParam`.

        Attributes:
            ner_model (NERModel): A `NERModel` object.
            param_grid (list(RangeParam) or list(ExactListParam)):
                A list of parameters to be optimized.
            entity_label (str): The named entity to optimize for.
            cv (int, optional): The number of folds in the k-fold cross
                validation. If 1, then `eval_split` will be used to determine
                the split. Defaults to 1.
            eval_split (float, optional): Only considered when `cv = 1`.
                Determines the split percentage of the train-test sets,
                `eval_split * len(X)` is used for training, and the rest for
                test. Defaults to 0.8.
            max_evals (int, optional): Max number of optimizer evaluations.
                Defaults to 10.
            shuffle_data (bool, optional): Whether to shuffle the data before
                the cross validation. Defaults to True.

        Returns:
            NERModel: Best performing `NERModel`.

        Example use:
        brat_input_train = BratInput("..")
        X = brat_input_train.transform()
        X_sentences = split_annotated_documents(X)

        hparams = {
            "c1": RangeParam(0.01, 0.5),
            "c2": RangeParam(0.01, 0.5)
        }

        model = CRF()
        opt = Optimize(model, hparams, "nif-antibody", max_evals=10)
        opt.optimize_and_return_best(X_sentences)

        print(opt.f_score_max)
        print(opt.best)
    """

    def __init__(self, ner_model, param_grid, entity_label, cv=1,
                 eval_split=0.8, max_evals=10, shuffle_data=True):
        self.ner_model = ner_model
        self.param_grid = param_grid
        self.entity_label = entity_label
        self.cv = cv
        self.eval_split = eval_split
        self.max_evals = max_evals
        self.shuffle_data = shuffle_data

        # Prepare the param_grid for hyperopt.
        hparams = {}
        for param_name in param_grid:
            if type(param_grid[param_name]) == ExactListParam:
                hparams[param_name] = hp.choice(param_name,
                                                param_grid[param_name]
                                                .list_of_values)
            elif type(param_grid[param_name]) == RangeParam:
                hparams[param_name] = hp.uniform(param_name,
                                                 param_grid[param_name].low,
                                                 param_grid[param_name].high)
            else:
                raise TypeError("Unknown param type detected in param_grid.")
        self._hparams = hparams

    def optimize_and_return_best(self, X):
        """ Main method to start the optimization process on a dataset. """

        def objective(hparams):
            kfold = KFoldCV(
                self.ner_model, self.cv, self.eval_split, self.entity_label,
                self.shuffle_data)
            f1 = kfold.cross_validate(X, hparams)
            log.debug("F-score is {}.".format(f1))
            return {"loss": -f1, 'status': STATUS_OK}

        log.debug("Started tuning model.")
        trials = Trials()
        self.best = fmin(fn=objective, space=self._hparams, algo=tpe.suggest,
                         max_evals=self.max_evals, trials=trials)
        self.f_score_max = -min(trials.losses())
        log.debug("Finished tuning model.")

        log.debug("Started training best model.")
        self.best_estimator = clone(self.ner_model)
        self.best_estimator.fit(X, **self.best)
        log.debug("Finished training best model.")

        log.debug("Best F-score is {}, for the configuration {}."
                  .format(self.f_score_max, self.best))
        return self.best_estimator, self.f_score_max
