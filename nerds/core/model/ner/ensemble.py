import json
import os
import os.path

import numpy as np

from nerds.core.model.evaluate.validation import KFoldCV
from nerds.core.model.input.document import AnnotatedDocument
from nerds.core.model.ner.base import NERModel
from nerds.core.model.ner.bilstm import BidirectionalLSTM
from nerds.core.model.ner.crf import CRF
from nerds.core.model.ner.spacy import SpaCyStatisticalNER
from nerds.util.file import mkdir
from nerds.util.logging import get_logger

log = get_logger()

""" The following lines define a mapping from the name of the directories where
    we persist models, to the class they instantiate once loaded.
    There is a similar mapping in nerds.core.model.config.ensemble that we use
    for loading the appropriate model configuration for each model.
"""

# If a new model is supported, add it here.
SUPPORTED_MODELS = ("BiLSTM_NER", "CRF_NER", "SpaCy_NER")
# Maps the model names to their respective classes (useful for instantiation).
MODEL_CLASS_MAPPING = {
    "BiLSTM_NER": BidirectionalLSTM,
    "CRF_NER": CRF,
    "SpaCy_NER": SpaCyStatisticalNER
}


class NERModelEnsemble(NERModel):
    """ Abstraction for ensembling multiple NER models and producing annotations.

        This class accepts a list of NER models as input and annotates a set of
        documents based on a voting mechanism. The `vote` method in this class
        will raise a NotImplementedError, and should be overriden by offspring.

        Attributes:
            models (list(NERModel)): The NER models that participate in the
                ensemble method.
    """

    def __init__(self, models):
        super().__init__()
        self.models = models

    def fit(self, X, y=None, hparams=None):
        """ Train each NER model in the ensemble. The input is a list of
            `AnnotatedDocument` instances.

            Args:
                hparams (dict(dict)): Every model has a key associated with it
                    e.g. CRF has the key "crf", BidirectionalLSTM "bilstm" etc.
                    This parameter is a dictionary where the keys are the model
                    keys, and the values are a dictionary with their
                    hyperparameters and their values. Example:
                    {
                        "crf": {
                            "c1": 0.1,
                            "c2": 0.1
                        },
                        "bilstm"
                        ...
                    }
        """
        for model in self.models:
            log.info("Training {}...".format(type(model).__name__))
            model.fit(X, y, **hparams[model.key])
            log.info("Done")
        return self

    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.

            Needs an implementation of the `vote` method.
        """
        annotated_entities_per_model = []
        for model in self.models:
            annotated_entities_per_model.append(model.extract(X, y))

        annotated_documents = []
        for doc_idx, document in enumerate(X):
            entity_matrix = np.array(
                annotated_entities_per_model)[:, doc_idx].tolist()
            annotated_documents.append(AnnotatedDocument(
                document.content,
                self.vote(entity_matrix),
                document.encoding))
        return annotated_documents

    def save(self, file_path):
        """ Saves an ensemble of models to the local disk, provided a
            file path.
        """
        save_path = os.path.join(file_path, "Ensemble_NER")
        mkdir(save_path)
        for model in self.models:
            model.save(save_path)

    def load(self, file_path):
        """ Loads an ensemble of models saved locally. """
        load_path = os.path.join(file_path, "Ensemble_NER")

        # The names of the folders are predefined internally
        model_filenames = [
            filename for filename in os.listdir(load_path)
            if filename in SUPPORTED_MODELS]

        self.models = []
        for model_filename in model_filenames:
            load_cur_model_path = os.path.join(load_path, model_filename)
            model = MODEL_CLASS_MAPPING[model_filename]
            model.load(load_cur_model_path)
            self.models.append(model)

    def vote(self, entity_matrix):
        """ If __k__ NER models have annotated a single document with entities,
            this method returns a single vector of entities as a result of an
            ensemble process. The ensemble process itself and thus the voting
            algorithm should be overriden in a subclass of this class.
        """
        raise NotImplementedError


class NERModelEnsemblePooling(NERModelEnsemble):
    def vote(self, entity_matrix):
        """ If __k__ NER models have annotated a single document with entities,
            this method returns a single vector with all unique entities that
            every model detected.

            Args:
                entity_matrix (2d list): Entities that have been annotated in a
                    single document by __k__ different NER models.

            Returns:
                list: Entities that have been selected after the ensemble.

            Example:
                If entity_matrix: [[x1, x2, x3], [x1, x3, x4], [x1, x2]]
                Then the result is: [x1, x2, x3, x4].
        """
        feature_set = set()
        for entity_vector in entity_matrix:
            feature_set.update(set(entity_vector))
        return sorted(list(feature_set))


class NERModelEnsembleMajorityVote(NERModelEnsemble):
    def vote(self, entity_matrix):
        """ If __k__ NER models have annotated a single document with entities,
            this method returns a single vector of entities as a result of a
            majority vote ensemble process.

            Args:
                entity_matrix (2d list): Entities that have been annotated in a
                    single document by __k__ different NER models.

            Returns:
                list: Entities that have been selected after the ensemble.

            Example:
                If entity_matrix: [[x1, x2, x3], [x1, x3, x4], [x1, x2]]
                Then the result is: [x1, x2, x3] because these entities have
                been annotated by 2/3 of the NER models.
        """
        feature_set = set()
        for entity_vector in entity_matrix:
            feature_set.update(set(entity_vector))
        feature_list = sorted(list(feature_set))
        feature_matrix = []
        for entity_vector in entity_matrix:
            feature_vector = []
            for feature in feature_list:
                if feature in entity_vector:
                    feature_vector.append(1)
                else:
                    feature_vector.append(0)
            feature_matrix.append(feature_vector)
        result = []
        for feature_idx, total in enumerate(np.sum(feature_matrix, axis=0)):
            if total >= len(self.models) / 2:
                result.append(feature_list[feature_idx])
        return result


class NERModelEnsembleWeightedVote(NERModelEnsemble):
    def __init__(self, models):
        self.confidence_scores = []
        super().__init__(models)

    def fit(self, X, y=None, cv=5, eval_split=0.8):
        """ Train each NER model in the ensemble, and keep their cross
            validation scores to later assign a confidence level during
            voting.
        """
        for model in self.models:
            kfold = KFoldCV(model, k=cv, eval_split=eval_split)
            hparams = {}  # TODO: nerds/issues/24 (passing hyperparams).
            # Right now it will fall back to the default hyperparameters.
            score = kfold.cross_validate(X, hparams)
            self.confidence_scores.append(score)
        return super().fit(X, y)

    def vote(self, entity_matrix):
        """ If __k__ NER models have annotated a single document with entities,
            this method returns a single vector of entities as a result of a
            weighted vote ensemble process. The weights are determined by the
            performance of each individual classifier during cross validation,
            i.e. the votes of strong predictors matter more.

            Args:
                entity_matrix (2d list): Entities that have been annotated in a
                    single document by __k__ different NER models.

            Returns:
                list: Entities that have been selected after the ensemble.

            Example:
                Let models m1, m2, m3 have a cross-validation f-score of
                f1 = 0.6, f2 = 0.9, f3 = 0.3 respectively.
                If entity_matrix: [[x1, x2, x3], [x1, x3, x4], [x1, x2]]
                Then the result is: [x1, x2, x3, x4] (unlike in the majority
                vote), because x4 was selected by a strong predictor.
        """
        feature_set = set()
        for entity_vector in entity_matrix:
            feature_set.update(set(entity_vector))
        feature_list = sorted(list(feature_set))
        feature_matrix = []
        for entity_vector in entity_matrix:
            feature_vector = []
            for feature in feature_list:
                if feature in entity_vector:
                    feature_vector.append(1)
                else:
                    feature_vector.append(0)
            feature_matrix.append(feature_vector)
        result = []
        for feature_idx, total in enumerate(np.apply_along_axis(
                self._calculate_weighted_sum, 0, feature_matrix)):
            # 1) Threshold is set at 0.5 because the factors (w_i/w) in
            # _calculate_weighted_sum always sum up to 1; so 0.5 always
            # represents half of the voting capacity.
            # 2) np.around because we often get floats like 0.499999997...
            if np.around(total, 3) >= 0.5:
                result.append(feature_list[feature_idx])
        return result

    def _calculate_weighted_sum(self, r):
        """ If we have 3 models: m1, m2, m3 with confidence scores w1, w2, w3 then:
            E = sum: (w_i/w) * r_i
            Where r_i is a vertical slice of the annotation matrix, i.e. what
            did model m_i vote for an annotated entity x_i.
        """
        w = np.sum(self.confidence_scores)
        w_i = np.true_divide(self.confidence_scores, w)
        E = np.dot(w_i, r)  # "E" stands for "Ensemble".
        return E

    def save(self, file_path):
        """ Saves an ensemble of models to the local disk, provided a
            file path. This implementation also saves the confidence scores
            as metadata.
        """
        super().save(file_path)
        save_path = os.path.join(file_path, "Ensemble_NER")
        metadata_filename = "Ensemble_metadata.json"
        metadata_save_path = os.path.join(save_path, metadata_filename)
        if len(self.confidence_scores) > 0:
            with open(metadata_save_path, "w") as fp:
                fp.write(json.dumps({
                    "confidence_scores": self.confidence_scores
                }))

    def load(self, file_path):
        """ Loads an ensemble of models saved locally. This implementation
            also loads the confidence scores, if available.
        """
        super().load(file_path)
        load_path = os.path.join(file_path, "Ensemble_NER")
        metadata_filename = "Ensemble_metadata.json"
        metadata_load_path = os.path.join(load_path, metadata_filename)
        try:
            with open(metadata_load_path, "r") as fp:
                init_metadata = json.loads(fp.read().strip())
            self.confidence_scores = init_metadata["confidence_scores"]
        except FileNotFoundError:
            self.confidence_scores = []
