from nerds.models import NERModel
from nerds.utils import get_logger

from sklearn.externals import joblib

import os
import sklearn_crfsuite
import spacy

log = get_logger()


class CrfNER(NERModel):

    def __init__(self, entity_label=None):
        """ Build a sklearn.crfsuite.CRF CRF model

            Args:
                entity_label (str): label for single entity NER, default None
        """
        super().__init__(entity_label)
        self.key = "crfsuite_crf"
        self.nlp = None
        self.model = None

    
    def fit(self, X, y,
            is_featurized=False,
            max_iterations=100,
            c1=0.1,
            c2=0.1):
        """ Build feature vectors and train CRF model. Wrapper for 
            sklearn_crfsuite.CRF model. The underlying model takes many
            parameters (for full list (and possible future enhancement), see
            https://sklearn-crfsuite.readthedocs.io/en/latest/_modules/sklearn_crfsuite/estimator.html#CRF)

            Args:
                X (list(list(str))) or (list(list(dict(str, str)))): list of 
                    sentences or features. Sentences are tokenized into list 
                    of words, and features are a list of word features, each
                    word feature is a dictionary of name-value pairs.
                y (list(list(str))): list of list of BIO tags.
                is_featurized (bool, default False): if True, X is a list of list
                    of features, else X is a list of list of words.
                max_iterations (int, default 100): maximum number of 
                    iterations to run CRF training
                c1 (float, default 0.1): L1 regularization coefficient.
                c2 (float, default 0.1): L2 regularization coefficient.
        """
        if not is_featurized:
            log.info("Generating features for {:d} samples...".format(len(X)))
            if self.nlp is None:
                self.nlp = self._load_language_model()
            features = [self._sent2features(sent, self.nlp) for sent in X]
        
        log.info("Building model...")
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
            verbose=True)

        log.info("Training model...")
        self.model.fit(X if is_featurized else features, y)

        return self


    def predict(self, X, is_featurized=False):
        """ Predicts using trained CRF model.

            Args:
                X (list(list(dict(str, str))) or list(list(str))): list
                of sentences or features.
                is_featurized (bool, default False): if True, X is a list
                    of list of features, else X is a list of list of tokens.
            Returns:
                y (list(list(str))): list of list of predicted BIO tags.
        """
        if self.model is None:
            raise ValueError("CRF model not found, run fit() to train or load() pre-trained model")

        if not is_featurized:
            log.info("Generating features for {:d} samples".format(len(X)))
            if self.nlp is None:
                self.nlp = self._load_language_model()
            features = [self._sent2features(sent, self.nlp) for sent in X]

        return self.model.predict(X if is_featurized else features)        


    def save(self, dirpath):
        """ Save a trained CRF model at dirpath.

            Args:
                dirpath (str): path to model directory.
        """
        if self.model is None:
            raise ValueError("No model to save, run fit() to train or load() pre-trained model")

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        model_file = os.path.join(dirpath, "crf-model.pkl")
        joblib.dump(self.model, model_file)


    def load(self, dirpath):
        """ Load a pre-trained CRF model from dirpath.

            Args:
                dirpath (str): path to model directory.
            Returns:
                this object populated with pre-trained model.
        """
        model_file = os.path.join(dirpath, "crf-model.pkl")
        if not os.path.exists(model_file):
            raise ValueError("No CRF model to load at {:s}, exiting.".format(model_file))

        self.model = joblib.load(model_file)
        return self


    def _load_language_model(self):
        return spacy.load("en")


    def _sent2features(self, sent, nlp):
        """ Converts a list of tokens to a list of features for CRF.
            Each feature is a dictionary of feature name value pairs.
        """
        doc = nlp(" ".join(sent))
        postags = [token.pos_ for token in doc]
        features = [self._word2featdict(sent, postags, i) for i in range(len(sent))]
        return features


    def _word2featdict(self, sent, postags, pos):
        """ Build up a default feature dictionary for each word in sentence.
            The default considers a window size of 2 around each word, so it
            includes word-1, word-2, word, word+1, word+2. For each word, we
            consider:
                - prefix and suffix of size 2 and 3
                - the word itself, lowercase
                - is_upper, is_lower, begin with upper, is_digit
                - POS tag, and POS tag prefix of size 2
        """
        # current word
        word = sent[pos]
        postag = postags[pos]
        feat_dict = {
            'bias': 1.0,
            'word[-2]': word[-2:],
            'word[-3:]': word[-3:],
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[0:2],
        }
        # word - 2
        if pos > 1:
            prev_word2 = sent[pos - 2]
            prev_postag2 = postags[pos - 2]
            feat_dict.update({
                '-2:word[-2]': prev_word2[-2:],
                '-2:word[-3]': prev_word2[-3:],
                '-2:word.lower()': prev_word2.lower(),
                '-2:word.istitle()': prev_word2.istitle(),
                '-2:word.isupper()': prev_word2.isupper(),
                '-2:word.isdigit()': prev_word2.isdigit(),
                '-2:postag': prev_postag2,
                '-2:postag[:2]': prev_postag2[:2],
            })
        # word - 1
        if pos > 0:
            prev_word = sent[pos - 1]
            prev_postag = postags[pos - 1]
            feat_dict.update({
                '-1:word[-2]': prev_word[-2:],
                '-1:word[-3]': prev_word[-3:],
                '-1:word.lower()': prev_word.lower(),
                '-1:word.istitle()': prev_word.istitle(),
                '-1:word.isupper()': prev_word.isupper(),
                '-1:word.isdigit()': prev_word.isdigit(),
                '-1:postag': prev_postag,
                '-1:postag[:2]': prev_postag[:2],
            })
        # first word
        if pos == 0:
            feat_dict['BOS'] = True
        # word + 1
        if pos < len(sent) - 1:
            next_word = sent[pos + 1]
            next_postag = postags[pos + 1]
            feat_dict.update({
                '+1:word[-2]': next_word[-2:],
                '+1:word[-3]': next_word[-3:],
                '+1:word.lower()': next_word.lower(),
                '+1:word.istitle()': next_word.istitle(),
                '+1:word.isupper()': next_word.isupper(),
                '+1:word.isdigit()': next_word.isdigit(),
                '+1:postag': next_postag,
                '+1:postag[:2]': next_postag[:2],
            })
        # word + 2
        if pos < len(sent) - 2:
            next_word2 = sent[pos + 2]
            next_postag2 = postags[pos + 2]
            feat_dict.update({
                '+2:word[-2]': next_word2[-2:],
                '+2:word[-3]': next_word2[-3:],
                '+2:word.lower()': next_word2.lower(),
                '+2:word.istitle()': next_word2.istitle(),
                '+2:word.isupper()': next_word2.isupper(),
                '+2:word.isdigit()': next_word2.isdigit(),
                '+2:postag': next_postag2,
                '+2:postag[:2]': next_postag2[:2],
            })
        # last word
        if pos == len(sent) - 1:
            feat_dict['EOS'] = True
        return feat_dict

