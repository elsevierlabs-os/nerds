from nerds.models import NERModel
from nerds.utils import get_logger, write_param_file

import os
import joblib
import sklearn_crfsuite
import spacy

log = get_logger()


class CrfNER(NERModel):

    def __init__(self,
            max_iter=100,
            c1=0.1,
            c2=0.1,
            featurizer=None):
        """ Construct a Conditional Random Fields (CRF) based NER. Implementation
            of CRF NER is provided by sklearn.crfsuite.CRF.

            Parameters
            ----------
            max_iter : int, optional, default 100
                maximum number of iterations to run CRF training
            c1 : float, optional, default 0.1
                L1 regularization coefficient.
            c2 : float, optional, default 0.1
                L2 regularization coefficient.
            featurizer : function, default None
                if None, the default featurizer _sent2features() is used to convert 
                list of tokens for each sentence to a list of features, where each 
                feature is a dictionary of name-value pairs. For custom features, a 
                featurizer function must be provided that takes in a list of tokens 
                (sentence) and returns a list of features.

            Attributes
            ----------
            model_ : reference to the internal sklearn_crfsuite.CRF model.
        """
        super().__init__()
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.featurizer = featurizer
        self._nlp = None
        self.model_ = None

    
    def fit(self, X, y):
        """ Build feature vectors and train CRF model. Wrapper for 
            sklearn_crfsuite.CRF model.

            Parameters
            ----------
            X : list(list(str))
                list of sentences. Sentences are tokenized into list 
                of words.
            y : list(list(str))
                list of list of BIO tags.

            Returns
            -------
            self
        """
        if self.featurizer is None:
            features = [self._sent2features(sent) for sent in X]
        else:
            features = [self.featurizer(sent) for sent in X]

        log.info("Building model...")
        self.model_ = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iter,
            all_possible_transitions=True,
            verbose=True)

        log.info("Training model...")
        self.model_.fit(features, y)

        return self


    def predict(self, X):
        """ Predicts using trained CRF model.

            Parameters
            ----------
            X : list(list(str))
                list of sentences. Sentences are tokenized into list of words.

            Returns
            -------
            y : list(list(str))
                list of list of predicted BIO tags.
        """
        if self.model_ is None:
            raise ValueError("CRF model not found, run fit() to train or load() pre-trained model")

        if self.featurizer is None:
            features = [self._sent2features(sent) for sent in X]
        else:
            features = [self.featurizer(sent) for sent in X]

        return self.model_.predict(features)


    def save(self, dirpath):
        """ Save a trained CRF model at dirpath.

            Parameters
            ----------
            dirpath : str
                path to model directory.

            Returns
            -------
            None
        """
        if self.model_ is None:
            raise ValueError("No model to save, run fit() to train or load() pre-trained model")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        model_file = os.path.join(dirpath, "crf-model.pkl")
        joblib.dump(self.model_, model_file)

        write_param_file(self.get_params(), os.path.join(dirpath, "params.yaml"))


    def load(self, dirpath):
        """ Load a pre-trained CRF model from dirpath.

            Parameters
            -----------
            dirpath : str
                path to model directory.
            
            Returns
            --------
            self
        """
        model_file = os.path.join(dirpath, "crf-model.pkl")
        if not os.path.exists(model_file):
            raise ValueError("No CRF model to load at {:s}, exiting.".format(model_file))

        self.model_ = joblib.load(model_file)
        return self


    def _load_language_model(self):
        return spacy.load("en")


    def _sent2features(self, sent):
        """ Converts a list of tokens to a list of features for CRF.
            Each feature is a dictionary of feature name value pairs.

            Parameters
            ----------
            sent : list(str))
                a list of tokens representing a sentence.

            Returns
            -------
            feats : list(dict(str, obj))
                a list of features, where each feature represents a token
                as a dictionary of name-value pairs.
        """
        if self._nlp is None:
            self._nlp = self._load_language_model()
        doc = self._nlp(" ".join(sent))
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

