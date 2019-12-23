import flair
import os
import torch

from flair.data import Corpus, Sentence, Token
from flair.embeddings import (CharacterEmbeddings, TokenEmbeddings, 
        WordEmbeddings, StackedEmbeddings)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from sklearn.model_selection import train_test_split
from torch.optim import SGD, Adam

from nerds.models import NERModel
from nerds.utils import get_logger, write_param_file

log = get_logger()

class FlairNER(NERModel):

    def __init__(self,
            basedir,
            hidden_dim=256,
            embeddings=None,
            use_crf=True,
            use_rnn=True,
            num_rnn_layers=1,
            dropout=0.0,
            word_dropout=0.05,
            locked_dropout=0.5,
            optimizer="sgd",
            learning_rate=0.1,
            batch_size=32,
            max_iter=10):
        """ Construct a FLAIR NER.

            Parameters
            ----------
            basedir : str
                directory where all model artifacts will be written.
            hidden_dim : int, optional, default 256
                dimension of RNN hidden layer.
            embeddings : flair.embeddings.TokenEmbeddings, optional
                if not provided, default embedding used is stacked GloVe 
                WordEmbeddings and CharacterEmbeddings.
            use_crf : bool, default True
                if True, CRF decoder layer is used in model, otherwise absent.
            use_rnn : bool, default True
                if True, RNN layer used after Embeddings, otherwise absent.
            dropout : float, optional, default 0.0
                dropout probability.
            word_dropout : float, optional, default 0.05
                word dropout probability.
            locked_dropout : float, optional, default 0.5
                locked dropout probability.
            optimizer : str, optional, default "sgd"
                valid values are "sgd" and "adam"
            learning_rate : float, optional, default 0.1
                learning rate for (SGD) optimizer.
            batch_size : int, optional, default 32
                batch size to use during training.
            max_iter : int, optional, default 10 
                number of epochs to train.

            Attributes
            ----------
            model_ : reference to the underlying flair.models.SequenceTagger model.
        """
        super().__init__()
        self.basedir = basedir
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.use_crf = use_crf
        self.use_rnn = use_rnn
        self.num_rnn_layers = num_rnn_layers
        self.dropout = dropout
        self.word_dropout = word_dropout
        self.locked_dropout = locked_dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.model_ = None


    def fit(self, X, y):
        """ Build feature vectors and train FLAIR model.

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
        log.info("Creating FLAIR corpus...")
        Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.1)
        sents_train = self._convert_to_flair(Xtrain, ytrain)
        sents_val = self._convert_to_flair(Xval, yval)
        corpus_train = Corpus(sents_train, sents_val, [], name="train-corpus")

        tag_dict = corpus_train.make_tag_dictionary(tag_type="ner")

        if self.embeddings is None:
            embedding_types = [
                WordEmbeddings("glove"),
                CharacterEmbeddings()    
            ]
            self.embeddings = StackedEmbeddings(embeddings=embedding_types)

        log.info("Building FLAIR NER...")
        self.model_ = SequenceTagger(hidden_size=self.hidden_dim,
            embeddings=self.embeddings,
            tag_dictionary=tag_dict,
            tag_type="ner",
            use_crf=self.use_crf,
            use_rnn=self.use_rnn,
            rnn_layers=self.num_rnn_layers,
            dropout=self.dropout,
            word_dropout=self.word_dropout,
            locked_dropout=self.locked_dropout)

        log.info("Training FLAIR NER...")
        opt = torch.optim.SGD if self.optimizer == "sgd" else torch.optim.Adam
        trainer = ModelTrainer(self.model_, corpus_train, opt)
        trainer.train(base_path=self.basedir,
            learning_rate=self.learning_rate,
            mini_batch_size=self.batch_size,
            max_epochs=self.max_iter)

        return self


    def predict(self, X):
        """ Predicts using trained FLAIR model.

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
            raise ValueError("Cannot predict with empty model, run fit() to train or load() pretrained model.")

        log.info("Generating predictions...")
        sents_test = self._convert_to_flair(X)
        sents_pred = self.model_.predict(sents_test,
            mini_batch_size=self.batch_size,
            all_tag_prob=True)
        _, ypred = self._convert_from_flair(sents_pred)

        return ypred


    def save(self, dirpath):
        """ Save trained FLAIR NER model at dirpath.

            Parameters
            ----------
            dirpath : str
                path to model directory.

            Returns
            -------
            None
        """
        if self.model_ is None:
            raise ValueError("Cannot save empty model, run fit() to train or load() pretrained model.")

        if not(os.path.exists(dirpath) and os.path.isdir(dirpath)):
            os.makedirs(dirpath)
        self.model_.save(os.path.join(dirpath, "final-model.pt"))

        write_param_file(self.get_params(), os.path.join(dirpath, "params.yaml"))   


    def load(self, dirpath):
        """ Load a pre-trained FLAIR NER model from dirpath.

            Parameters
            ----------
            dirpath : str
                path to model directory.
            
            Returns
            -------
            self
        """
        if not(os.path.exists(dirpath) and os.path.isdir(dirpath)):
            raise ValueError("Model directory {:s} not found".format(dirpath))

        if not os.path.exists(os.path.join(dirpath, "final-model.pt")):
            raise ValueError("No model file in directory {:d}".format(dirpath))

        self.model_ = SequenceTagger.load(os.path.join(dirpath, "final-model.pt"))

        return self


    def _convert_to_flair(self, data, labels=None):
        """ Convert data and labels into a list of flair.data.Sentence objects.

            Parameters
            ----------
            data : list(list(str))
                list of list of tokens, each inner list represents a list of
                    tokens or words in sentence, and each outer list represents
                    a sentence.
            labels : list(list(str)), can be None
                list of list of NER tags corresponding to tokens in data.

            Returns
            -------
            sentences : list(flair.data.Sentence)
        """
        sentences = []
        if labels is None:
            labels = data
            use_dummy_labels = True
        else:
            use_dummy_labels = False
        for tokens, tags in zip(data, labels):
            sentence = Sentence()
            for token, tag in zip(tokens, tags):
                t = Token(token)
                if not use_dummy_labels:
                    t.add_tag("ner", tag)
                sentence.add_token(t)
            sentences.append(sentence)
        return sentences


    def _convert_from_flair(self, sentences):
        """ Convert a list of flair.data.Sentence objects to parallel lists for
            data and label lists.

            Parameters
            ----------
            sentences : list(flair.data.Sentence)
                list of FLAIR Sentence objects populated with tag predictions.

            Returns
            -------
            data : list(list(str))
                list of list of tokens.
            labels : list(list(str))
                list of list of tags.
        """
        data, labels = [], []
        for sentence in sentences:
            tokens = [t.text for t in sentence.tokens]
            tags = [t.tags["ner"].value for t in sentence.tokens]
            data.append(tokens)
            labels.append(tags)
        return data, labels
