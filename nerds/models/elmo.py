from anago.utils import load_data_and_labels, load_glove, filter_embeddings
from anago.models import ELModel, save_model, load_model
from anago.preprocessing import ELMoTransformer
from anago.trainer import Trainer
from anago.tagger import Tagger

from keras.optimizers import Adam

from nerds.models import NERModel
from nerds.utils import get_logger, write_param_file

from sklearn.model_selection import train_test_split

import os

log = get_logger()


class ElmoNER(NERModel):

    def __init__(self,
            word_embedding_dim=100,
            char_embedding_dim=25,
            word_lstm_size=100,
            char_lstm_size=25,
            fc_dim=100,
            dropout=0.5,
            embeddings=None,
            embeddings_file="glove.6B.100d.txt",
            batch_size=16, 
            learning_rate=0.001, 
            max_iter=2):
        """ Construct a ELMo based NER model. Model is similar to the BiLSTM-CRF
            model except that the word embeddings are contextual, since they are
            returned by a trained ELMo model. ELMo model requires an additional 
            embedding, which is Glove-100 by default. ELMo model is provided by
            the (dev) Anago project.

            Parameters
            ----------
            word_embedding_dim : int, optional, default 100
                word embedding dimensions.
            char_embedding_dim : int, optional, default 25
                character embedding dimensions.
            word_lstm_size: int, optional, default 100
                character LSTM feature extractor output dimensions.
            char_lstm_size : int, optional, default 25
                word tagger LSTM output dimensions.
            fc_dim : int, optional, default 100
                output fully-connected layer size.
            dropout : float, optional, default 0.5
                dropout rate.
            embeddings : numpy array
                word embedding matrix.
            embeddings_file : str
                path to embedding file.
            batch_size : int, optional, default 16
                training batch size.
            learning_rate : float, optional, default 0.001
                learning rate for Adam optimizer.
            max_iter : int, optional, default 2
                number of epochs of training.

            Attributes
            ----------
            preprocessor_ : reference to Anago preprocessor.
            model_ : reference to the internal Anago ELModel
            trainer_ : reference to the internal Anago Trainer object.
            tagger_ : reference to the internal Anago Tagger object.
        """
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_lstm_size = word_lstm_size
        self.char_lstm_size = char_lstm_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.embeddings = embeddings
        self.embeddings_file = embeddings_file
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        # populated by fit() and load(), expected by save() and transform()
        self.preprocessor_ = None
        self.model_ = None
        self.trainer_ = None
        self.tagger_ = None


    def fit(self, X, y):
        """ Trains the NER model. Input is list of AnnotatedDocuments.

            Parameters
            ----------
            X : list(list(str))
                list of list of tokens
            y : list(list(str))
                list of list of BIO tags

            Returns
            -------
            self
        """
        if self.embeddings is None and self.embeddings_file is None:
            raise ValueError("Either embeddings or embeddings_file should be provided, exiting.")

        log.info("Preprocessing dataset...")
        self.preprocessor_ = ELMoTransformer()
        self.preprocessor_.fit(X, y)

        if self.embeddings is None:
            self.embeddings = load_glove(self.embeddings_file)
            embeddings_dim != self.embeddings[list(self.embeddings.keys())[0]].shape[0]
            self.embeddings = filter_embeddings(self.embeddings, 
                self.preprocessor_._word_vocab.vocab, 
                embeddings_dim)

        log.info("Building model...")
        self.model_ = ELModel(
            char_embedding_dim=self.char_embedding_dim,
            word_embedding_dim=self.word_embedding_dim,
            char_lstm_size=self.char_lstm_size,
            word_lstm_size=self.word_lstm_size,
            char_vocab_size=self.preprocessor_.char_vocab_size,
            word_vocab_size=self.preprocessor_.word_vocab_size,
            num_labels=self.preprocessor_.label_size,
            embeddings=self.embeddings,
            dropout=self.dropout)

        self.model_, loss = self.model_.build()
        optimizer = Adam(lr=self.learning_rate)
        self.model_.compile(loss=loss, optimizer=optimizer)
        self.model_.summary()

        log.info('Training the model...')
        self.trainer_ = Trainer(self.model_, preprocessor=self.preprocessor_)

        x_train, x_valid, y_train, y_valid = train_test_split(X, y, 
            test_size=0.1, random_state=42)
        self.trainer_.train(x_train, y_train, x_valid=x_valid, y_valid=y_valid,
            batch_size=self.batch_size, epochs=self.max_iter)

        self.tagger_ = Tagger(self.model_, preprocessor=self.preprocessor_)

        return self


    def predict(self, X):
        """ Predicts using the NER model.

            Parameters
            ----------
            X : list(list(str))
                list of list of tokens.
            
            Returns
            -------
            y : list(list(str))
                list of list of predicted BIO tags.
        """
        if self.tagger_ is None:
            raise ValueError("No tagger found, either run fit() to train or load() a trained model")

        log.info("Predicting from model...")
        ypreds = [self.tagger_.predict(" ".join(x)) for x in X]
        return ypreds


    def save(self, dirpath):
        """ Saves model to local disk, given a dirpath 
        
            Parameters
            -----------
            dirpath : str
                a directory where model artifacts will be saved. Model saves a 
                weights.h5 weights file, a params.json parameter file, and a 
                preprocessor.pkl preprocessor file.

            Returns
            -------
            None
        """
        if self.model_ is None or self.preprocessor_ is None:
            raise ValueError("No model artifacts to save, either run fit() to train or load() a trained model")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        weights_file = os.path.join(dirpath, "weights.h5")
        params_file = os.path.join(dirpath, "params.json")
        preprocessor_file = os.path.join(dirpath, "preprocessor.pkl")

        save_model(self.model_, weights_file, params_file)
        self.preprocessor_.save(preprocessor_file)

        write_param_file(self.get_params(), os.path.join(dirpath, "params.yaml"))


    def load(self, dirpath):
        """ Loads a trained model from local disk, given the dirpath

            Parameters
            ----------
            dirpath : str
                a directory where model artifacts are saved.

            Returns
            -------
            self
        """
        if not os.path.exists(dirpath):
            raise ValueError("Model directory not found: {:s}".format(dirpath))

        weights_file = os.path.join(dirpath, "weights.h5")
        params_file = os.path.join(dirpath, "params.json")
        preprocessor_file = os.path.join(dirpath, "preprocessor.pkl")

        if not (os.path.exists(weights_file) or 
                os.path.exists(params_file) or
                os.path.exists(preprocessor_file)):
            raise ValueError("Model files may be corrupted, exiting")
        
        self.model_ = load_model(weights_file, params_file)
        self.preprocessor_ = ELMoTransformer.load(preprocessor_file)
        self.tagger_ = Tagger(self.model_, preprocessor=self.preprocessor_)

        return self


