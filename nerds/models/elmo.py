from anago.utils import load_data_and_labels, load_glove, filter_embeddings
from anago.models import ELModel, save_model, load_model
from anago.preprocessing import ELMoTransformer
from anago.trainer import Trainer
from anago.tagger import Tagger

from keras.optimizers import Adam

from nerds.models import NERModel
from nerds.utils import get_logger

from sklearn.model_selection import train_test_split

import os

log = get_logger()


class ElmoNER(NERModel):

    def __init__(self, entity_label=None):
        """ Build a Anago Bi-LSTM CRF model.

            Args:
                entity_label: label for single entity NER, default None
        """
        super().__init__(entity_label)
        self.key = "anago_elmo"
        # populated by fit() and load(), expected by save() and transform()
        self.preprocessor = None
        self.model = None
        self.trainer = None
        self.tagger = None


    def fit(self, X, y,
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
            num_epochs=2):
        """ Trains the NER model. Input is list of AnnotatedDocuments.

            Args:
                X list(list(str)): list of list of tokens
                y list(list(str)): list of list of BIO tags
                word_embedding_dim (int): word embedding dimensions.
                char_embedding_dim (int): character embedding dimensions.
                word_lstm_size (int): character LSTM feature extractor output dimensions.
                char_lstm_size (int): word tagger LSTM output dimensions.
                fc_dim (int): output fully-connected layer size.
                dropout (float): dropout rate.
                embeddings (numpy array): word embedding matrix.
                embeddings_file (str): path to embedding file.
                use_char (boolean): add char feature.
                use_crf (boolean): use crf as last layer.
                batch_size training batch size.
                learning_rate learning rate for Adam optimizer.
                num_epochs number of epochs of training.
        """
        if embeddings is None and (embeddings_file is None or embeddings_dim is None):
            raise ValueError("Either embeddings should be provided, or both embeddings_file and embeddings_dim should be provided, exiting.")

        log.info("Preprocessing dataset...")
        self.preprocessor = ELMoTransformer()
        self.preprocessor.fit(X, y)

        if embeddings is None:
            embeddings = load_glove(embeddings_file)
            embeddings_dim != embeddings[list(embeddings.keys())[0]].shape[0]
            embeddings = filter_embeddings(embeddings, 
                self.preprocessor._word_vocab.vocab, 
                embeddings_dim)

        log.info("Building model...")
        self.model = ELModel(
            char_embedding_dim=char_embedding_dim,
            word_embedding_dim=word_embedding_dim,
            char_lstm_size=char_lstm_size,
            word_lstm_size=word_lstm_size,
            char_vocab_size=self.preprocessor.char_vocab_size,
            word_vocab_size=self.preprocessor.word_vocab_size,
            num_labels=self.preprocessor.label_size,
            embeddings=embeddings,
            dropout=dropout)

        self.model, loss = self.model.build()
        optimizer = Adam(lr=learning_rate)
        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.summary()

        log.info('Training the model...')
        self.trainer = Trainer(self.model, preprocessor=self.preprocessor)

        x_train, x_valid, y_train, y_valid = train_test_split(X, y, 
            test_size=0.1, random_state=42)
        self.trainer.train(x_train, y_train, x_valid=x_valid, y_valid=y_valid,
            batch_size=batch_size, epochs=num_epochs)

        self.tagger = Tagger(self.model, preprocessor=self.preprocessor)

        return self


    def predict(self, X):
        """ Predicts using the NER model.

            Args:
                X list(list(str)): list of list of tokens.
            Returns:
                y list(list(str)): list of list of predicted BIO tags.
        """
        if self.tagger is None:
            raise ValueError("No tagger found, either run fit() to train or load() a trained model")

        log.info("Predicting from model...")
        ypreds = [self.tagger.predict(" ".join(x)) for x in X]
        return ypreds


    def save(self, dirpath):
        """ Saves model to local disk, given a dirpath 
        
        Args:
            dirpath (str): a directory where model artifacts will be saved.
                Model saves a weights.h5 weights file, a params.json parameter
                file, and a preprocessor.pkl preprocessor file.
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("No model artifacts to save, either run fit() to train or load() a trained model")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        weights_file = os.path.join(dirpath, "weights.h5")
        params_file = os.path.join(dirpath, "params.json")
        preprocessor_file = os.path.join(dirpath, "preprocessor.pkl")

        save_model(self.model, weights_file, params_file)
        self.preprocessor.save(preprocessor_file)


    def load(self, dirpath):
        """ Loads a trained model from local disk, given the dirpath

        Args:
            dirpath (str): a directory where model artifacts are saved.
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
        
        self.model = load_model(weights_file, params_file)
        self.preprocessor = ELMoTransformer.load(preprocessor_file)
        self.tagger = Tagger(self.model, preprocessor=self.preprocessor)

        return self


