import joblib
import nerds
import os
import pandas as pd
import random
import torch

from simpletransformers.ner.ner_model import NERModel as ST_NERModel

from nerds.models import NERModel
from nerds.utils import (flatten_list, get_logger, 
                         write_param_file, get_labels_from_data)

from sklearn.model_selection import train_test_split

log = get_logger()

class TransformerNER(NERModel):

    def __init__(self,
            lang_model_family="bert",
            lang_model_name="bert-base-cased",
            model_dir="models",
            max_sequence_length=128,
            batch_size=32,
            max_iter=4,
            learning_rate=4e-5,
            padding_tag="O",
            random_state=42):
        """ Construct a Transformer NER model. This is a generic front-end
            NER class that can work with multiple Transformer architectures.

            Parameters
            ----------
            model_dir : str, optional, default "./models"
                the directory to which model artifacts will be written out to.
            lang_model_family : str, optional, default "bert"
                the Transformer Language Model (LM) Family to use. Following LM
                families are supported - BERT, RoBERTa, DistilBERT, CamemBERT,
                and XLM-RoBERTa.
            lang_model_name : str, optional, default "bert-base-cased"
                name of the pre-trained LM to use.
            model_dir : string, optional, default "models"
                directory path to folder where model artifacts will be written
            max_sequence_length : int, optional, default 128
                maximum number of tokens in each input sentence. Note that
                because of word-piece tokenization, this is not the actual
                number of tokens, but the number of word-pieces.
            batch_size : int, optional, default 32
                the batch size to use during training and prediction.
            max_iter : int, optional, default 4
                the number of epochs to train the model.
            learning_rate: float, optional, default 4e-5
                learning rate for Adam optimizer.
            padding_tag : str, default "O"
                padding tag to use when number of predicted tags is smaller
                than the number of label tags because of word-piece tokenization.
                Default value ensures that you won't have to align, at the cost
                of a drop in reported performance. You should choose a non-default
                value and align using nerds.utils.align_labels_and_predictions().
            random_state : int, optional, default 42
                random state to set.

            Attributes
            ----------
            model_ : reference to the SimpleTranformers NERModel object.
            model_args_ : flat dictionary composed of values from constructor.
            labels_ : list of labels to use in model.
        """
        super().__init__()
        self.model_dir = model_dir
        self.lang_model_family = lang_model_family
        self.lang_model_name = lang_model_name
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.padding_tag = padding_tag
        self.random_state = random_state
        # attributes
        self.model_ = None
        self.model_args_ = None
        self.labels_ = None


    def fit(self, X, y):
        """ Trains the NER model. Input is list of list of tokens and tags.

            Parameters
            ----------
            X : list(list(str))
                list of list of tokens
            y : list(list(str))
                list of list of BIO tags.

            Returns
            -------
            self
        """
        self._build_model_args()
        self.labels_ = get_labels_from_data(y)
        self.model_ = ST_NERModel(
            self.lang_model_family,
            self.lang_model_name,
            labels=self.labels_,
            use_cuda=torch.cuda.is_available(),
            args=self.model_args_)
        
        os.makedirs(self.model_dir, exist_ok=True)

        Xtrain, Xval, ytrain, yval = train_test_split(X, y, 
            test_size=0.1, random_state=self.random_state)
        train_df = self._build_dataframe_from_data_labels(Xtrain, ytrain)
        eval_df = self._build_dataframe_from_data_labels(Xval, yval)
        self.model_.train_model(train_df, eval_df=eval_df)
        return self


    def predict(self, X):
        """ Predicts using the NER model

            Parameters
            ----------
            X : list(list(str))
                list of list of tokens

            Returns
            -------
            y : list(list(str))
                list of list of predicted BIO tags.
        """
        if self.model_ is None:
            raise ValueError("No model found, either run fit() to train or load() to load a trained model.")

        predictions, _ = self.model_.predict([" ".join(toks) for toks in X])
        # predictions are list of {token:tag} dicts
        predictions = [[tag for token_tag_dict in prediction 
                            for (token, tag) in token_tag_dict.items()] 
                            for prediction in predictions]
        # handle possible truncation of prediction (and subsequent mismatch
        # with labels) because of too long token list.
        predictions_a = []
        for prediction, tokens in zip(predictions, X):
            if len(prediction) < len(tokens):
                prediction.extend(
                    [self.padding_tag] * (len(tokens) - len(prediction)))
            predictions_a.append(prediction)
        return predictions_a


    def save(self, dirpath=None):
        """ This is a no-op for this NER, model artifacts are saved automatically
            after every epoch.

            Parameters
            ----------
            dirpath : str, optional
                directory to which the param file will be written. If not 
                specified, it will use the folder specified by the model's 
                output_dir.

            Returns
            -------
            None
        """
        if self.model_ is None:
            raise ValueError("No model artifacts to save, either run fit() to train or load() pretrained model.")
        if dirpath is None:
            self._build_model_args()
            dirpath = self.model_args_["output_dir"]
        attr_dict = {
            "model_args": self.model_args_,
            "labels": self.labels_
        }
        joblib.dump(attr_dict, os.path.join(dirpath, "attr_dict.pkl"))
        write_param_file(self.get_params(), os.path.join(dirpath, "params.yaml"))


    def load(self, dirpath=None):
        """ Loads a trained model from specified folder on disk.

            Parameters
            ----------
            dirpath : str, optional
                directory from which model artifacts should be loaded. If
                not provided, uses the model_args_["output_dir].

            Returns
            -------
            self
        """
        if dirpath is None:
            self._build_model_args()
            dirpath = self.model_args_["output_dir"]
        if not os.path.exists(dirpath):
            raise ValueError("Model directory not found: {:s}".format(dirpath))
        attr_dict = joblib.load(os.path.join(dirpath, "attr_dict.pkl"))
        self.model_args_ = attr_dict["model_args"]
        self.labels_ = attr_dict["labels"]
        self.model_ = ST_NERModel(self.lang_model_family, dirpath,
            args=self.model_args_,
            labels=self.labels_,
            use_cuda=torch.cuda.is_available())
        return self


    def _build_model_args(self):
        """ Builds the model_arg dictionary from constructor parameters.

            Parameters
            ----------
            none

            Returns
            -------
            none
        """
        self.model_args_ = {
            "output_dir": os.path.join(self.model_dir, "outputs"),
            "cache_dir": os.path.join(self.model_dir, "cache"),
            "fp16": False,
            "fp16_opt_level": "01",
            "max_seq_length": self.max_sequence_length,
            "train_batch_size": self.batch_size,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": self.max_iter,
            "weight_decay": 0,
            "learning_rate": self.learning_rate,
            "adam_epsilon": 1e-8,
            "warmup_ratio": 0.06,
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "eval_batch_size": self.batch_size,
            "logging_steps": 50,
            "save_steps": 2000,
            "overwrite_output_dir": True,
            "reprocess_input_data": True,
            "evaluate_during_training": True,
            "process_count": os.cpu_count() - 2 if os.cpu_count() > 2 else 1,
            "n_gpu": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


    def _build_dataframe_from_data_labels(self, data, labels):
        """ Builds Pandas dataframe from data and labels.

            Parameters
            ----------
            data : list(list(str))
                list of list of tokens
            labels : list(list(str))
                list of list of tags

            Returns
            -------
            Pandas DataFrame with columns (sentence_id, words, labels).
        """
        columns = ["sentence_id", "words", "labels"]
        recs = []
        for sid, (tokens, tags) in enumerate(zip(data, labels)):
            for token, tag in zip(tokens, tags):
                recs.append((sid, token, tag))
        data_df = pd.DataFrame.from_records(recs, columns=columns)
        return data_df

