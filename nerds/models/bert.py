import joblib
import numpy as np
import os
import random
import time
import torch

from nerds.models import NERModel
from nerds.utils import flatten_list, get_logger, write_param_file

from transformers import AdamW
from transformers import BertForTokenClassification, BertTokenizer
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


log = get_logger()

class BertNER(NERModel):

    def __init__(self,
            lang_model="bert-base-cased",
            max_sequence_length=128,
            learning_rate=2e-5,
            batch_size=32,
            max_iter=4,
            padding_tag="O",
            verbose=False,
            random_state=42):
        """ Construct a BERT NER model. Uses a pretrained BERT language model
            and a Fine tuning model for NER is provided by the HuggingFace 
            transformers library.

            NOTE: this is an experimental NER that did not perform very well, and
            is only here for reference purposes. It has been superseded by the
            TransformerNER model, which offers the same functionality (and improved
            performance) not only with BERT as the underlying language model (as this 
            one does), but allows other BERT-like language model backends as well.

            Parameters
            ----------
            lang_model : str, optional, default "bert-base-cased"
                pre-trained BERT language model to use.
            max_sequence_length : int, optional, default 128
                maximum sequence length in tokens for input sentences. Shorter
                sentences will be right padded and longer sentences will be
                truncated.
            learning_rate : float, optional, default 2e-5
                learning rate for the ADAM optimizer.
            batch_size : int, optional, default 32
                batch size to use for training.
            max_iter : int, default 4
                number of epochs to fine tune.
            padding_tag : str, default "O"
                tag to pad predictions with if len(tokens) > len(predicted_tags).
            verbose : bool, optional, default False
                whether to display log messages on console.
            random_state : int, optional, default 42
                random state to set for repeatable results

            Attributes
            ----------
            model_ : reference to underlying BertForTokenClassification object.
            tokenizer_ : reference to underlying BertTokenizer object.
            label2id_ : mapping from string labels to internal int ids.
            id2label_ : mapping from internal int label ids to string labels.
            train_losses_ : list(float) of per epoch training losses.
            val_accs_ : list(float) of per epoch validation accuracies.
            special_tokens_ : set of tokenizer special tokens.
        """
        super().__init__()
        # model parameters
        self.lang_model = lang_model
        self.max_sequence_length = max_sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.padding_tag = padding_tag
        self.verbose = verbose
        self.random_state = random_state
        self._set_random_state(random_state)
        # model attributes
        self.model_ = None
        self.tokenizer_ = None
        self.label2id_ = None
        self.id2label_ = None
        self.train_losses_ = None
        self.val_accs_ = None
        self.special_tokens_ = None
        # hidden variables
        self._pad_label_id = torch.nn.CrossEntropyLoss().ignore_index
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
        log.info("Converting data and labels to features...")
        self.label2id_, self.id2label_ = self._build_label_id_mappings(y)

        Xtrain, Xval, ytrain, yval = train_test_split(
            X, y, test_size=0.1, random_state=self.random_state)

        self.tokenizer_ = BertTokenizer.from_pretrained(
            self.lang_model, do_basic_tokenize=False)
        self.special_tokens_ = set([
            self.tokenizer_.pad_token, self.tokenizer_.unk_token, 
            self.tokenizer_.sep_token, self.tokenizer_.cls_token])

        train_feats = self._data_labels_to_features(Xtrain, ytrain)
        train_loader = self._create_dataloader(train_feats, "random")
        val_feats = self._data_labels_to_features(Xval, yval)
        val_loader = self._create_dataloader(val_feats, "sequential")

        log.info("Building model...")
        self.model_ = BertForTokenClassification.from_pretrained(self.lang_model,
            num_labels=len(self.label2id_),
            output_attentions=False,
            output_hidden_states=False)
        self.model_.to(self._device)

        total_steps = len(train_loader) * self.max_iter
        optimizer = AdamW(self.model_.parameters(), lr=self.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=0, num_training_steps=total_steps)

        self.train_losses_, self.val_accs_ = [], []
        for epoch in range(self.max_iter):
            
            log.info("==== Epoch {:d}/{:d}".format(epoch + 1, self.max_iter))
            log.info("Training...")
            t0 = time.time()
            total_loss = 0
            self.model_.train()

            for step, batch in enumerate(train_loader):
                if step % 100 == 0:
                    elapsed = time.time() - t0
                    log.info("  Batch {:d} of {:d}, elapsed: {:.3f}s".format(
                        step + 1, len(train_loader), elapsed))
                b_input_ids = batch[0].to(self._device)
                b_attention_mask = batch[1].to(self._device)
                b_token_type_ids = batch[2].to(self._device)
                b_label_ids = batch[3].to(self._device)

                self.model_.zero_grad()
                outputs = self.model_(b_input_ids,
                    attention_mask=b_attention_mask, 
                    token_type_ids=b_token_type_ids,
                    labels=b_label_ids)

                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            self.train_losses_.append(avg_train_loss)

            log.info("  Average training loss: {:.3f}".format(avg_train_loss))
            log.info("  Training epoch took: {:.3f}s".format(time.time() - t0))

            log.info("Validation...")
            t0 = time.time()
            self.model_.eval()

            val_acc, val_steps = 0, 0
            for batch in val_loader:
                batch = tuple(b.to(self._device) for b in batch)
                b_input_ids, b_attention_mask, b_token_type_ids, b_label_ids, _ = batch
                with torch.no_grad():
                    outputs = self.model_(b_input_ids,
                        attention_mask=b_attention_mask,
                        token_type_ids=b_token_type_ids)
                    logits = outputs[0].detach().cpu().numpy()
                    b_preds = np.argmax(logits, axis=-1).flatten()
                    b_labels = b_label_ids.detach().cpu().numpy().flatten()
                    b_val_acc = accuracy_score(b_preds, b_labels)
                    val_acc += b_val_acc
                    val_steps += 1

                val_acc = val_acc / val_steps

            self.val_accs_.append(val_acc)
            log.info("  Accuracy: {:.3f}".format(val_acc))
            log.info("  Validation took {:.3f}s".format(time.time() - t0))

        log.info("==== Training complete ====")
        return self


    def predict(self, X):
        """ Predicts using the NER model. Note that because of the
            way BERT re-tokenizes incoming tokens to word-pieces, it
            is possible that some incoming tokens may not be presented
            to the model for NER tagging, and hence the list of predicted
            tags will padded with a pseudo-tag (default 'O'). If you chose
            a different pseudo-tag, you will need to re-align labels and
            predictions using nerds.utils.align_lists().

            Parameters
            ----------
            X : list(list(str))
                list of list of tokens

            Returns
            -------
            y : list(list(str))
                list of list of predicted BIO tags.
        """
        if self.model_ is None or self.tokenizer_ is None:
            raise ValueError("No model and/or tokenizer found, either run fit() to train or load() to load a trained model.")

        log.info("Converting data to features...")
        test_feats = self._data_labels_to_features(X, None)
        test_loader = self._create_dataloader(test_feats, "sequential")

        log.info("Predicting from model...")
        predictions = []
        self.model_.eval()
        for batch in test_loader:
            batch = tuple(b.to(self._device) for b in batch)
            b_input_ids, b_attention_mask, b_token_type_ids, b_ids = batch
            with torch.no_grad():
                outputs = self.model_(b_input_ids,
                    attention_mask=b_attention_mask,
                    token_type_ids=b_token_type_ids)
                logits = outputs[0].detach().cpu().numpy()
                b_pred_ids = np.argmax(logits, axis=-1)
                b_ids = b_ids.detach().cpu().numpy()
                b_id_min, b_id_max = b_ids[0], b_ids[-1]
                b_X = X[b_id_min : b_id_max + 1]
                predictions.extend(self._align_predictions(b_X, b_pred_ids))

        return predictions


    def save(self, dirpath):
        """ Saves model and related artifacts to specified folder on disk

            Parameters
            ----------
            dirpath : str
                a directory where model artifacts are to be saved. Artifacts for
                this NER are the HuggingFace model and tokenizer, a pickled file
                containing the label-to-id and id-to-label mappings, and the NER
                configuration YAML file.

            Returns
            -------
            None
        """
        if self.model_ is None or self.tokenizer_ is None:
            raise ValueError("No model artifacts to save, either run fit() to train or load() pretrained model.")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        self.model_.save_pretrained(dirpath)
        self.tokenizer_.save_pretrained(dirpath)
        label_map = {
            "label2id": self.label2id_, 
            "id2label": self.id2label_,
            "special_tokens": self.special_tokens_
        }
        joblib.dump(label_map, os.path.join(dirpath, "label_mappings.pkl"))
        write_param_file(self.get_params(), os.path.join(dirpath, "params.yaml"))


    def load(self, dirpath):
        """ Loads a trained model from specified folder on disk.

            Parameters
            ----------
            dirpath : str
                directory from which model artifacts should be loaded

            Returns
            -------
            self
        """
        if not os.path.exists(dirpath):
            raise ValueError("Model directory not found: {:s}".format(dirpath))

        label_mappings = joblib.load(os.path.join(dirpath, "label_mappings.pkl"))
        self.label2id_ = label_mappings["label2id"]
        self.id2label_ = label_mappings["id2label"]
        self.special_tokens_ = label_mappings["special_tokens"]
        self.model_ = BertForTokenClassification.from_pretrained(dirpath,
            num_labels=len(self.label2id_),
            output_attentions=False,
            output_hidden_states=False)
        self.model_.to(self._device)
        self.tokenizer_ = BertTokenizer.from_pretrained(dirpath, 
            do_basic_tokenize=False)

        return self


    def _set_random_state(self, seed):
        """ Set the random seed for reproducible results.

            Parameters
            ----------
            seed : int
                a numeric random seed.

            Returns
            -------
            None
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def _build_label_id_mappings(self, labels):
        """ Build label (string) to label_id (int) mappings

            Parameters
            ----------
            labels : list(list(str))
                labels as provided by the utils.load_data_and_labels() function.

            Returns
            -------
            label2id, id2label
        """
        label2id = {l:i for i, l in 
            enumerate(sorted(list(set(flatten_list(labels, strip_prefix=False)))))}
        id2label = {v:k for k, v in label2id.items()}
        return label2id, id2label


    def _data_labels_to_features(self, data, labels):
        """ Convert data and labels from utils.load_data_and_labels() function
            to list of features needed by the HuggingFace BertForTokenClassification
            object.

            Parameters
            ----------
            data : list(list(str))
                list of list of input tokens
            labels : list(list(str)), can be None.
                list of list of input BIO tags.

            Returns
            -------
            input_ids : list(list(int))
                list of zero-padded fixed length token_ids.
            attention_mask : list(list(int))
                mask to avoid performing attention on padding tokens.
            token_type_ids : list(list(int))
                segment token indices, all zero since we consider single sequence.
            label_ids : list(list(int)) or None
                list of zero-padded fixed length label_ids. Set to None if 
                labels parameter is None.
        """
        input_ids, attention_mask, token_type_ids, label_ids = [], [], [], []
        # if labels is None (not supplied), then replace with pseudo labels
        labels_supplied = True
        if labels is None:
            labels_supplied = False
            labels = []
            for tokens in data:
                labels.append(["O"] * len(tokens))

        # input is (list(list(str)), list(list(str)))
        # format of input is: [CLS] sentence [SEP]
        for i, (tokens, tags) in enumerate(zip(data, labels)):
            tokens_sent, tags_sent = [], []
            for token, tag in zip(tokens, tags):
                subwords = self.tokenizer_.tokenize(token)
                if len(subwords) == 0:
                    tokens_sent.append(token)
                else:
                    tokens_sent.extend(subwords)
                tags_sent.append(self.label2id_[tag])
                if len(subwords) > 1:
                    tags_sent.extend([self._pad_label_id] * (len(subwords) - 1))
                # if len(subwords) > 1:
                #     # repeat tag for all subwords following the specified word, see
                #     # https://github.com/google-research/bert/issues/646#issuecomment-519868110
                #     if tag.startswith("B-"):
                #         tag = tag.replace("B-", "I-")
                #     tags_sent.extend([self.label2id_[tag]] * (len(subwords) - 1))

            # truncate to max_sequence_length - 2 (account for special tokens CLS and SEP)
            tokens_sent = tokens_sent[0:self.max_sequence_length - 2]
            tags_sent = tags_sent[0:self.max_sequence_length - 2]
            
            # prepend [CLS] and append [SEP]
            tokens_sent = [self.tokenizer_.cls_token] + tokens_sent + [self.tokenizer_.sep_token]
            tags_sent = [self._pad_label_id] + tags_sent + [self._pad_label_id]

            # pad upto the max_sequence_length - 2 (account for special tokens CLS and SEP)
            tokens_to_pad = self.max_sequence_length - len(tokens_sent)
            tokens_sent.extend([self.tokenizer_.pad_token] * tokens_to_pad)
            tags_sent.extend([self._pad_label_id] * tokens_to_pad)
            
            # feature: input_ids
            input_ids.append(self.tokenizer_.convert_tokens_to_ids(tokens_sent))
            # feature: attention_mask
            attention_mask.append([0 if t == self.tokenizer_.pad_token else 1 for t in tokens_sent])
            # feature: token_type_ids
            token_type_ids.append([0] * self.max_sequence_length)
            # feature: label_ids
            label_ids.append(tags_sent)

            if self.verbose and i < 5:
                log.info("row[{:d}].features:".format(i))
                log.info("  input_tokens:", tokens_sent)
                log.info("  input_ids:", input_ids[i])
                log.info("  attention_mask:", attention_mask[i])
                log.info("  token_type_ids:", token_type_ids[i])
                log.info("  label_ids:", label_ids[i])

        if labels_supplied:
            return input_ids, attention_mask, token_type_ids, label_ids
        else:
            return input_ids, attention_mask, token_type_ids, None


    def _create_dataloader(self, features, sampling):
        """ Converts features to Torch DataLoader for different data splits.

            Parameters
            ----------
            features : (input_ids, attention_mask, token_type_ids, label_ids)
            sampling : "random" (training) or "sequential" (everything else)

            Returns
            -------
            dataloader : reference to Torch DataLoader.
        """
        input_ids, attention_mask, token_type_ids, label_ids = features
        # convert to torch tensors
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.long)
        ids_t = torch.tensor(np.arange(len(token_type_ids)), dtype=torch.long)

        # wrap tensors into dataset
        if label_ids is not None:
            label_ids_t = torch.tensor(label_ids, dtype=torch.long)
            dataset = TensorDataset(input_ids_t, attention_mask_t, 
                token_type_ids_t, label_ids_t, ids_t)
        else:
            dataset = TensorDataset(input_ids_t, attention_mask_t, 
                token_type_ids_t, ids_t)

        # wrap dataset into dataloader and return dataloader
        if sampling == "random":
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
        return dataloader


    def _align_predictions(self, data, pred_ids):
        """ Align internal predictions from model that are aligned to
            wordpieces with external labels that are aligned to tokens.

            Parameters
            ----------
            data : list(list(str))
                list of jagged list of input tokens.
            pred_ids : list(list(long))
                list of same size list of prediction ids.

            Returns
            -------
            predictions : list(list(str))
                list of list of predictions aligned to input tokens
                and using same class names as input labels. 
        """
        data_a, preds_a = [], []
        for tokens, pred_tag_ids in zip(data, pred_ids):
            tokens_x = []
            for token in tokens:
                tokens_x.extend(self.tokenizer_.tokenize(token))
            tokens_r, preds_r = [], []
            for t, p in zip(tokens_x, pred_tag_ids):
                if t in self.special_tokens_:
                    continue
                if t.startswith("##"):
                    tokens_r[-1] = tokens_r[-1] + t[2:]
                else:
                    tokens_r.append(t)
                    preds_r.append(self.id2label_[p])

            if len(tokens_r) < len(tokens):
                # pad any truncated sentences with [PAD]/O
                num_pad_tokens = len(tokens) - len(tokens_r)
                tokens_r.extend([self.tokenizer_.pad_token] * num_pad_tokens)
                preds_r.extend([self.padding_tag] * num_pad_tokens)

            data_a.append(tokens_r)
            preds_a.append(preds_r)

        return preds_a

