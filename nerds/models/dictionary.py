from nerds.models import NERModel
from nerds.utils import get_logger

import ahocorasick
import joblib
import os

log = get_logger()

class DictionaryNER(NERModel):

    def __init__(self, entity_label=None):
        super().__init__(entity_label)
        self.key = "aho-corasick-dict-ner"
        self.model = None


    def fit(self, X, y,
            combine_tokens=True):
        """ Build dictionary of phrases of different entity types.

            Args:
                X (list(list(str))): list of list of tokens or phrases.
                combine_tokens (bool, default True): if combine tokens
                    is True, then input is tokenized as individual words.
                    This would be the expected format if the input came
                    directly from a training set.

                        X = [..., [..., "New", "York", "City", ...], ...]
                        y = [..., [..., "B-loc", "I-loc", "I-loc", ...], ...]
                    
                    If combine_tokens is False, then phrases have been 
                    pre-chunked. This would be the expected format if the 
                    input came from a third party dictionary.
                    
                        X = [..., [..., "New York City", ...], ...]
                        y = [..., [..., "loc", ...], ...]

                y (list(list(str))): list of list of labels. If combine_tokens
                    is True, then labels are IOB tags. If combine_tokens is False,
                    labels are entity types (without leading B and I), and without
                    any O labels.

                combine_tokens (bool, default True): if True, input comes from
                    standard training set, and an additional step to chunk 
                    phrases is needed. If False, input comes from a dictionary
                    with phrase chunking already done.
        """
        self.model = ahocorasick.Automaton()

        if combine_tokens:
            for idx, (tokens, labels) in enumerate(zip(X, y)):
                phrase_tokens, phrase_labels = self._combine_tokens(tokens, labels)
                for phrase, label in zip(phrase_tokens, phrase_labels):
                    self.model.add_word(phrase, (label, phrase))
        else:
            for token, label in zip(X, y):
                self.model.add_word(token, (label, token))
        self.model.make_automaton()

        return self


    def predict(self, X):
        if self.model is None:
            raise ValueError("No model found, use fit() to train or load() pretrained.")
        
        predictions = []
        for tokens in X:
            sent = " ".join(tokens)
            matched_phrases = []
            for end_index, (tag, phrase) in self.model.iter(sent):
                start_index = end_index - len(phrase) + 1
                # filter out spurious matches on partial words
                self._add_if_not_spurious_match(
                    start_index, end_index, tag, sent, matched_phrases)
            # remove subsumed phrases
            longest_phrases = self._remove_subsumed_matches(matched_phrases, 1)
            # convert longest matches to IOB format
            pred = self._convert_matches_to_iob_tags(tokens, longest_phrases)
            predictions.append(pred)

        return predictions


    def save(self, dirpath=None):
        if self.model is None:
            raise ValueError("No model found, use fit() to train or load() pretrained.")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        log.info("Saving model...")
        model_file = os.path.join(dirpath, "dictionary-ner.pkl")
        joblib.dump(self.model, model_file)


    def load(self, dirpath=None):
        model_file = os.path.join(dirpath, "dictionary-ner.pkl")
        if not os.path.exists(model_file):
            raise ValueError("Saved model {:s} not found.".format(model_file))

        self.model = joblib.load(model_file)
        return self


    def _combine_tokens(self, tokens, labels):
        """ Combine consecutive word tokens for some given entity type
            to create phrase tokens.

            Args:
                tokens (list(str)): a list of tokens representing a sentence.
                labels (list(str)): a list of IOB tags for sentence.

            Returns:
                phrases (list(str)): list of multi-word phrases.
                phrase_labels (list(str)): list of phrase entity types.
        """
        phrases, phrase_labels = [], []
        phrase_tokens = []
        for token, label in zip(tokens, labels):
            if label == "O" and len(phrase_tokens) > 0:
                phrases.append(" ".join(phrase_tokens))
                phrase_labels.append(prev_label.split("-")[-1])
                phrase_tokens = []
            if label.startswith("B-"):
                phrase_tokens = [token]
            if label.startswith("I-"):
                phrase_tokens.append(token)
            prev_label = label

        if len(phrase_tokens) > 0:
            phrases.append(" ".join(phrase_tokens))
            phrase_labels.append(prev_label.split("-")[-1])

        return phrases, phrase_labels


    def _add_if_not_spurious_match(self, start_index, end_index, tag,
            sentence, matched_phrases):
        """ Aho-Corasick can match across word boundaries, and often matches
            parts of longer words. This function checks to make sure any
            matches it reports don't do so.

            Args:
                start_index (int): reported start index of matched phrase.
                end_index (int): reported end index of matched phrase.
                tag (str): the entity type.
                sentence (str): the sentence in which match occurs.
                matched_phrases (list(str)): list of matched phrases, updated
                    in place by function.
        """
        if start_index == 0:
            if end_index < len(sentence):
                if sentence[end_index + 1] == " ":
                    matched_phrases.append((start_index, end_index + 1, tag))
        elif end_index + 1 == len(sentence):
            if start_index > 0:
                if sentence[start_index - 1] == " ":
                    matched_phrases.append((start_index, end_index + 1, tag))
        else:
            if sentence[start_index - 1] == " " and sentence[end_index + 1] == " ":
                matched_phrases.append((start_index, end_index + 1, tag))


    def _remove_subsumed_matches(self, matched_phrases, k):
        """ Remove matches that are subsumed in longer matches. This ensures
            that the matches reported are the longest ones. Function works as
            follows -- we sort the list by longest phrase first, and then check
            to see if any shorter phrases are contained within the longest one 
            and remove them if so. We then recursively apply this same function
            to the remaining list, moving one position down for the longest
            phrase to match against. Function stops when we have seen all the
            phrases.

            Args:
                matched_phrases (list((start, end, iob_tag))): list of 
                    matched phrase tuples.
                k (int): starting position.

            Returns:
                matched_phrases: without the shorter subsumed phrase tuples.
        """
        if k >= len(matched_phrases):
            return matched_phrases
        sorted_matches = sorted(matched_phrases, key=lambda x: x[1]-x[0], reverse=True)
        longest_matches = sorted_matches[0:k]
        ref_offsets = (longest_matches[-1][0], longest_matches[-1][1])
        for phrase in sorted_matches[k:]:
            if phrase[0] >= ref_offsets[0] and phrase[1] <= ref_offsets[1]:
                continue
            else:
                longest_matches.append(phrase)
        return self._remove_subsumed_matches(longest_matches, k+1)


    def _convert_matches_to_iob_tags(self, tokens, matched_phrases):
        """ Merges the longest matches with the original tokens to 
            produce a list of IOB tags for the sentence.

            Args:
                tokens (list(str)): list of tokens for the sentence.
                matched_phrase (list((start, end, tag))): list of longest
                    matched phrase tuples.

            Returns:
                iob_tags (list(str)): list of IOB tags, each tag 
                    corresponds to a word token.
        """
        iob_tags = []
        curr_offset = 0
        prev_label = "O"
        for token in tokens:
            start_offset = curr_offset
            end_offset = start_offset + len(token)
            token_matched = False
            matched_label = None
            for phrase_start, phrase_end, phrase_label in matched_phrases:
                if start_offset >= phrase_start and end_offset <= phrase_end:
                    token_matched = True
                    matched_label = phrase_label
                    break
            if token_matched:
                iob_tags.append(
                    "I-" + phrase_label if prev_label == phrase_label 
                    else "B-" + phrase_label)
                prev_label = phrase_label
            else:
                iob_tags.append("O")            
                prev_label = "O"
            curr_offset = end_offset + 1
        return iob_tags
