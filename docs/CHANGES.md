# Improvements and Changes

## Completed

* Replace AnnotatedDocument common data format to List of List format borrowed from Anago.
* Removes dependency on NLTK
* Model
  * NERModel -- base class extends ClassifierMixin, so exposes predict() instead of transform().
  * DictionaryNER
    * similar to ExactMatchDictionaryNER except
      * takes Anago style IO
      * handles multiple classes (as well as single class as special case)
      * can handle Anago style input via fit(X, y, combine_tokens=True) and dictionary style input via fit(X, y, combine_tokens=False).
  * CrfNER
    * similar to CRF except
      * takes Anago style IO (native IO format to wrapped model sklearn_crfsuite.CRF)
      * replaces dependency on nltk.tokenize_pos() to SpaCy
      * allows features to be directly passed to fit() using is_featurized=False.
  * SpacyNER
    * similar to SpacyStatisticalNER, except
      * takes Anago style IO
      * more robust to large data sizes, uses mini-batches for training
  * BiLstmCrfNER
    * similar to BidirectionalLSTM except
      * takes Anago style IO
      * works against most recent Anago API changes
      * does not give timestep size errors
  * ElmoNER
    * New, available in Anago DEV repo, same API as Anago's BiLSTMCRF
  * EnsembleNER
    * simpler interface 
    * weights from each classifier
    * fit() and predict() can use multiple parallel jobs (`n_jobs`).
* Utils
  * Thin wrapper over anago's `load_data_and_labels`
  * `flatten_list` and `unflatten_list` to convert between `list(list(str))` produced by NERDS models and `list(str)` required by `sklearn`, scikit-learn metrics can be used.
  * `tokens_to_spans` and `spans_to_tokens` -- utility functions to convert between sentence and span format (used by the other 2 of 5 provided models) from and to BIO format.
* Converters
  * Converter from BRAT (.txt and .ann) to IOB format
* Miscellaneous
  * replaced deprecated sklearn.external.joblib -> joblib
  * True Scikit-Learn interoperability -- moving parameters to constructor has resulted in estimators being serializable and usable via joblib.Parallel. However, `sklearn.utils.check_estimator` still fails, most likely because the parameters to `fit()` and `predict()` are `list(list(str))` rather than `list(str)`.

## Planned

* Convert Docs to numpy docstring format -- https://numpydoc.readthedocs.io/en/latest/format.html
* BERT Transformer based NER
* FLAIR based NER

