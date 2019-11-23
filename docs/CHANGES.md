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
    * New, available in Anago, same API as Anago's BiLSTMCRF
  * EnsembleNER
    * simpler interface 
    * weights from each classifier
    * joblib.Parallel -- improve?
* Utils
  * Thin wrapper over anago's `load_data_and_labels`
  * Converter for output so scikit-learn metrics can be used.

* Other stuff
  * remove deprecated sklearn.external.joblib -> joblib

## Planned

* Scikit-Learn interoperability.
* BERT Transformer based NER
