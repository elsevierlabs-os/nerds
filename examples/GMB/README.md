# Dataset description

Annotated Corpus for Named Entity Recognition using GMB (Groningen Meaning Bank) corpus for entity classification with enhanced and popular features by Natural Language Processing applied to the data set. Downloaded from [Kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus) to `train.csv` file locally.

In addition, [GloVe (Global Vectors for Word Representation) vectors](https://nlp.stanford.edu/projects/glove/) are needed to run the ElmoNER model, please download them by running the provided `download_glove.sh` script.

## Overall number of entities

```
    699 art
    561 eve
  45058 geo
  16068 gpe
    252 nat
  36927 org
  34241 per
  26861 tim
```

## Training

We train with the full set of data, and the entire run across all the provided models can be fairly time consuming. If it is desired to keep the training time reasonable, you can train only with 5000 sentences by uncommenting lines 47-49 in `test_models.py`.

## Results

### Dictionary NER (from_dictionary=False)

```
              precision    recall  f1-score   support

         art       0.01      0.15      0.02       215
         eve       0.22      0.43      0.29       169
         geo       0.35      0.74      0.48     13724
         gpe       0.93      0.90      0.92      4850
         nat       0.27      0.53      0.36        94
         org       0.41      0.67      0.51     10884
         per       0.77      0.74      0.75     10342
         tim       0.14      0.92      0.25      8140

   micro avg       0.32      0.77      0.45     48418
   macro avg       0.39      0.64      0.45     48418
weighted avg       0.48      0.77      0.55     48418
```

### CRF NER (c1=0.1, c2=0.1, max_iter=100, featurizer=Default)

```
              precision    recall  f1-score   support

         art       0.28      0.05      0.08       215
         eve       0.54      0.33      0.41       169
         geo       0.87      0.89      0.88     13724
         gpe       0.95      0.92      0.94      4850
         nat       0.71      0.32      0.44        94
         org       0.80      0.78      0.79     10884
         per       0.88      0.88      0.88     10342
         tim       0.93      0.87      0.90      8140

   micro avg       0.87      0.85      0.86     48418
   macro avg       0.74      0.63      0.66     48418
weighted avg       0.87      0.85      0.86     48418

```

The entity types which have enough examples have good results!

### SpaCy NER (dropout=0.1, max_iter=20, batch_size=32)

```
              precision    recall  f1-score   support

         art       0.26      0.07      0.10       215
         eve       0.61      0.24      0.34       169
         geo       0.87      0.87      0.87     13724
         gpe       0.94      0.93      0.93      4850
         nat       0.87      0.28      0.42        94
         org       0.79      0.77      0.78     10884
         per       0.85      0.90      0.88     10342
         tim       0.96      0.83      0.89      8140

   micro avg       0.87      0.85      0.86     48418
   macro avg       0.77      0.61      0.65     48418
weighted avg       0.87      0.85      0.86     48418

```

### BiLSTM-CRF NER (word_embedding_dim=100, char_embedding_dim=25, word_lstm_size=100, char_lstm_size=25, fc_dim=100, dropout=0.5, embeddings=None, use_char=True, use_crf=True, batch_size=16, learning_rate=0.001, max_iter=10)

```
              precision    recall  f1-score   support

         art       0.25      0.09      0.14       215
         eve       0.37      0.29      0.33       169
         geo       0.84      0.89      0.87     13724
         gpe       0.95      0.93      0.94      4850
         nat       0.71      0.31      0.43        94
         org       0.84      0.72      0.77     10884
         per       0.87      0.90      0.89     10342
         tim       0.89      0.89      0.89      8140

   micro avg       0.86      0.85      0.86     48418
   macro avg       0.72      0.63      0.66     48418
weighted avg       0.86      0.85      0.85     48418

```

### ELMo NER (word_embedding_dim=100, char_embedding_dim=25, word_lstm_size=100, char_lstm_size=25, fc_dim=100, dropout=0.5, embeddings=None, embeddings_file="glove.6B.100d.txt", batch_size=16, learning_rate=0.001, max_iter=2)

```
              precision    recall  f1-score   support

         art       0.13      0.15      0.14       215
         eve       0.35      0.46      0.40       169
         geo       0.88      0.89      0.88     13724
         gpe       0.94      0.94      0.94      4850
         nat       0.71      0.21      0.33        94
         org       0.82      0.76      0.79     10884
         per       0.86      0.93      0.89     10342
         tim       0.91      0.88      0.90      8140

   micro avg       0.87      0.87      0.87     48418
   macro avg       0.70      0.65      0.66     48418
weighted avg       0.87      0.87      0.86     48418

```

### FLAIR NER (hidden_dim=256, embeddings=StackedEmbeddings(WordEmbeddings("glove"), CharEmbeddings()), use_crf=True, use_rnn=True, num_rnn_layers=1, dropout=0.0, word_dropout=0.05, locked_dropout=0.5, optimizer="sgd", learning_rate=0.1, batch_size=32, max_iter=10)

```
              precision    recall  f1-score   support

         art       0.00      0.00      0.00       215
         eve       0.71      0.20      0.31       169
         geo       0.84      0.91      0.87     13724
         gpe       0.95      0.92      0.94      4850
         nat       0.50      0.06      0.11        94
         org       0.85      0.67      0.75     10884
         per       0.84      0.92      0.88     10342
         tim       0.90      0.88      0.89      8140

   micro avg       0.86      0.84      0.85     48418
   macro avg       0.70      0.57      0.59     48418
weighted avg       0.86      0.84      0.85     48418

```

### Transformer NER (lang_model_family="bert", lang_model_name="bert-base-cased", max_sequence_length=128, batch_size=32, max_iter=4, learning_rate=4e-5, padding_tag="O", random_state=42)

```
              precision    recall  f1-score   support

         art       0.11      0.24      0.15        97
         eve       0.41      0.55      0.47       126
         geo       0.90      0.88      0.89     14016
         gpe       0.94      0.96      0.95      4724
         nat       0.34      0.80      0.48        40
         org       0.80      0.81      0.81     10669
         per       0.91      0.90      0.90     10402
         tim       0.89      0.93      0.91      7739

   micro avg       0.87      0.88      0.88     47813
   macro avg       0.66      0.76      0.69     47813
weighted avg       0.88      0.88      0.88     47813

```

### Majority voting ensemble (pretrained Dictionary NER, CRF NER, SpaCy NER, and BiLSTM-CRF NER)

```
              precision    recall  f1-score   support

         art       0.17      0.08      0.11       215
         eve       0.47      0.22      0.30       169
         geo       0.83      0.87      0.85     13724
         gpe       0.98      0.89      0.93      4850
         nat       0.76      0.31      0.44        94
         org       0.84      0.64      0.73     10884
         per       0.93      0.71      0.81     10342
         tim       0.90      0.86      0.88      8140

   micro avg       0.87      0.78      0.82     48418
   macro avg       0.73      0.57      0.63     48418
weighted avg       0.87      0.78      0.82     48418
```
