# Dataset description

Data comes from the [Report on Bio-Entity Recognition Task at BioNLP/NLPBA 2004](http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html) page. The page describes the provenance and characteristics of the data. 

In addition, [GloVe (Global Vectors for Word Representation) vectors](https://nlp.stanford.edu/projects/glove/) are needed to run the ElmoNER model.

To make the data available for use by our example, execute the script `data_prep.sh` in the current directory. This script will create a `data` directory, and also download the GloVe vectors needed by the example.

## Entity distribution

```
  25307 DNA
   2481 RNA
  11217 cell_line
  15466 cell_type
  55117 protein
```

## Training

Our example will use the `data/train/Genia4ERtask1.iob2` file for training, and the `data/test/Genia4EReval1.iob2` file for evaluation. Both files are already in BIO format. Entity distribution shown above is for training data.

## Results

### Dictionary NER

```
              precision    recall  f1-score   support

   cell_line       0.63      0.47      0.54      1489
   cell_type       0.71      0.63      0.67      4912
     protein       0.72      0.65      0.69      9841
         DNA       0.63      0.50      0.56      2845
         RNA       0.57      0.46      0.51       305

   micro avg       0.70      0.61      0.65     19392
   macro avg       0.65      0.54      0.59     19392
weighted avg       0.70      0.61      0.65     19392
```

### CRF NER (max_iterations=100, c1=0.1, c2=0.1)

```
              precision    recall  f1-score   support

   cell_line       0.58      0.70      0.63      1489
   cell_type       0.88      0.71      0.79      4912
     protein       0.79      0.80      0.80      9841
         DNA       0.77      0.73      0.75      2845
         RNA       0.77      0.72      0.74       305

   micro avg       0.79      0.76      0.77     19392
   macro avg       0.76      0.73      0.74     19392
weighted avg       0.79      0.76      0.77     19392
```

### SpaCy NER (num_epochs=20, dropout=0.1)

```
              precision    recall  f1-score   support

   cell_line       0.56      0.76      0.65      1489
   cell_type       0.89      0.66      0.76      4912
     protein       0.78      0.84      0.81      9841
         DNA       0.77      0.76      0.77      2845
         RNA       0.77      0.76      0.77       305

   micro avg       0.78      0.78      0.78     19392
   macro avg       0.76      0.76      0.75     19392
weighted avg       0.79      0.78      0.78     19392
```

### BiLSTM-CRF NER (word_embedding_dim=100, char_embedding_dim=25, word_lstm_size=100, char_lstm_size=25, fc_dim=100, dropout=0.5, embeddings=None, use_char=True, use_crf=True, batch_size=16, learning_rate=0.001, num_epochs=10)

```
              precision    recall  f1-score   support

   cell_line       0.53      0.77      0.63      1489
   cell_type       0.88      0.71      0.78      4912
     protein       0.81      0.78      0.79      9841
         DNA       0.73      0.83      0.78      2845
         RNA       0.80      0.78      0.79       305

   micro avg       0.78      0.77      0.77     19392
   macro avg       0.75      0.77      0.76     19392
weighted avg       0.79      0.77      0.78     19392
```

### ELMo NER (word_embedding_dim=100, char_embedding_dim=25, word_lstm_size=100, char_lstm_size=25, fc_dim=100, dropout=0.5, embeddings=None, embeddings_file="glove.6B.100d.txt", batch_size=16, learning_rate=0.001, num_epochs=2)

```
              precision    recall  f1-score   support

   cell_line       0.53      0.73      0.61      1489
   cell_type       0.85      0.75      0.79      4912
     protein       0.80      0.87      0.83      9841
         DNA       0.77      0.86      0.81      2845
         RNA       0.77      0.86      0.81       305

   micro avg       0.78      0.82      0.80     19392
   macro avg       0.74      0.81      0.77     19392
weighted avg       0.79      0.82      0.80     19392
```

### Majority voting ensemble (pre-trained Dictionary NER, CRF NER, SpaCy NER, and BiLSTM-CRF NER)

```
              precision    recall  f1-score   support

   cell_line       0.67      0.70      0.69      1489
   cell_type       0.91      0.69      0.78      4912
     protein       0.83      0.77      0.80      9841
         DNA       0.83      0.74      0.78      2845
         RNA       0.81      0.73      0.77       305

   micro avg       0.84      0.74      0.78     19392
   macro avg       0.81      0.73      0.76     19392
weighted avg       0.84      0.74      0.78     19392
```
