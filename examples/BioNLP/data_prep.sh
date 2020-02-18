#!/bin/bash
echo "Creating directories..."
mkdir data
cd data
mkdir train test

echo "Downloading training data..."
cd train
curl -O http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz
tar xvf Genia4ERtraining.tar.gz
rm Genia4ERtraining.tar.gz

echo "Downloading test data..."
cd ../test
curl -O http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz
tar xvf Genia4ERtest.tar.gz
rm Genia4ERtest.tar.gz

cd ../..
echo "Downloading GloVe embeddings..."
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -a glove.6B.zip
rm glove.6B.200d.txt glove.6B.300d.txt glove.6B.50d.txt glove.6B.zip

