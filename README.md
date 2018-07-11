# Emotion Prediction GRU

Emotion prediction of Tweets using a uni-directional GRU.

## Word Embeddings

Word embeddings were generated using word2vec in gensim and given a size of 100. Because of the nature of language 
on Twitter infrequent words were left in with min count=1.

## The Lexicon

The NRC Hashtag Emotion Lexicon contains 16,862 words with a corresponding emotion
(anger, anticipation, disgust, fear, joy, sadness, surprise, or trust) and value (indicating the
strength of the corresponding emotion). The lexicon was automatically generated from tweets containing
emotion word hashtags, such as #happy. (Mohammad and Kiritchenko, 2015; Mohammad,
2012) In order to be consistent with the data, only words tagged as anger, disgust, fear, joy, sadness,
or surprise were used in the model.

## The Model

16,840 tweets were used for training data and 2,105 tweets were used as test data with a batch
size of 421. Words and emotions from the data and lexicon were recoded into integer representations
before being used in the model. The maximum number of timesteps was set to the maximum
tweet length of 31. Dropout was applied during training with a dropout rate of 0.95.

### Model Comparisons

Model | precision | recall | f1
--- | --- | --- | ---
GRU | **54.73** | **54.73** | **54.73**
Bidirectional GRU | 53.83 | 53.83 | 53.83
Stacked GRU | 53.82 | 53.82 | 53.82
Stacked bidirectional GRU | 53.11 | 53.11 | 53.11
LSTM | 53.87 | 53.87 | 53.87
Bidriectional LSTM | 53.78 | 53.78 | 53.78
Stacked LSTM | 53.82 | 53.82 | 53.82

Stacked RNNs had 3 layers per direction.

## References

Saif M Mohammad and Svetlana Kiritchenko. 2015. Using hashtags to capture fine emotion categories from tweets. 
*Computational Intelligence*, 31(2):301–326.  

Saif M Mohammad. 2012. # emotional tweets. In *Proceedings of the First Joint Conference on Lexical and Computational 
Semantics-Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth 
International Workshop on Semantic Evaluation*, pages 246–255. Association for Computational Linguistics.
