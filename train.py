from enum import Enum
import os
import sys
import string
import gensim

import numpy as np
import tensorflow as tf

from config import DefaultConfig
from model import Model, Phase
from numberer import Numberer
from sklearn.metrics import precision_score, recall_score, f1_score

def read_lexicon(filename):
	with open(filename, "r") as f:
		lex = {}

		for line in f:
			parts = line.split('\t')

			if len(parts) == 3:
				iden = parts[0].replace(":", "")
				emotion = parts[2].replace(":: ", "").rstrip()

				tag = {}

				tweet = parts[1]
				tweet = tweet.split()
				for i in range(0, len(tweet)):
					word = tweet[i]
					word = "".join(l for l in word if l not in string.punctuation)
					word = word.lower()
					tweet[i] = word

				tweet = list(filter(None, tweet))

				tag[emotion] = tweet

				lex[iden] = tag

		return lex

def read_nrc_lexicon(filename):
	with open(filename, "r") as f:
		nrc_lex = {}

		for line in f:
			parts = line.split('\t')
			word = parts[1]
			emotion = parts[0]
			score = parts[2]

			if word in nrc_lex:
				emotions = nrc_lex[word]
			else:
				emotions = {}
			
			emotions[emotion] = score
			nrc_lex[word] = emotions

		return nrc_lex

def recode_lexicon(lexicon, words, emotions, train=True):
    int_lex = []

    for (iden, tags) in lexicon.items():
    	for (emotion, tweet) in tags.items():

    		for i in range(0, len(tweet)):
    			word = tweet[i]
    			tweet[i] = words.number(word, train)

    		int_lex.append((tweet, emotions.number(emotion, train)))

    return int_lex

def recode_nrc_lexicon(lexicon, words, emotions, train=False):
	int_nrc = {}

	for (word, emtns) in lexicon.items():
		
		int_emotions = []
		for (emotion, score) in emtns.items():
			int_emotions.append((emotions.number(emotion, train), score))

		int_nrc[words.number(word, train)] = int_emotions

	return int_nrc


def generate_instances(
        data,
        lexicon,
        max_emotions, # 6
        max_timesteps, # 31
        batch_size):
    n_batches = len(data) // batch_size

    emotions = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_emotions),
        dtype=np.int32)
    lengths = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int32)
    tweets = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps),
        dtype=np.int32)
    nrc = np.zeros(
    	shape=(
    		n_batches,
    		batch_size,
    		max_timesteps,
    		max_emotions),
    	dtype=np.float32)

    for batch in range(n_batches):
        for i in range(batch_size):
            (tweet, emotion) = data[(batch * batch_size) + i]

            # Add emotion distribution
            emotions[batch, i, (emotion-1)] = 1

            # Sequence
            timesteps = min(max_timesteps, len(tweet))

            # Sequence length (time steps)
            lengths[batch, i] = timesteps

            # Tweet
            tweets[batch, i, :timesteps] = tweet[:timesteps]

            # NRC lexicon emotion distribution
            for t in range(timesteps):
            	word = tweet[t]
            	
            	if word in lexicon:
            		ems = lexicon.get(word)
            		
            		for e in range(0, len(ems)):
            			x = ems[e]
            			nrc[batch, i, t, (x[0]-1)] = x[1]

    return (tweets, lengths, emotions, nrc)

def train_model(config, train_batches, validation_batches, embed_matrix):
    train_batches, train_lens, train_emotions, train_nrc_emotions = train_batches
    validation_batches, validation_lens, validation_emotions, validation_nrc_emotions = validation_batches

    with tf.Session() as sess:
        with tf.variable_scope("model", reuse=False):
            train_model = Model(
                config,
                train_batches,
                train_lens,
                train_emotions,
                train_nrc_emotions,
                embed_matrix,
                phase=Phase.Train)

        with tf.variable_scope("model", reuse=True):
            validation_model = Model(
                config,
                validation_batches,
                validation_lens,
                validation_emotions,
                validation_nrc_emotions,
                embed_matrix,
                phase=Phase.Validation)

        sess.run(tf.global_variables_initializer())

        for epoch in range(config.n_epochs):
            validation_gold = []
            validation_pred = []

            # Train on all batches.
            for batch in range(train_batches.shape[0]):
                loss, _ = sess.run([train_model.loss, train_model.train_op], {
                    train_model.x: train_batches[batch], train_model.lens: train_lens[batch], 
                    train_model.y: train_emotions[batch], train_model.embed: embed_matrix,
                    train_model.lexicon: train_nrc_emotions[batch]})

            # Validation on all batches.
            for batch in range(validation_batches.shape[0]):
                gold, pred = sess.run([validation_model.gold, validation_model.pred], {
                    validation_model.x: validation_batches[batch], validation_model.lens: validation_lens[batch], 
                    validation_model.y: validation_emotions[batch], validation_model.embed: embed_matrix,
                    validation_model.lexicon: validation_nrc_emotions[batch]})
                gold = gold.tolist()
                pred = pred.tolist()
                validation_gold += gold
                validation_pred += pred

            precision = precision_score(validation_gold, validation_pred, average='micro')
            recall = recall_score(validation_gold, validation_pred, average='micro')
            f1 = f1_score(validation_gold, validation_pred, average='micro')

            print(
            	"epoch %d - precision: %.2f, recall: %.2f, f1: %.2f" %
            	(epoch, precision * 100, recall * 100, f1 * 100))

# TRAIN_DATA = part0-part7
# TEST_DATA = part8

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: %s WORD_EMBEDDINGS TRAIN_SET DEV_SET\n" % sys.argv[0])
        sys.exit(1)

    config = DefaultConfig()

    # Load the word embeddings and get the embedding matrix
    embeds = gensim.models.Word2Vec.load(sys.argv[1])
    embedding_matrix = embeds.wv.syn0

    # Read training and validation data.
    train_lexicon = read_lexicon(sys.argv[2])
    validation_lexicon = read_lexicon(sys.argv[3])
    nrc_lexicon = read_nrc_lexicon("NRC-Hashtag-Emotion-Lexicon.txt")

    # Convert tweets and emotion labels to numeral representations
    words = Numberer()
    emotions = Numberer()
    train_lexicon = recode_lexicon(train_lexicon, words, emotions)
    validation_lexicon = recode_lexicon(validation_lexicon, words, emotions)
    nrc_lexicon = recode_nrc_lexicon(nrc_lexicon, words, emotions)

    # Generate batches
    train_batches = generate_instances(
        train_lexicon,
        nrc_lexicon,
        config.max_emotions,
        config.max_timesteps,
        batch_size=config.batch_size)
    validation_batches = generate_instances(
        validation_lexicon,
        nrc_lexicon,
        config.max_emotions,
        config.max_timesteps,
        batch_size=config.batch_size)

    # Train the model
    train_model(config, train_batches, validation_batches, embedding_matrix)
