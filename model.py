from enum import Enum

import tensorflow as tf
from tensorflow.contrib import rnn

class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2

class Model:
    def __init__(
            self,
            config,
            batch,
            lens_batch,
            emotion_batch,
            nrc_batch,
            embed_matrix,
            phase=Phase.Predict):
        batch_size = batch.shape[1]
        input_size = batch.shape[2] # 31
        emotion_size = emotion_batch.shape[2] # 6

        # The tweets. input_size is the (maximum) number of timesteps, i.e. maximum tweet length
        self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

        # This tensor provides the actual number of timesteps for each
        # instance (words in a tweet).
        self._lens = tf.placeholder(tf.int32, shape=[batch_size])

        # The emotion distribution
        if phase != Phase.Predict:
            self._y = tf.placeholder(tf.int32, shape=[batch_size, emotion_size])

        # Embedding matrix
        self._embed = tf.placeholder(tf.float32, shape=[embed_matrix.shape[0], embed_matrix.shape[1]])
        word_embeddings = tf.nn.embedding_lookup(self._embed, self._x)

        # Lexicon
        self._lexicon = lexicon = tf.placeholder(tf.float32, shape=[batch_size, input_size, emotion_size])

        features = tf.concat([word_embeddings, lexicon], axis=2)

        cell = rnn.GRUCell(100)
        
        if phase == Phase.Train:
            regularized_cell = rnn.DropoutWrapper(cell, input_keep_prob=config.input_dropout, 
                state_keep_prob=config.hidden_dropout)
            _, hidden = tf.nn.dynamic_rnn(regularized_cell, features, sequence_length=self._lens, dtype=tf.float32)
        else:
            _, hidden = tf.nn.dynamic_rnn(cell, features, sequence_length=self._lens, dtype=tf.float32)

        w = tf.get_variable("w", shape=[hidden.shape[1], emotion_size])
        b = tf.get_variable("b", shape=[emotion_size])
        logits = tf.matmul(hidden, w) + b

        if phase == Phase.Train or Phase.Validation:
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=self._y, logits=logits)
            self._loss = loss = tf.reduce_sum(losses)

        if phase == Phase.Train:
            start_lr = 0.01
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(start_lr, global_step, batch.shape[0], 0.90)
            self._train_op = tf.train.AdamOptimizer(learning_rate) \
                .minimize(losses, global_step=global_step)
            self._probs = probs = tf.nn.softmax(logits)

        if phase == Phase.Validation:
            # Emotions of the gold data
            self._gold = gold_emotions = tf.argmax(self.y, axis=1)

            # Predicted emotions
            self._pred = pred_emotions = tf.argmax(logits, axis=1)

            correct = tf.equal(gold_emotions, pred_emotions)
            correct = tf.cast(correct, tf.float32)
            self._accuracy = tf.reduce_mean(correct)

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lens(self):
        return self._lens

    @property
    def loss(self):
        return self._loss

    @property
    def probs(self):
        return self._probs

    @property
    def train_op(self):
        return self._train_op

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def embed(self):
        return self._embed

    @property
    def lexicon(self):
        return self._lexicon

    @property
    def gold(self):
        return self._gold

    @property
    def pred(self):
        return self._pred
