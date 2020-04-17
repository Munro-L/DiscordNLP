#!/usr/bin/python3
import numpy as np
import pandas
import time
import re
import math
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds


cols = ["sentiment", "text"]
train_data = pandas.read_csv(
    "data/training_data_short.csv",
    header=None,
    names=cols,
    engine="python",
    encoding="latin1"
)

data_clean = train_data["text"].tolist()

# create tokenizer, splits sentences into sub-sequences that can be represented as a vector
# the tokenizer takes forever to generate, save it as a pickle once we're done
try:
    with open("models/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
except:
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        data_clean, target_vocab_size=2**16
    )
    with open("models/tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

data_input = [tokenizer.encode(sentence) for sentence in data_clean]

# pad sentences with 0's to match the longest sentence in the data set 
max_sentence_length = max([len(sentence) for sentence in data_input])
data_input = tf.keras.preprocessing.sequence.pad_sequences(
    data_input, value=0, padding="post", maxlen=max_sentence_length
)
data_labels = train_data["sentiment"].to_numpy()
test_idx = np.random.randint(0, math.floor(len(data_clean)/2), max(math.floor(len(data_clean)/200), 100))
test_idx = np.concatenate((test_idx, test_idx+math.floor(len(data_clean)/2)))
test_inputs = data_input[test_idx]
test_labels = data_labels[test_idx]
train_inputs = np.delete(data_input, test_idx, axis=0)
train_labels = np.delete(data_labels, test_idx)


# class DCNN(tf.keras.Model):  
#     def __init__(self,
#                  vocab_size,
#                  emb_dim=128,
#                  nb_filters=50,
#                  FFN_units=512,
#                  nb_classes=2,
#                  dropout_rate=0.1,
#                  training=False,
#                  name="dcnn"):
#         super(DCNN, self).__init__(name=name)
        
#         self.embedding = layers.Embedding(vocab_size, emb_dim)
#         self.bigram = layers.Conv1D(filters=nb_filters,
#                                     kernel_size=2,
#                                     padding="valid",
#                                     activation="relu")
#         self.trigram = layers.Conv1D(filters=nb_filters,
#                                      kernel_size=3,
#                                      padding="valid",
#                                      activation="relu")
#         self.fourgram = layers.Conv1D(filters=nb_filters,
#                                       kernel_size=4,
#                                       padding="valid",
#                                       activation="relu")
#         self.pool = layers.GlobalMaxPool1D() # no training variable so we can
#                                              # use the same layer for each
#                                              # pooling step
#         self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
#         self.dropout = layers.Dropout(rate=dropout_rate)
#         if nb_classes == 2:
#             self.last_dense = layers.Dense(units=1, activation="sigmoid")
#         else:
#             self.last_dense = layers.Dense(units=nb_classes, activation="softmax")


#     def call(self, inputs, training):
#         x = self.embedding(inputs)
#         x_1 = self.bigram(x)
#         x_1 = self.pool(x_1)
#         x_2 = self.trigram(x)
#         x_2 = self.pool(x_2)
#         x_3 = self.fourgram(x)
#         x_3 = self.pool(x_3)
#         merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3 * nb_filters)
#         merged = self.dense_1(merged)
#         merged = self.dropout(merged, training)
#         output = self.last_dense(merged)
#         return output

VOCAB_SIZE = tokenizer.vocab_size
EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = len(set(train_data["sentiment"]))
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
NB_EPOCHS = 5

# Dcnn = tf.keras.models.Sequential()
# Dcnn.add(layers.Embedding(VOCAB_SIZE, 128))
# Dcnn.add(layers.Conv1D(filters=NB_FILTERS, kernel_size=2, padding="valid", activation="relu"))
# Dcnn.add(layers.GlobalMaxPool1D())
# Dcnn.add(layers.Conv1D(filters=NB_FILTERS, kernel_size=3, padding="valid", activation="relu"))
# Dcnn.add(layers.GlobalMaxPool1D())
# Dcnn.add(layers.Conv1D(filters=NB_FILTERS, kernel_size=4, padding="valid", activation="relu"))
# Dcnn.add(layers.GlobalMaxPool1D())
# Dcnn.add(layers.Dense(units=NB_CLASSES, activation="softmax"))


embedding = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=128)
pool = layers.GlobalMaxPool1D()
dense = layers.Dense(units=NB_CLASSES, activation="softmax")
dropout = layers.Dropout(rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    last_dense = layers.Dense(units=1, activation="sigmoid")
else:
    last_dense = layers.Dense(units=NB_CLASSES, activation="softmax")

bigram = layers.Conv1D(filters=NB_FILTERS, kernel_size=2, padding="valid", activation="relu")(embedding)
trigram = layers.Conv1D(filters=NB_FILTERS, kernel_size=3, padding="valid", activation="relu")(embedding)
quadgram = layers.Conv1D(filters=NB_FILTERS, kernel_size=4, padding="valid", activation="relu")(embedding)
biconcat = pool()(bigram)
triconcat = pool()(trigram)
quadconcat = pool()(quadgram)
concat = layers.Concatenate()([biconcat, triconcat, quadconcat])
dense_1 = dense()(concat)
dropout_1 = dropout()(dense_1)
last_dense_1 = last_dense()(dropout_1)
Dcnn = Model(input=embedding, output=last_dense_1)

# Dcnn = tf.keras.models.Sequential()
# Dcnn.add(embedding)
# Dcnn.add(layers.Concatenate([biconcat, triconcat, quadconcat], axis=1))
# Dcnn.add(dense)
# Dcnn.add(dropout)
# Dcnn.add(last_dense)




# Dcnn = DCNN(vocab_size=VOCAB_SIZE,
#             emb_dim=EMB_DIM,
#             nb_filters=NB_FILTERS,
#             FFN_units=FFN_UNITS,
#             nb_classes=NB_CLASSES,
#             dropout_rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])
else:
    Dcnn.compile(loss="sparse_categorical_crossentropy",
                 optimizer="adam",
                 metrics=["sparse_categorical_accuracy"])

checkpoint_path = "chkpts"

# Create a callback that saves the model's weights
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

ckpt = tf.train.Checkpoint(Dcnn=Dcnn)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")


Dcnn.fit(train_inputs,
         train_labels,
         batch_size=BATCH_SIZE,
         epochs=NB_EPOCHS,
         callbacks=[checkpoint_callback])
ckpt_manager.save()

Dcnn.save("smol_completed_model.h5")

results = Dcnn.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE)
print(results)

print(Dcnn(np.array([tokenizer.encode("You are so funny")]), training=False).numpy())
print(Dcnn(np.array([tokenizer.encode("You suck, go die in a hole")]), training=False).numpy())
print(Dcnn(np.array([tokenizer.encode("Teaching an AI to love is pretty heartwarming")]), training=False).numpy())