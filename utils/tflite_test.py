import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# test importing of tflite models
text = "you can say bad bot as long as it's not the only thing you say"
tokenizer = pickle.load(open("../models/tokenizer.pickle", "rb"))
interperter = tf.lite.Interpreter(model_path="../models/converted_big_model.tflite")

interperter.allocate_tensors()
input_details = interperter.get_input_details()
output_details = interperter.get_output_details()
input_shape = input_details[0]["shape"]

tokenized_input = np.array([tokenizer.encode(text)], dtype=np.int32)
tokenized_input = tf.keras.preprocessing.sequence.pad_sequences(
    tokenized_input, value=0, padding="post", maxlen=73
)
interperter.set_tensor(input_details[0]["index"], tokenized_input)
interperter.invoke()
sentiment = interperter.get_tensor(output_details[0]["index"])
print("Input Text: {0}\nScore: {1}".format(text, sentiment[0][0]))


# set importing of h5 models (reusing tokenizer and tokenized_input)
new_model = tf.keras.models.load_model("../models/sentiment_model.h5")
new_model.summary()
print(new_model(tokenized_input, training=False).numpy())