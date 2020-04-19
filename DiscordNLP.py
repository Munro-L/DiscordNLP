#!/usr/bin/python3
import sys
import discord
import configparser
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds


# Initialize Discord client, config, vocabulary tokenizer and trained tflite model
client = discord.Client()
config = configparser.ConfigParser()

try:
    config.read("config.txt")
except:
    print("[!!] Error: config.txt could not be read. Fill out config_template.txt and rename it to config.txt.")
    sys.exit(0)

print("[*] Loading volcabulary tokenizer from: {0}".format(config["DEFAULT"]["tokenizer_path"]))
try:
    with open(config["DEFAULT"]["tokenizer_path"], "rb") as f:
        tokenizer = pickle.load(f)
except:
    print("[!!] Could not find tokenizer")
    sys.exit(0)

print("[*] Loading pre-trained tflite model")
tflite = False
h5 = False
if config["DEFAULT"]["model_path"].split(".")[-1] == "tflite":
    try:
        interperter = tf.lite.Interpreter(model_path=config["DEFAULT"]["model_path"])
        interperter.allocate_tensors()
        input_details = interperter.get_input_details()
        output_details = interperter.get_output_details()
        tflite = True
    except:
        print("[!!] Error loading tflite model")
        sys.exit(0)
elif config["DEFAULT"]["model_path"].split(".")[-1] == "h5":
    try:
        h5_model = tf.keras.models.load_model(config["DEFAULT"]["model_path"])
        h5 = True
    except:
        print("[!!] Error loading h5 model")
        sys.exit(0)
else:
    print("[!!] Saved model is not of supported type. Export as either .h5 or .tflite.")
    sys.exit(0)



async def measure_sentiment(text):
    tokenized_input = np.array([tokenizer.encode(text)], dtype=np.int32)
    tokenized_input = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_input, value=0, padding="post", maxlen=73
    )
    if tflite:
        interperter.set_tensor(input_details[0]["index"], tokenized_input)
        interperter.invoke()
        sentiment = interperter.get_tensor(output_details[0]["index"])
    elif h5:
        sentiment = h5_model(tokenized_input, training=False).numpy()
    return sentiment


@client.event
async def on_ready():
    print("[*] Logged in as {0.user}".format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    elif message.content.startswith("$sentiment"):
        async for x in message.channel.history(limit=2):
            target_message = x
        if target_message != None:
            sentiment = await measure_sentiment(target_message.content)
            await message.channel.send("Score: {0}".format(sentiment[0][0]))   


client.run(config["DEFAULT"]["client_token"])
