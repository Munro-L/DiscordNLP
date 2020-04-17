#!/usr/bin/python3
import discord
import configparser
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds


# Initialize Discord client, config, trained CNN model, and vocabulary tokenizer
client = discord.Client()
config = configparser.ConfigParser()
try:
    config.read("config.txt")
except:
    print("[!!] Error: config.txt could not be read. Fill out config_template.txt and rename it to config.txt.")
print("[*] Loading volcabulary tokenizer from: {0}".format(config["DEFAULT"]["tokenizer_path"]))
with open(config["DEFAULT"]["tokenizer_path"], "rb") as f:
    tokenizer = pickle.load(f)

tokenizer = pickle.load(open("models/tokenizer.pickle", "rb"))
interperter = tf.lite.Interpreter(model_path="models/converted_big_model.tflite")
interperter.allocate_tensors()
input_details = interperter.get_input_details()
output_details = interperter.get_output_details()


# async def cnn_response(text):
#     TODO


async def measure_sentiment(text):
    tokenized_input = np.array([tokenizer.encode(text)], dtype=np.int32)
    tokenized_input = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_input, value=0, padding="post", maxlen=73
    )
    interperter.set_tensor(input_details[0]["index"], tokenized_input)
    interperter.invoke()
    sentiment = interperter.get_tensor(output_details[0]["index"])
    return sentiment


@client.event
async def on_ready():
    print("[*] We have logged in as {0.user}".format(client))


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
