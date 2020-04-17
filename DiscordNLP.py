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
print("[*] Loading pre-trained CNN model from: {0}".format(config["DEFAULT"]["model_path"]))
cnn_model = tf.keras.models.load_model(config["DEFAULT"]["model_path"])
print("[*] Loading volcabulary tokenizer from: {0}".format(config["DEFAULT"]["tokenizer_path"]))
with open(config["DEFAULT"]["tokenizer_path"], "rb") as f:
    tokenizer = pickle.load(f)


# async def cnn_response(text):
#     TODO


async def measure_sentiment(text):
    sentiment = cnn_model(np.array([tokenizer.encode(text)]), training=False).numpy()
    return sentiment


@client.event
async def on_ready():
    print("[*] We have logged in as {0.user}".format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith("$reply"):
        async for x in message.channel.history(limit=2):
            target_message = x
        if target_message != None:
            reply = await cnn_response(target_message.content)
            await message.channel.send(target_message.content)
    elif message.content.startswith("$sentiment"):
        async for x in message.channel.history(limit=2):
            target_message = x
        if target_message != None:
            sentiment = await measure_sentiment(target_message.content)
            await message.channel.send(str(sentiment))     


client.run(config["DEFAULT"]["client_token"])
