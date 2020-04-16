import discord
import configparser
import tensorflow as tf

config = configparser.ConfigParser()
try:
    config.read("config.txt")
except:
    print("[!!] Error: config.txt could not be read. Fill out config_template.txt and rename it to config.txt.")

client = discord.Client()

@client.event
async def on_ready():
    print("We have logged in as {0.user}".format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith("$hello"):
        await message.channel.send("Hello!")

client.run(config["DEFAULT"]["client_token"])