#!/usr/bin/python3
import pandas
from bs4 import BeautifulSoup
import re

'''
This script is mostly hardcoded specifically for the Sentiment 140 data set, located in a folder named "data".
Cleaned data will also be written back to "data".
Grab your own copy of the data set from: https://www.kaggle.com/kazanova/sentiment140
'''

def clean_message(msg):
    msg = BeautifulSoup(msg, "lxml").get_text()
    msg = re.sub(r"@[A-Za-z0-9]+", ' ', msg)
    msg = re.sub(r"https?://[A-Za-z0-9./]+", " ", msg)
    msg = re.sub(r"[^a-zA-Z.!?']", " ", msg)
    msg = re.sub(r" +", " ", msg)
    return msg

def normalize_sentiment(score):
    if score == 4 or score == 2:
        score = 1
    return score


cols = ["sentiment", "id", "date", "query", "user", "text"]
data = pandas.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    header=None,
    names=cols,
    engine="python",
    encoding="latin1"
)
data.drop(["id", "date", "query", "user"], axis=1, inplace=True)
data["text"] = data["text"].apply(clean_message)
data["sentiment"] = data["sentiment"].apply(normalize_sentiment)
data.to_csv("data/training_data_long.csv", index=False, header=False)


data = pandas.read_csv(
    "data/testdata.manual.2009.06.14.csv",
    header=None,
    names=cols,
    engine="python",
    encoding="latin1"
)
data.drop(["id", "date", "query", "user"], axis=1, inplace=True)
data["text"] = data["text"].apply(clean_message)
data["sentiment"] = data["sentiment"].apply(normalize_sentiment)
data.to_csv("data/training_data_short.csv", index=False, header=False)
