# DiscordNLP
A Discord bot that uses neural network models to perform live natural language processing.

## Currently Available Commands

 - `$sentiment` -> The bot determine if the previous message had a positive or negative tone
 - More coming soon

## Installation Setup

 1. Use Discord to register an app and create a new bot
 2. Rename `config_template.txt` to `config.txt` and fill in the bot token
 3. Generate and save trained models and tokenizers with Tensorflow/Keras (see Jupyter notebook in `helper_scripts` as an example)
 4. Enter path to trained models/tokenizers in `config.txt`
 5. Either install dependencies with `requirements.txt` and run `DiscordNLP.py`, or launch the bot with Docker

## Installation Through Docker (After Setup is Complete)
Running `docker_up.sh bot` in the project root pulls a Tensorflow Docker container and installs needed dependencies. The project folder is mounted to `/DiscordNLP` inside the container. The bot will launch automatically if trained models are present and the config is populated. 

## Training Your Own Models
Running `docker_up.sh training` in the project root also pulls a Tensorflow Docker container, however it also installs and launches a Jupyter notebook that you can access through your browser. The model created for the `$sentiment` command used the Sentiment 140 data set: https://www.kaggle.com/kazanova/sentiment140. See the example Jupyter notebook in `helper_scripts` for generating a trained model from this data set.

## References
The CNN used in the sentiment analyzer was developed by following along with: https://www.udemy.com/course/modern-nlp/. 
