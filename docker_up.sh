#!/bin/bash
if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    echo "Execute 'docker_up.sh bot' to run the bot, or 'docker_up.sh trainer' to run the trainer"
    exit
fi

if ! [[ $(which docker) && $(docker --version) ]]; then
    echo "Docker is not installed. Make sure it is installed, and your user is in the Docker group."
    exit
fi

if [ "$1" == "bot" ]; then
    docker build -t discordnlp:bot - < Dockerfile_bot
    docker run -i -t --network=host --workdir=/DiscordNLP -v $(pwd):/DiscordNLP discordnlp:bot python3 DiscordNLP.py

elif [ "$1" == "trainer" ]; then
    docker build -t discordnlp:trainer - < Dockerfile_trainer
    docker run -i -t --network=host --workdir=/DiscordNLP -v $(pwd):/DiscordNLP discordnlp:trainer jupyter notebook --allow-root --port=8889
fi
