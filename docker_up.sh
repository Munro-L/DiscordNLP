#!/bin/bash
docker build -t discordnlptrainer:latest .
docker run -i -t --network=host --workdir=/DiscordNLP -v $(pwd):/DiscordNLP discordnlptrainer:latest jupyter notebook --allow-root --port=8889
