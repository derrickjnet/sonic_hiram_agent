#!/usr/bin/env bash

docker build -f hiram-agent.docker -t $DOCKER_REGISTRY/simple-agent:v$1 .docker build -f hiram-agent.docker -t $DOCKER_REGISTRY/simple-agent:v$1 .

docker push $DOCKER_REGISTRY/simple-agent:v$1