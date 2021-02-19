#!/bin/bash
eval "$(conda shell.bash hook)"
ENVS=$(conda env list | awk '{print $1}' )

echo $ENVS
if [[ $ENVS = *"mlproject"* ]]; then
    echo "exists and update"
    conda env update -f ./env/env.yml
else 
    echo "doesn't exist create"
    conda env create -f ./env/env.yml
fi;
