# Docker Lab 1 – Changes
## Overview
This lab exlpores the basic setup of docker and running an ML model in containerised environment.
## Dataset
Swapped out the original dataset for the **Wine dataset** (`sklearn.datasets.load_wine`). The model now trains on 13 physicochemical features to classify wines into one of three cultivar categories.
## Steps to run the code 
    docker build -t lab1:v1 .
    docker save lab1:v1 > docker_lab1.tar
    docker run lab1:v1