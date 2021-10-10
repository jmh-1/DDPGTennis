# Project 3: Collaboration and Competition 

## Project Details

This project trains a DDPG agent to play a tennis game with itself without allowing the ball to hit the ground or go out of bounds. The state space has 24 continuous values representing the location and velocity of the ball and the agent's racquet for 3 consecutive time steps. And the action space has 2 continuous values representing horizontal movement and jumping. 

The task is episodic and is considered solved when an agent gets an average score of at least .5 over 100 consecutive episodes.

## Getting Started

Running the project requires the following python dependencies which can be installed with pip 

		Pillow>=4.2.1
		matplotlib
		numpy>=1.11.0
		jupyter
		pytest>=3.2.2
		docopt
		pyyaml
		protobuf==3.5.2
		grpcio
		torch
		pandas
		scipy
		ipykernel

It also requires the Tennis environment, which can be dowloaded [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip), unzip this file in the root directory. 


Tennis.ipynb can be used to test the agent created for this project, by running the cells in order. 
