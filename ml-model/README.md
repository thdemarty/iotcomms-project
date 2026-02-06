# ML Model

## Scenario I

In scenario I, the goal is to uniquely identify the deployment environments. For this scenario, mix all the RSSI time series belonging to the same environments.

## Scenario II

In scenario II, the goal is to uniquely identify the sensor nodes. For this scenario, mix all the RSSI time series belonging to the same sensor node.

## Machine Learning

You train a CNN model and a ResNet model to classify the deployment environments and the sensor nodes by providing these models link quality parameters (RSSI) as inputs.
You need to consider two train/test strategies: with and without previously seen data.
In the first, the test and train data originate from the same dataset.
In the second, the test and train data originate from different datasets.
Here are different possibilities of organising the datasets (scenario II is used as an example):

- (1) You mix the data from all the deployment environments belonging to Node X in a single dataset. Use some portion of the dataset (say 75%) to train the models and the rest (25%) to test.
- (2) Mix the data from 4 of the deployment environments belonging to Node X in a single dataset but the data of the fifth environment you keep in a different dataset. Use the first to train the model and the second to test the model. Use the same strategy for classifying the environments.

## Run the modes

Models:

- CNN_strategie1
- CNN_strategie2
- ResNet_strategie1
- ResNet_strategie2

### Prerequisites

`!pip3 install torch torchvision gensim`

### Train and Test

`python <model>.py <options>`

DESCRIPTION

`-n [NODE]`		The data of which node should be used for training. \
`-e [EXCLUDED_ENV]`	ONLY FOR STRATEGIE 2!!!! Which of the environments should be used for testing and not training. \
`-f [FRAMESIZE]`	Framesize for model inputs in measurements (10 per sec). \
`-o [OVERLAP]`		Overlap of the frames in percent. E.g. 50
