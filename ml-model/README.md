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

## Run the models

### Prerequisites

`!pip3 install torch torchvision gensim`

### Usage

1. Dump your csv with the correct format into ../data/
2. Train the model for your data (see [Train and Test](#train-and-test))
3. Work in progress

### Train and Test

> [!IMPORTANT]
> You have to train all of CNN so the combined CNN can load in all the trained weights.

Models:

- CNN
- ResNet

Training:

- train_CNN_scenario1_method1
- train_CNN_scenario1_method2
- train_CNN_scenario2_method1
- train_CNN_scenario2_method2

`python <model>.py`

#### .env file for settings

`FILTER_ID`     : Id of the env or node left out for training with method 2. \
`TEST_SIZE`     : Percentage of test data for method 1 training. \
`RANDOM_STATE`  : Randomness for the test data extraction. \
`BATCH_SIZE`    : Batch size for the training steps. \
`DEVICE`        : Run the model on "cpu" or "gpu". \
`NUM_DEVICES`   : How many sending nodes are in the data. \
`NUM_ENVS`      : How many environments are in the data. \
`EPOCHS`        : How long the Model should train. \
`FRAMESIZE`	    : Framesize for model inputs in measurements (10 per sec -> 100 frames at 10 Hz). \
`OVERLAP`		: Overlap of the frames in percent. E.g. 0.5
