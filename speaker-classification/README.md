# Speaker Identification using Deep Learning

This repository contains a deep learning project focused on speaker identification using various neural network architectures. The project experiments with different models such as Shallow Neural Networks, AlexNet, VGGNet, and a combination of CNNs and LSTMs to analyze and classify speakers based on their voice characteristics. It utilizes the Accents of the British Isles (ABI-1) Corpus for training and testing purposes.

## Project Overview
Speaker identification is a task of recognizing who is speaking by analyzing audio characteristics. This project applies deep learning techniques to learn and identify different speakers from the ABI-1 dataset. It explores various architectures and evaluates their performance in identifying speakers correctly.

## Features
Multiple architecture support: Shallow Neural Networks, AlexNet, VGGNet, CNN + LSTM.
Customizable training parameters.
Use of the ABI-1 Corpus for a diverse accent dataset.
Evaluation metrics to assess model performance.

## Requirements
Before you begin, ensure you have met the following requirements:

- Python 3.6+
- TensorFlow 2.x
- Librosa (for audio processing)
- NumPy
- argparse (for command-line options)

## Installation
Clone the repository to your local machine:

`git clone git@github.com:saranabhani/speaker-classification.git`

`cd speaker-classification`

Install the required packages:

`pip install -r requirements.txt`

## Usage
The training script can be run from the command line, providing flexibility in terms of the model architecture and hyperparameters used. Below is the syntax for the script with available options:

`python main.py [-h] -d DATASET [-m MODEL] [-l LOSS] [-o OPTIMIZER] [-e EPOCHS]
               [-b BATCH_SIZE] [-eb E_BATCH_SIZE] [-met METRIC] [-lr LEARNING_RATE]
               [-do DROPOUT] [-fs FRAME_SIZE] [-nm N_MELS] [-cs CHUNK_SECONDS]
               [-mn MODEL_NAME]
`

### Command-Line Arguments:
* -d, --dataset: Path to the input dataset (required).
* -m, --model: Model architecture to use (default: "vgg"). Options: "vgg", "alexnet", "shallow", "mixed" (CNN + LSTM).
* -l, --loss: Loss function (default: "categorical_crossentropy").
* -o, --optimizer: Optimizer to use. Options: "adam", "sgd" (default: "sgd").
* -e, --epochs: Number of epochs (default: 50).
* -b, --batch_size: Batch size (default: 32).
* -eb, --e_batch_size: Evaluation batch size (default: 10).
* -met, --metric: Metric to use (default: "accuracy").
* -lr, --learning_rate: Learning rate (default: 0.005).
* -do, --dropout: Dropout rate (default: 0.5).
* -fs, --frame_size: Frame size (default: 1024).
* -nm, --n_mels: Number of Mel bands to generate (default: 128).
* -cs, --chunk_seconds: Length of audio chunks (in seconds) (default: 3).
* -mn, --model_name: Model name (required).

### Example:
`python main.py -d "data/ABI-1 Corpus" -m "vgg" -o "adam" -e 50 -b 32 -eb 10 -lr 0.005 -do 0.5 -fs 1024 -nm 128 -cs 3 -mn "vgg_model"`

This will train a VGGNet model using the ABI-1 Corpus, with the Adam optimizer, 50 epochs, a batch size of 32, an evaluation batch size of 10, a learning rate of 0.005, a dropout rate of 0.5, a frame size of 1024, 128 Mel bands, and audio chunks of 3 seconds. The model information will be saved as "vgg_model" in the "models" directory.