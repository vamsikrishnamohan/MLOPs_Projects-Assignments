[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/l5PrKmPi)
# MLOps Assignment -  DVC Continuous integration

This project demonstrates an end-to-end MLOps pipeline for training and evaluating a machine learning model. The pipeline is managed using DVC (Data Version Control) to track data, models, and experiments.

## Table of Contents
1. Project Overview

2. Setup Instructions

3. Running Experiments

4. Reproducing Results

5. Project Structure

6. DVC Configuration


## Project Overview
This project trains a CNN model on a dataset split into multiple versions (v1, v2, v3, v1+v2, v1+v2+v3). The pipeline includes the following stages:

1. Data Loading: Load and preprocess the dataset.

2. Data Preparation: Split the data into training, validation, and test sets.

3. Model Training: Train a CNN model on the prepared data.

4. Model Evaluation: Evaluate the model and generate metrics (e.g., accuracy, precision, recall).

5. Experiment Tracking: Use DVC to track experiments, data, and models.

## Setup Instructions
1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```
2. Install Dependencies
Install the required Python packages:

```bash

pip install -r requirements.txt
Install DVC (if not already installed):

pip install dvc
```
3. Configure DVC Remote Storage

I have used a shared local directory, which can configure the DVC remote storage as follows:
```bash

dvc remote add -d myremote /path/to/shared/directory
```

4. Pull Data and Models
Download the data and models tracked by DVC:

```bash

dvc pull
```
## Running Experiments
1. Run a Single Experiment
To run an experiment with specific parameters, use the dvc exp run command:

```bash

dvc exp run -S data.version=v1 -S seed=42 
```
2. Run All Experiments
To run all experiments (combinations of dataset versions and random seeds), use the provided Powershell script:

```bash

.\run_experiments.py
```
#### Running one experiment at a time is recommended as running through the shell script can raise errors is creating and deleting the existing directories because of folder permissions.

## Reproducing Results

1. Reproduce the Pipeline
To reproduce the entire pipeline, use the dvc repro command:

```bash
dvc repro
```
2. Explore Experiment History
To view the history of experiments, use the dvc exp show command:

```bash
dvc exp show
```
This command displays a table of all experiments, including their parameters, metrics, and status.

## Project Structure
```
├── data/
│   ├── raw/                  # Raw dataset
│   ├── processed/            # Processed dataset (output of prepare stage)
│   └── current/              # Current dataset version (used for training)
├── models/                   # Trained models
├── metrics/                  # Evaluation metrics (e.g., accuracy, precision)
├── reports/                  # Reports (e.g., confusion matrices)
├── src/                      # Source code
│   ├── data_loader.py        # Script to load and preprocess data
│   ├── prepare.py            # Script to prepare data
│   ├── train.py              # Script to train the model
│   └── evaluate.py           # Script to evaluate the model
├── params.yaml               # Configuration file for parameters
├── dvc.yaml                  # DVC pipeline definition
├── requirements.txt          # Python dependencies
├── run_experiments.ps1        # Script to run all experiments
└── README.md                 # Project documentation
```
## DVC Configuration
1. DVC Pipeline
The pipeline is defined in dvc.yaml. It includes the following stages:

prepare: Preprocess the data.

train: Train the model.

evaluate: Evaluate the model and generate metrics.

2. DVC Remote Storage
Data and models are stored in a remote storage location (shared local directory). The remote storage is configured using:

```bash
dvc remote add -d myremote <remote-storage-url>
```
3. DVC Experiments
Experiments are tracked using DVC. Each experiment is associated with a unique set of parameters, metrics, and data versions. To view the experiment history, use:

```bash
dvc exp show
```
