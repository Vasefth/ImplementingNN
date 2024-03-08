# ImplementingNN

# Life Expectancy Prediction using Neural Networks

## Overview

This project leverages a neural network model to predict the life expectancy of countries around the globe. It uses a dataset that encompasses a range of indicators from the World Health Organization’s Global Health Observatory (GHO) data repository, covering the years 2000 to 2015. These indicators include immunization factors, mortality factors, economic factors, social factors, and other health-related factors. The goal is to use these indicators to accurately predict life expectancy, helping inform countries about potential areas of improvement for increasing their populations' life expectancy.

## Dataset

The dataset life_expectancy.csv incorporates several indicators for all countries from 2000 to 2015, such as:

- Immunization factors
- Mortality factors
- Economic factors
- Social factors
- Other health-related factors

*Note:* The Country column is dropped from the dataset as it is not used in the prediction model.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- TensorFlow

## Installation

First, ensure that Python and pip are installed on your machine. Then, install the required packages using pip:

pip install pandas scikit-learn tensorflow

## Usage

Ensure the life_expectancy.csv file is in the same directory as the script. Run the script with Python to train the model and evaluate its performance:

python life_expectancy_prediction.py

The script performs the following operations:

1. Loads and preprocesses the dataset.
2. Splits the data into training and testing sets.
3. Applies standard scaling to the numerical features.
4. Constructs a neural network model for regression.
5. Trains the model on the training data.
6. Evaluates the model's performance on the testing data and prints the mean squared error (MSE) and mean absolute error (MAE).

## Model Architecture

The model consists of:

- An input layer tailored to the feature set's dimensions.
- A hidden layer with 64 neurons and ReLU activation.
- An output layer designed to predict life expectancy in years.

## Optimizer and Loss Function

- *Optimizer:* Adam with a learning rate of 0.01.
- *Loss Function:* Mean Squared Error (MSE) for regression tasks.

## Evaluation

After training, the model's performance is evaluated using the test set, with results presented in terms of MSE and MAE.

## Contributing

Your contributions are welcome! Please fork the repository, make your improvements, and submit a pull request.
