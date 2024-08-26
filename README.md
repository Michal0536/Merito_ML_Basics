# Merito ML Basics

This repository includes basic machine learning projects created during an introductory machine learning course. The projects show how to use Python and popular libraries to build decision trees, neural networks, and convolutional neural networks (CNNs).

## Projects

### 1. Titanic Decision Tree
A decision tree model was created using the Titanic dataset (`titanic.csv`) to predict whether passengers survived. The tree has a maximum depth of 5 for easier understanding.

- **Script:** `TitanicTree.py`
- **Dataset:** `titanic.csv`
- **Visualization:** The tree structure is saved in `titanicTreeOutput.txt`. You can view it using a tool like [Graphviz](https://www.devtoolsdaily.com/graphviz/).

### 2. Two-Layer Neural Network
A two-layer neural network was built using Keras to predict Titanic passengers' survival. The model includes data preprocessing like scaling and one-hot encoding.

- **Script:** `siec_dwuwarstwowa.py`
- **Dataset:** `titanic.csv`
- **Output:** The script creates graphs showing accuracy and mean squared error (MSE) during training.

### 3. Simple Convolutional Neural Network (CNN) for MNIST
A simple CNN was created using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

- **Script:** `mnist.py`
- **Dataset:** MNIST (downloaded automatically via TensorFlow)
- **Output:** The script shows test accuracy after training.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Michal0536/Merito_ML_Basics.git
    ```

2. Go to the project directory:
    ```bash
    cd Merito_ML_Basics
    ```

3. Run the script you want:
    ```bash
    python script_name.py
    ```