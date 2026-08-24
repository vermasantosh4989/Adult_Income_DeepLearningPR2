# Adult Income Classification using Deep Learning

## Project Overview

This project uses the **Adult Income Dataset** to build and evaluate Artificial Neural Network (ANN) models for binary income classification.

The objective is to predict whether a person's annual income is:

- `<=50K` → Class 0
- `>50K` → Class 1

The project compares different activation functions, weight initializers, loss functions, Batch Normalization configurations, and optimizers.

## Dataset

**Dataset:** Adult Income Dataset  
**Source:** Kaggle  
**Kaggle Dataset:** `wenruliu/adult-income-dataset`

The dataset is downloaded automatically inside the Jupyter Notebook using `kagglehub`.

### Dataset Features

- Age
- Workclass
- Education
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

### Target

The `income` column is converted to:

```text
<=50K → 0
>50K  → 1
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras
- KaggleHub
- Jupyter Notebook

## Project Structure

```text
Adult-Income-ANN/
│
├── Adult_Income_ANN_PR2.ipynb
├── adult_income_ann.keras
├── scaler.pkl
└── README.md
```

## Dataset Loading with KaggleHub

The dataset is downloaded automatically:

```python
%pip install -q kagglehub

import kagglehub

dataset_path = kagglehub.dataset_download(
    "wenruliu/adult-income-dataset"
)

print(dataset_path)
```

The downloaded CSV file is then loaded with Pandas.

## Data Preprocessing

The notebook performs:

1. Dataset loading using KaggleHub
2. Column and categorical-value cleaning
3. Replacement of `?` with missing values
4. Removal of rows containing missing values
5. Dropping redundant columns:
   - `fnlwgt`
   - `education`
6. Target encoding
7. One-hot encoding of categorical variables
8. StandardScaler on numerical features
9. Stratified train-test split

## Exploratory Data Analysis

The notebook includes:

- Dataset shape
- Data types
- Missing-value analysis
- Income class distribution
- Age distribution by income
- Hours worked per week by income
- Education level versus income
- Correlation heatmap

## Baseline ANN

Architecture:

```text
Input
  ↓
Dense(128, ReLU)
  ↓
Dense(64, ReLU)
  ↓
Dense(1, Sigmoid)
```

Configuration:

- Activation: ReLU
- Initializer: Glorot Uniform
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Epochs: 50
- Batch Size: 64

## Activation Function Comparison

The following activation functions are compared:

- ReLU
- Tanh
- Sigmoid
- ELU

The notebook also checks the percentage of zero activations in the ReLU hidden layer.

## Weight Initialization Comparison

The following initializers are compared:

- Zeros
- Random Normal
- Glorot Uniform
- He Normal
- He Uniform

Validation-accuracy curves are plotted to compare convergence.

The zero-initialization experiment demonstrates why using identical zero weights is not suitable for hidden neural-network layers.

## Loss Function Comparison

Four approaches are compared:

1. Binary Cross-Entropy
2. Mean Squared Error
3. Weighted Binary Cross-Entropy
4. Focal Loss

Weighted BCE and Focal Loss are included because the target classes are imbalanced.

## Batch Normalization

The project compares ANN training with and without Batch Normalization.

Two BatchNorm positions are also tested:

```text
Dense → BatchNorm → ReLU
```

and

```text
Dense → ReLU → BatchNorm
```

The learned `gamma` and `beta` parameters are also inspected.

## Optimizer Comparison

The following optimizers are compared:

- SGD
- SGD + Momentum
- RMSProp
- Adam
- Adam with explicit beta values

Validation accuracy is plotted over 50 epochs.

## Learning Rate Experiment

### SGD

```text
0.001
0.01
0.1
```

### Adam

```text
0.0001
0.001
0.01
```

This experiment demonstrates the effect of learning rate on convergence.

## Final ANN

The final model combines the techniques explored during the project.

Architecture:

```text
Input
  ↓
Dense(128, He Normal)
  ↓
Batch Normalization
  ↓
ReLU
  ↓
Dense(64, He Normal)
  ↓
Batch Normalization
  ↓
ReLU
  ↓
Dense(1, Sigmoid)
```

Configuration:

- Activation: ReLU
- Initializer: He Normal
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Batch Normalization: Yes
- Batch Size: 64
- Epochs: 80
- Class weights: Used

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- PR-AUC
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve

Because the Adult Income dataset is imbalanced, accuracy is not considered the only important metric. F1, Recall and PR-AUC are also examined for the `>50K` class.

## Results

The notebook automatically creates comparison tables for:

- Activation functions
- Weight initializers
- Loss functions
- Optimizers
- Baseline ANN
- Final ANN

The exact numerical results should be taken from the executed notebook because training results can vary slightly between environments.

## Model Saving

The final trained model is saved as:

```text
adult_income_ann.keras
```

The fitted scaler is saved as:

```text
scaler.pkl
```

## How to Run

### Install packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow kagglehub joblib jupyter
```

### Open the notebook

```text
Adult_Income_ANN_PR2.ipynb
```

Run the notebook from the first cell. The dataset will be downloaded automatically from Kaggle using KaggleHub.

## Conclusion

This project demonstrates a complete deep-learning workflow for tabular binary classification.

It covers data cleaning, exploratory analysis, categorical encoding, feature scaling, ANN architecture design, activation functions, weight initialization, loss functions, Batch Normalization, optimizers, learning-rate sensitivity, class imbalance handling and model evaluation.

The final ANN combines the selected techniques into a complete Adult Income classification model.

## Project Deliverables

- `Deep_LearningPR2.ipynb` → Complete Jupyter Notebook
- `adult_income_ann.keras` → Trained ANN model
- `scaler.pkl` → Feature scaler
- `README.md` → Project documentation
