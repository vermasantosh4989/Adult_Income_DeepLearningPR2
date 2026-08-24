# Adult Income ANN Project - Summary Report

## 1. Project Overview

The objective of this project is to develop a Deep Learning based binary classification model using the Adult Income Dataset.

The model predicts whether an individual's annual income is:

- `<=50K` → Class 0
- `>50K` → Class 1

The project focuses on understanding how preprocessing, ANN architecture, activation functions, weight initialization, loss functions, Batch Normalization, and optimizers influence model performance.

## 2. Dataset

The project uses the **Adult Income Dataset** available on Kaggle.

**Kaggle Dataset:** `wenruliu/adult-income-dataset`

The dataset is loaded directly using KaggleHub:

```python
import kagglehub

dataset_path = kagglehub.dataset_download(
    "wenruliu/adult-income-dataset"
)
```

The dataset contains demographic, employment, education, and financial information.

Important features include:

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

The target variable is `income`.

## 3. Data Cleaning and Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using KaggleHub.
2. Removed unnecessary spaces from column names and categorical values.
3. Replaced `?` values with missing values.
4. Removed records containing missing values.
5. Dropped `fnlwgt` because it is a census sampling weight rather than a useful predictive demographic feature.
6. Dropped `education` because the same information is represented numerically by `education.num`.
7. Converted the target into binary values:
   - `<=50K` = 0
   - `>50K` = 1
8. Applied one-hot encoding to categorical features.
9. Applied StandardScaler to numerical features.
10. Used a stratified train-test split to preserve the class distribution.

## 4. Exploratory Data Analysis

EDA was performed to understand the structure and distribution of the dataset.

The analysis included:

- Dataset shape and data types
- Missing-value analysis
- Income class distribution
- Age distribution by income class
- Hours worked per week by income
- Education and income relationship
- Correlation analysis

The target variable is imbalanced, with the `<=50K` class representing the majority and the `>50K` class representing the minority.

Therefore, accuracy alone is not sufficient for judging model quality.

## 5. Baseline ANN

The baseline ANN uses the following architecture:

```text
Input Layer
    ↓
Dense(128, ReLU)
    ↓
Dense(64, ReLU)
    ↓
Dense(1, Sigmoid)
```

Configuration:

- Weight initializer: Glorot Uniform
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Batch size: 64
- Epochs: 50
- Validation split: 10%

The baseline model provides a reference point for all later experiments.

## 6. Activation Function Experiment

Four activation functions were compared:

- ReLU
- Tanh
- Sigmoid
- ELU

The experiment used the same basic network structure so that the effect of activation functions could be compared fairly.

### Main observation

ReLU is generally well suited to hidden layers because it is computationally simple and helps maintain stronger gradients for positive inputs.

Tanh and Sigmoid can suffer more from saturation in deep networks.

ELU can provide smoother negative outputs than ReLU and may reduce some dead-neuron effects.

The exact validation accuracy and F1-score for each activation should be taken from the executed notebook results table.

## 7. ReLU Dead Neuron Analysis

The output of the first hidden ReLU layer was inspected for 500 test samples.

The percentage of zero activations was calculated for each neuron.

A histogram was plotted to visualize the distribution of dead-neuron percentages.

A ReLU neuron can output zero for negative inputs. If a neuron remains inactive for most examples, it may contribute very little to learning.

This experiment helps demonstrate why activation behavior should be inspected rather than considering only final accuracy.

## 8. Weight Initialization Experiment

Five initializers were compared:

- Zeros
- Random Normal
- Glorot Uniform
- He Normal
- He Uniform

### Zero Initialization

The zero-initialized network demonstrates a major problem with setting all hidden-layer weights to zero.

When neurons start with identical weights, they receive identical gradients and continue learning the same representations.

As a result, the neurons cannot effectively break symmetry.

### Glorot Initialization

Glorot initialization is designed to maintain a suitable variance of activations and gradients across layers.

### He Initialization

He initialization is particularly useful with ReLU-family activations because it accounts for the fact that ReLU can set approximately half of its inputs to zero.

The validation curves in the notebook show the differences in convergence between initialization methods.

## 9. Loss Function Experiment

Four loss approaches were evaluated:

### Binary Cross-Entropy

Binary Cross-Entropy is the standard loss for binary classification.

It directly measures the difference between predicted probabilities and the true binary labels.

### Mean Squared Error

MSE was tested as a comparison.

Although it can be used with a sigmoid output, it is generally less appropriate than Binary Cross-Entropy for binary classification.

### Weighted Binary Cross-Entropy

Class weights were introduced to give greater importance to the minority `>50K` class.

This can improve the model's ability to identify minority-class examples.

### Focal Loss

Focal Loss gives greater attention to difficult examples and reduces the relative contribution of easy examples.

This makes it useful for imbalanced classification problems.

## 10. Batch Normalization Experiment

Batch Normalization was compared with the baseline ANN.

The main architecture was:

```text
Dense → BatchNorm → ReLU
```

Batch Normalization normalizes intermediate activations during training and can improve training stability.

A second experiment compared:

```text
Dense → BatchNorm → ReLU
```

with:

```text
Dense → ReLU → BatchNorm
```

The notebook compares their validation accuracy and training behavior.

The learned BatchNorm `gamma` and `beta` parameters were also inspected.

## 11. Optimizer Experiment

The following optimizers were compared:

- SGD
- SGD with Momentum
- RMSProp
- Adam

The models were trained using the same general architecture and training configuration.

### SGD

Vanilla SGD updates parameters using the current gradient and learning rate.

### SGD with Momentum

Momentum uses information from previous updates to make optimization smoother and faster.

### RMSProp

RMSProp adapts the learning rate using a moving average of squared gradients.

### Adam

Adam combines momentum-like first-moment tracking with second-moment tracking.

For this type of tabular ANN problem, Adam is expected to provide fast and stable convergence.

The final optimizer selection should be based on the actual validation performance and convergence curves produced by the notebook.

## 12. Learning Rate Sensitivity

Different learning rates were tested for SGD and Adam.

### SGD

- 0.001
- 0.01
- 0.1

### Adam

- 0.0001
- 0.001
- 0.01

The experiment demonstrates that learning rate strongly affects convergence.

A learning rate that is too small can make training slow, while a learning rate that is too large can cause unstable training.

## 13. Final ANN Configuration

The final ANN combines the techniques explored throughout the project.

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
- Batch Normalization: Enabled
- Batch size: 64
- Epochs: 80
- Class weights: Applied when appropriate

## 14. Evaluation Metrics

The project evaluates the models using:

- Accuracy
- Precision
- Recall
- F1-Score
- PR-AUC
- ROC-AUC
- Confusion Matrix

Special attention is given to the minority `>50K` class.

### Why F1-Score?

F1-score combines Precision and Recall and is useful when the class distribution is imbalanced.

### Why Recall?

Recall measures how many actual `>50K` cases are correctly identified.

### Why PR-AUC?

Precision-Recall AUC is useful for evaluating minority-class performance when the target distribution is imbalanced.

## 15. Results Summary

The notebook generates a final comparison table containing:

| Model / Experiment | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline ANN | ReLU | 0.84 | 0.69 | 0.62 | 0.65 | 0.74 | 0.89 | 
| Best Activation | Sigmoid | 0.85 | 0.72 | 0.65 | 0.68 | 0.78 | -- |
| Best Initializer | glorot_uniform | 0.84 | 0.70 | 0.62 | 0.66 | 0.74 | -- |
| Weighted BCE | BCE | 0.80 | 0.58 | 0.79 | 0.61 | 0.74 | -- |
| Focal Loss | BCE | 0.83 | 0.69 | 0.62 | 0.66 | 0.74 | -- |
| Batch Normalization | BatchNorm | 0.84 | 0.70 | 0.60 | 0.65 | 0.75 | -- |
| Best Optimizer | SGD | 0.85 | 0.72 | 0.64 | 0.68 | 0.77 | 0.90 |
| Final ANN | ReLU | 0.80 | 0.56 | 0.81 | 0.66 | 0.74 | 0.89 |


## 16. Overall Findings

The experiments demonstrate several important principles of deep learning:

1. Proper preprocessing is essential for tabular neural networks.
2. Categorical variables must be converted into numerical representations before ANN training.
3. Numerical features should be scaled.
4. ReLU is generally a strong activation function for hidden layers.
5. Zero initialization is unsuitable for hidden neural-network layers because it prevents effective symmetry breaking.
6. He initialization is a suitable choice for ReLU-based networks.
7. Binary Cross-Entropy is more appropriate than MSE for standard binary classification.
8. Weighted loss functions and Focal Loss can help address class imbalance.
9. Batch Normalization can improve training stability.
10. Adam generally provides fast convergence for this type of ANN.
11. Learning rate has a major effect on optimization.
12. Accuracy should not be used as the only metric for an imbalanced classification problem.

## 17. Conclusion

This project provides a complete practical study of Artificial Neural Networks for tabular binary classification.

Starting from the Adult Income Dataset, the workflow covers data cleaning, exploratory analysis, encoding, scaling, ANN architecture design, activation functions, weight initialization, loss functions, Batch Normalization, optimizer selection, learning-rate experiments, and final model evaluation.

The final model combines the most useful techniques identified during the experiments.

The most important lesson is that neural-network performance depends not only on the architecture but also on preprocessing, initialization, loss function, optimization strategy, learning rate, and the way model performance is evaluated.

The final model and its exact performance metrics should be reported from the executed notebook.

## 18. Deliverables

- `Deep_LearningPR2.ipynb` - Complete project notebook
- `adult_income_ann.keras` - Final trained ANN model
- `scaler.pkl` - Saved feature scaler
- `README.md` - Project documentation
- `summary_report.md` - Project summary report
