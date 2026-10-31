# Logistic Regression From Scratch: A First-Principles Classification Engine

This repository contains a Jupyter Notebook that builds a fully functional Logistic Regression classifier entirely from scratch using Python and NumPy. It demonstrates the transition from continuous numerical prediction (Linear Regression) to binary classification by implementing the Sigmoid activation function and Binary Cross-Entropy loss.

## Overview

The core objective of this project is to demystify the internal mechanics of classification algorithms. By writing the gradient descent optimization line-by-line, we expose the underlying calculus and linear algebra. The custom-built engine is then rigorously benchmarked against the industry-standard `scikit-learn` library using real-world medical data (the Breast Cancer Wisconsin dataset).

## Key Mathematical Foundations

This notebook explicitly codes the mathematical formulas required to "bend" a linear equation into a strict probability distribution:

1. **The Sigmoid Activation Function ($\sigma$):**
   Maps the raw linear output ($z = Xw + b$) to a strict probability range between 0.0 and 1.0.
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

2. **Binary Cross-Entropy Loss (Log Loss):**
   Replaces Mean Squared Error (MSE) to maintain a convex loss landscape when paired with the Sigmoid function. It heavily penalizes the model for being highly confident but incorrect.
   $$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

3. **Gradient Descent (Calculus & Chain Rule):**
   Calculates the partial derivatives of the log loss function with respect to the weights and bias. Because the derivatives of the Log Loss and Sigmoid functions elegantly cancel each other out, the resulting gradient formula matches Linear Regression:
   $$dw = \frac{1}{N} X^T (\hat{y} - y)$$
   $$db = \frac{1}{N} \sum (\hat{y} - y)$$

## Implementation Details

* **Object-Oriented API:** The model is encapsulated in a `ScratchLogisticRegression` class, mimicking the standard `fit()`, `predict_proba()`, and `predict_classes()` interface used in production environments.
* **Data Preprocessing:** Implements manual **Z-Score Standardization** (Zero-Mean, Unit-Variance) to ensure a spherical loss landscape and stable, rapid convergence.
* **Algorithmic Fairness:** To perform a mathematically pure comparison, Scikit-Learn's default L2 regularization is explicitly disabled (`penalty=None`), forcing both models to minimize the pure, unconstrained log loss function.

## Key Insights: Multicollinearity and Interpretability

A significant finding in this notebook is the behavior of the models when exposed to **Multicollinearity** (highly correlated features such as tumor radius, perimeter, and area). 

* **The Scratch Model:** Handled the correlation intuitively by assigning logical, uniform negative weights to all size-based features, correctly indicating that an increase in size decreases the probability of the tumor being benign (Class 1).
* **Scikit-Learn (Unregularized):** Destabilized under the multicollinearity, assigning a massive positive weight to radius (+33.17) and massive negative weights to perimeter and area to mathematically balance the equation. 

This highlights the critical importance of understanding algorithms from first principles, particularly in clinical and healthcare AI, where model explainability and biological logic are just as important as raw accuracy.

## Dependencies

* `numpy`: Core mathematical engine for matrix algebra and vectorized calculus.
* `pandas`: Data manipulation.
* `scikit-learn`: Used exclusively for loading the dataset (`load_breast_cancer`) and benchmarking metrics (`LogisticRegression`, `log_loss`, `accuracy_score`).

## How to Run

1. Clone the repository and open the notebook in your preferred environment (Jupyter, Colab, VS Code).
2. Execute the cells sequentially.
3. Observe the final printout comparing the Custom Model vs. Scikit-Learn on Accuracy, Log Loss, and Feature Weights.