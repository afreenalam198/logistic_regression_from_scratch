# Logistic Regression From Scratch

This repository contains a Notebook that builds a fully functional Logistic Regression classifier from scratch using Python and NumPy. 

The core objective of this project is twofold:
1. Demystify the internal mechanics of classification algorithms.
2. Demonstrate how unregularized mathematical solvers destabilize when presented with redundant, highly correlated data (multicolinearity).

The from scratch engine is benchmarked against the industry-standard `scikit-learn` library using real-world medical data (the Breast Cancer Wisconsin dataset).

## Key Mathematical Foundations

This notebook explicitly codes the mathematical formulas required to "bend" a linear equation into a strict probability distribution:

1. **The Sigmoid Activation Function ($\sigma$):**
   Maps the raw linear output ($z = Xw + b$) to a strict probability range between 0.0 and 1.0.  
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

2. **Binary Cross-Entropy Loss (Log Loss):**
   We use Binary Cross-Entropy Loss to maintain a convex loss landscape when paired with the Sigmoid function. It heavily penalizes the model for being highly confident but incorrect.  
   $$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

3. **Gradient Descent (Calculus & Chain Rule):**
   Calculates the partial derivatives of the log loss function with respect to the weights and bias. Because the derivatives of the Log Loss and Sigmoid functions elegantly cancel each other out, the resulting gradient formula matches Linear Regression:  
   $$dw = \frac{1}{N} X^T (\hat{y} - y)$$  
   $$db = \frac{1}{N} \sum (\hat{y} - y)$$

## Implementation Details

* **Object-Oriented API:** The model is encapsulated in a `ScratchLogisticRegression` class, mimicking the standard `fit()`, `predict_proba()`, and `predict_classes()` interface used in production environments.
* **Data Preprocessing:** Implements manual **Z-Score Standardization** (Zero-Mean, Unit-Variance) to ensure a spherical loss landscape and stable, rapid convergence.
* **Algorithmic Fairness:** To perform a mathematically pure comparison, Scikit-Learn's default L2 regularization is explicitly disabled (`penalty=None`), forcing both models to minimize the pure, unconstrained log loss function.

## Exploring Multicollinearity: 2 Features vs. 4 Features

To test the robustness of both our scratch model and Scikit-Learn, the notebook trains two separate models:

* **The 2-Feature Model:** Uses mean radius and mean texture (mostly independent features).  
* **The 4-Feature Model:** Adds mean perimeter and mean area. (Note: Radius, perimeter, and area are highly correlated as they are essentially measuring the exact same biological trait).

## Key Insights:

When running the evaluations, a stark contrast emerges between the two models:

* **The 2-Feature Baseline:** Both the Scratch model and Scikit-Learn converge to exact same results (Identical coefficients, accuracy and log loss).

* **The 4-Feature Model:** When correlated features are introduced, the results diverge. Since, mean perimeter and mean area are strongly correlated with mean radius, it introduces multicolinearity. This makes the optimization problem poorly conditioned.
  * **Scikit-learn (L-BFGS):**
    * Achieves a lower log loss (0.185735).
    * Produces substantially different coefficients, including a large positive radius coefficient and large negative perimeter and area coefficients.
    * These coefficients reflect conditional relationships between highly correlated features.

  * **Scratch Model (Gradient Descent):**
    * Achieves a higher log loss (0.233614).
    * Produces substantially different coefficients from Scikit-learn.
    * Its fixed learning rate and basic gradient descent approach may struggle to converge efficiently on this poorly conditioned optimization problem.

**Main Takeaway:**
  * The experiment demonstrates that implementing the correct gradient descent equations is only part of building a successful machine learning model.
  * The choice of optimization algorithm, feature correlation, and convergence criteria can significantly affect training.
  * It also demonstrates why individual coefficients should be interpreted cautiously when a model contains highly correlated features.

## Dependencies

* `numpy`: Core mathematical engine for matrix algebra and vectorized calculus.
* `pandas`: Data manipulation.
* `scikit-learn`: Used exclusively for loading the dataset (`load_breast_cancer`) and benchmarking metrics (`LogisticRegression`, `log_loss`, `accuracy_score`).

## How to Run

1. Clone the repository and open the notebook in your preferred environment (Jupyter, Colab, VS Code).
2. Execute the cells sequentially.
3. Observe the final printout comparing the Custom Model vs. Scikit-Learn on Accuracy, Log Loss, and Feature Weights.
