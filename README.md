# CSE5104 Project 1

- **code/**: `gradient_descent_optimization.py` implements gradient descent and evaluation, `scaling_convergence_speed.py` compares raw and standardized features, and `non_linear_transformations.py` analyzes log-transformed features and compares the three optimal models. `scaling_optimal_solution.py` compares the optimal raw and standardized models, `univariate_vs_multivariate.py` trains the eight univariate models and compares their coefficients with the multivariate ones, and `statistical_significance.py` extracts feature p-values with statsmodels.
- **results/**: `scaling_convergence.png` compares raw and standardized training losses, `nonlinear_convergence.png` adds the log-transformed model, and `optimal_models.csv` lists training and testing MSE and R² for the three optimal models. `univariate_gradient_descent.csv` and `univariate_vs_multivariate_coefficients.csv` cover the univariate analysis, and `p_values.csv` lists feature p-values for the three configurations.

## How to reproduce

Data split: rows 501-630 of `data/Concrete_Data.xls` are the testing set (130 samples) and the other 900 rows are the training set.

Install the dependencies (Python 3):

```
python3 -m pip install -r requirements.txt
```

Run every script from the repository root. The scripts are independent, so the order does not matter:

```
python3 code/gradient_descent_optimization.py
python3 code/scaling_convergence_speed.py
python3 code/scaling_optimal_solution.py
python3 code/non_linear_transformations.py
python3 code/univariate_vs_multivariate.py
python3 code/statistical_significance.py
```

Figures and tables are saved in `results/`. `gradient_descent_optimization.py` and `scaling_optimal_solution.py` only print their results.
