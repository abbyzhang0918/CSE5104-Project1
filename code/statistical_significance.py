import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from gradient_descent_optimization import load_data
from scaling_optimal_solution import feature_names


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    X_train, y_train, X_test, y_test = load_data()

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean) / std
    X_train_log = np.log1p(X_train)

    names = ['untransformed_p_value', 'standardized_p_value', 'log_transformed_p_value']
    training_sets = [X_train, X_train_scaled, X_train_log]
    table = pd.DataFrame({'feature': ['Intercept'] + feature_names})

    for i in range(3):
        model = sm.OLS(y_train, sm.add_constant(training_sets[i])).fit()
        table[names[i]] = model.pvalues

    table.to_csv('results/p_values.csv', index=False)
    print('Feature p-values (statsmodels OLS):')
    print(table.to_string(index=False))
