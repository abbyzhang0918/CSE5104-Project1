import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from gradient_descent_optimization import load_data, gradient_descent, evaluate
from scaling_optimal_solution import feature_names


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    X_train, y_train, X_test, y_test = load_data()

    learning_rates = [0.000005, 0.000035, 0.00006, 0.000015,
                      0.005, 0.0000005, 0.0000008, 0.00008]
    iterations = 20000

    rows = []
    for i in range(8):
        x_train = X_train[:, i]
        x_test = X_test[:, i]

        weights, bias, losses = gradient_descent(
            x_train, y_train, learning_rates[i], iterations)
        train_mse, train_r2 = evaluate(y_train, x_train * weights[0] + bias)
        test_mse, test_r2 = evaluate(y_test, x_test * weights[0] + bias)

        rows.append([feature_names[i], learning_rates[i], weights[0], bias,
                     train_mse, test_mse, train_r2, test_r2])

    gd_summary = pd.DataFrame(rows, columns=[
        'feature', 'learning_rate', 'm', 'b',
        'training_mse', 'testing_mse', 'training_r2', 'testing_r2'])
    gd_summary.to_csv('results/univariate_gradient_descent.csv', index=False)
    print('Univariate gradient descent results:')
    print(gd_summary.to_string(index=False))

    multivariate = LinearRegression()
    multivariate.fit(X_train, y_train)

    rows = []
    for i in range(8):
        x_train = X_train[:, i].reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_train, y_train)
        train_mse, train_r2 = evaluate(y_train, model.predict(x_train))
        rows.append([feature_names[i], model.coef_[0], multivariate.coef_[i],
                     model.coef_[0] - multivariate.coef_[i], train_r2])

    coefficients = pd.DataFrame(rows, columns=[
        'feature', 'univariate_coef', 'multivariate_coef', 'difference',
        'univariate_training_r2'])
    coefficients.to_csv('results/univariate_vs_multivariate_coefficients.csv', index=False)
    print('\nUnivariate vs multivariate coefficients (LinearRegression):')
    print(coefficients.to_string(index=False))

    correlations = pd.DataFrame(X_train, columns=feature_names).corr()
    print('\nFeature correlations (training set):')
    print(correlations.round(2).to_string())
