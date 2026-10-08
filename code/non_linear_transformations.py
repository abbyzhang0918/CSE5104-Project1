import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scaling_convergence_speed import run_scaling
from gradient_descent_optimization import load_data, gradient_descent, evaluate


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()

    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)
    learning_rate = 0.003
    iterations = 20000
    weights, bias, log_losses = gradient_descent(
        X_train_log, y_train, learning_rate, iterations)

    train_mse, train_r2 = evaluate(y_train, X_train_log @ weights + bias)
    test_mse, test_r2 = evaluate(y_test, X_test_log @ weights + bias)
    log_summary = pd.DataFrame([
        ['Log-transformed', learning_rate, train_mse, test_mse, train_r2, test_r2]
    ], columns=['model', 'learning_rate', 'training_mse', 'testing_mse', 'training_r2', 'testing_r2'])
    raw_losses, scaled_losses, scaling_summary = run_scaling()
    gd_summary = pd.concat([scaling_summary, log_summary], ignore_index=True)
    print('Gradient descent results:')
    print(gd_summary.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.plot(raw_losses, label='Untransformed')
    plt.plot(scaled_losses, label='Standardized')
    plt.plot(log_losses, label='Log-transformed')
    plt.xscale('symlog', linthresh=1)
    plt.xlabel('Iterations (symlog scale)')
    plt.ylabel('Training MSE (MPa squared)')
    plt.title('Non-Linear Transformations')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/nonlinear_convergence.png', dpi=150)
    plt.close()

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    names = ['Untransformed', 'Standardized', 'Log-transformed']
    training_sets = [X_train, X_train_scaled, X_train_log]
    testing_sets = [X_test, X_test_scaled, X_test_log]
    rows = []

    for i in range(3):
        model = LinearRegression()
        model.fit(training_sets[i], y_train)
        train_predictions = model.predict(training_sets[i])
        test_predictions = model.predict(testing_sets[i])
        train_mse, train_r2 = evaluate(y_train, train_predictions)
        test_mse, test_r2 = evaluate(y_test, test_predictions)
        rows.append([names[i], train_mse, test_mse, train_r2, test_r2])

    optimal_summary = pd.DataFrame(rows, columns=[
        'model', 'training_mse', 'testing_mse', 'training_r2', 'testing_r2'])
    optimal_summary.to_csv('results/optimal_models.csv', index=False)
    print('\nOptimal models (LinearRegression):')
    print(optimal_summary.to_string(index=False))
