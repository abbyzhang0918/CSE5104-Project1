import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gradient_descent_optimization import load_data, gradient_descent, evaluate


def run_scaling():
    X_train, y_train, X_test, y_test = load_data()

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    raw_rate = 0.0000003
    scaled_rate = 0.2
    iterations = 20000

    raw_weights, raw_bias, raw_losses = gradient_descent(
        X_train, y_train, raw_rate, iterations)
    scaled_weights, scaled_bias, scaled_losses = gradient_descent(
        X_train_scaled, y_train, scaled_rate, iterations)

    raw_train_mse, raw_train_r2 = evaluate(y_train, X_train @ raw_weights + raw_bias)
    raw_test_mse, raw_test_r2 = evaluate(y_test, X_test @ raw_weights + raw_bias)
    scaled_train_mse, scaled_train_r2 = evaluate(
        y_train, X_train_scaled @ scaled_weights + scaled_bias)
    scaled_test_mse, scaled_test_r2 = evaluate(
        y_test, X_test_scaled @ scaled_weights + scaled_bias)

    summary = pd.DataFrame([
        ['Untransformed', raw_rate, raw_train_mse, raw_test_mse, raw_train_r2, raw_test_r2],
        ['Standardized', scaled_rate, scaled_train_mse, scaled_test_mse, scaled_train_r2, scaled_test_r2]
    ], columns=['model', 'learning_rate', 'training_mse', 'testing_mse', 'training_r2', 'testing_r2'])
    return raw_losses, scaled_losses, summary


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    raw_losses, scaled_losses, summary = run_scaling()
    print(summary.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.plot(raw_losses, label='Untransformed')
    plt.plot(scaled_losses, label='Standardized')
    plt.xscale('symlog', linthresh=1)
    plt.xlabel('Iterations (symlog scale)')
    plt.ylabel('Training MSE (MPa squared)')
    plt.title('Scaling & Convergence Speed')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/scaling_convergence.png', dpi=150)
    plt.close()
