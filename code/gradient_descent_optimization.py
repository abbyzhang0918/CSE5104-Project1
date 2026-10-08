import numpy as np
import pandas as pd


def load_data():
    data = pd.read_excel('data/Concrete_Data.xls')
    train_data = pd.concat([data.iloc[:500], data.iloc[630:]])
    test_data = data.iloc[500:630]

    X_train = train_data.iloc[:, :8].to_numpy()
    y_train = train_data.iloc[:, 8].to_numpy()
    X_test = test_data.iloc[:, :8].to_numpy()
    y_test = test_data.iloc[:, 8].to_numpy()
    return X_train, y_train, X_test, y_test


def gradient_descent(X, y, learning_rate, iterations=20000):
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    losses = []

    predictions = X @ weights + bias
    losses.append(np.mean((predictions - y) ** 2))

    for i in range(iterations):
        predictions = X @ weights + bias
        errors = predictions - y

        weight_gradient = (2 / n_samples) * (X.T @ errors)
        bias_gradient = 2 * np.mean(errors)

        weights = weights - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

        predictions = X @ weights + bias
        mse = np.mean((predictions - y) ** 2)
        if not np.isfinite(mse):
            raise ValueError('Loss is not finite. Try a smaller learning rate.')
        losses.append(mse)

    return weights, bias, losses


def evaluate(y, predictions):
    mse = np.mean((y - predictions) ** 2)
    r2 = 1 - mse / np.var(y)
    return mse, r2


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    weights, bias, losses = gradient_descent(X_train, y_train, 0.0000003)
    train_mse, train_r2 = evaluate(y_train, X_train @ weights + bias)
    test_mse, test_r2 = evaluate(y_test, X_test @ weights + bias)

    print('Training samples:', len(y_train))
    print('Testing samples:', len(y_test))
    print('Weights:', weights)
    print('Bias:', bias)
    print('Training MSE:', train_mse, 'Training R2:', train_r2)
    print('Testing MSE:', test_mse, 'Testing R2:', test_r2)
