import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from gradient_descent_optimization import load_data, evaluate


feature_names = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water',
                 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    raw_model = LinearRegression()
    raw_model.fit(X_train, y_train)
    scaled_model = LinearRegression()
    scaled_model.fit(X_train_scaled, y_train)

    converted_weights = scaled_model.coef_ / std
    converted_bias = scaled_model.intercept_ - np.sum(scaled_model.coef_ * mean / std)

    rows = []
    for i in range(8):
        rows.append([feature_names[i], raw_model.coef_[i],
                     scaled_model.coef_[i], converted_weights[i]])
    rows.append(['Intercept', raw_model.intercept_,
                 scaled_model.intercept_, converted_bias])
    coefficients = pd.DataFrame(rows, columns=[
        'feature', 'untransformed_coef', 'standardized_coef', 'standardized_coef_in_raw_units'])
    print('Optimal coefficients (LinearRegression):')
    print(coefficients.to_string(index=False))

    raw_train_mse, raw_train_r2 = evaluate(y_train, raw_model.predict(X_train))
    raw_test_mse, raw_test_r2 = evaluate(y_test, raw_model.predict(X_test))
    scaled_train_mse, scaled_train_r2 = evaluate(
        y_train, scaled_model.predict(X_train_scaled))
    scaled_test_mse, scaled_test_r2 = evaluate(
        y_test, scaled_model.predict(X_test_scaled))

    summary = pd.DataFrame([
        ['Untransformed', raw_train_mse, raw_test_mse, raw_train_r2, raw_test_r2],
        ['Standardized', scaled_train_mse, scaled_test_mse, scaled_train_r2, scaled_test_r2]
    ], columns=['model', 'training_mse', 'testing_mse', 'training_r2', 'testing_r2'])
    print('\nPerformance of the two optimal models:')
    print(summary.to_string(index=False))

    difference = raw_model.predict(X_test) - scaled_model.predict(X_test_scaled)
    print('\nLargest difference between test predictions:', np.max(np.abs(difference)))
