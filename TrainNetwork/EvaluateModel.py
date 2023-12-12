import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from TrainNetwork.BuildResultsDir import build_results_dir

def evaluate_model(model, X_test, y_test, model_name,metric):
    """
    Evaluate the model using test data and generate metrics and plots.

    Parameters:
    model (tf.keras.Model): Trained model to be evaluated.
    X_test (array-like): Test data features.
    y_test (array-like): True labels for the test data.
    model_name (str): Name of the model being evaluated.
    """
    # Create results directory
    results_dir = build_results_dir(model_name)

    # Generate predictions using the test set
    predictions_ts = model.predict(X_test)

    # Initialize lists to store individual metric values
    total_mse, total_mae, total_r2 = [], [], []
    total_min_err, total_max_err, total_std_err, total_percentage_err = [], [], [], []

    # Define path for metrics CSV file
    csv_path = os.path.join(results_dir, 'metrics.csv')

    # Write metrics to a CSV file
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)

        # Write headers for the CSV file
        headers = ["Output", "MSE", "MAE", "R2", "Min Error", "Max Error", "STD Error", "Percentage Error"]
        writer.writerow(headers)

        # Calculate and write metrics for each output
        for i in range(3):
            y_true = y_test[:, i]
            y_pred = predictions_ts[:, i]

            err = np.abs(y_pred - y_true)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            min_err = np.min(err)
            max_err = np.max(err)
            std_err = np.std(err)
            percentage_err = (sum(err) / len(err)) * 100

            # Append values to respective lists
            total_mse.append(mse)
            total_mae.append(mae)
            total_r2.append(r2)
            total_min_err.append(min_err)
            total_max_err.append(max_err)
            total_std_err.append(std_err)
            total_percentage_err.append(percentage_err)

            # Print individual metrics
            print(f"Output {i + 1}:")
            print(f"Mean Squared Error (MSE): {mse}")
            print(f"Mean Absolute Error (MAE): {mae}")
            print(f"R-squared (R2) Score: {r2}")
            print(f"Minimum Error : {min_err}")
            print(f"Maximum Error: {max_err}")
            print(f"STD of Error: {std_err}")
            print(f"Percentage of Error: {percentage_err}")
            print("--------------")

            # Write metrics to the CSV row
            row = [i + 1, mse, mae, r2, min_err, max_err, std_err, percentage_err]
            writer.writerow(row)

    # Print aggregated metrics
    print(f"Min Error of all: {np.min(total_min_err)}")
    print(f"Max Error of all: {np.max(total_max_err)}")
    print(f"Standard Deviation of all variables: {np.average(total_std_err)}")
    print(f"Total Percentage Error: {np.average(total_percentage_err)}")

    # Write aggregated metrics to the existing CSV file
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)

        # Append headers for aggregated metrics
        headers = ["Overall Output", "Min Error of all", "Max Error of all", "Average STD Error",
                   "Average Percentage Error","TotalParams","ComputationalTime"]
        writer.writerow(headers)

        # Append aggregated row
        agg_row = ["----", np.min(total_min_err), np.max(total_max_err),
                   np.average(total_std_err), np.average(total_percentage_err),metric['num_params'],metric['train_time']]

        writer.writerow(agg_row)

    # Generate plots for each output
    for i in range(3):
        plt.figure(figsize=(15, 10), dpi=250)
        plt.rcParams['axes.facecolor'] = 'lightgray'  # Set background color

        # Plot real vs predicted values
        plt.plot(y_test[:, i], label='Real values')
        plt.plot(predictions_ts[:, i], label='Predicted values')
        plt.legend()
        plt.grid(which='major', color='#dddddd')
        plt.grid(which='minor', color='#eeeeee')
        plt.minorticks_on()

        # Save the plot
        plot_path = os.path.join(results_dir, str(i) + '_plot.png')
        plt.savefig(plot_path)
        plt.close()
