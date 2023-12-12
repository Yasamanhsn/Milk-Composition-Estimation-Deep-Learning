import pandas as pd

def load_raw_csv_data(file_path, target_file_name):
    """
    Load raw data from a CSV file, create target data, and separate visible and UV signals.

    Parameters:
    file_path (str): Path to the input CSV file.
    target_file_name (str): Name for the target file to be created.

    Returns:
    numpy.ndarray, numpy.ndarray: Arrays containing visible and UV signals.
    """
    # Load data from CSV file
    data = pd.read_csv(file_path)

    # Create the target data from the loaded data
    data.iloc[:, :3].to_csv(target_file_name, index=False)

    # Get the input signal
    data_slice = data.values[:, 3:]
    kd = 288
    kf = kd

    # Separate the visible and UV signals
    visible_signal = data_slice[:, :kd]
    uv_signal = data_slice[:, kf:]

    return visible_signal, uv_signal
