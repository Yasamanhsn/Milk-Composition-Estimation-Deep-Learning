from TrainNetwork.LoadRawCsvData import load_raw_csv_data
from TrainNetwork.SignalToScalogram import signal_to_scalogram
from TrainNetwork.SaveScalogramAsImage import save_scalogram_as_image
import os


def load_signal_and_convert_to_image(dataset_folder_name, csv_file_name, target_file_name):
    """
    Load raw data from a CSV file, convert signals to scalograms, and save them as images.

    Parameters:
    dataset_folder_name (str): Name of the dataset folder.
    csv_file_name (str): Name of the input CSV file.
    target_file_name (str): Name of the target file.

    Returns:
    None
    """
    # Get the folder path
    folder_path = os.path.join(os.getcwd(), dataset_folder_name)

    # Load raw CSV data
    vsb_sig, uv_sig = load_raw_csv_data(csv_file_name, target_file_name)

    # Process each row in the data
    for i in range(len(vsb_sig)):
        # Convert signals to scalograms
        scalogram_vsb_sig = signal_to_scalogram(vsb_sig[i, :])
        scalogram_uv_sig = signal_to_scalogram(uv_sig[i, :])

        # Save scalograms as images
        save_scalogram_as_image(scalogram_vsb_sig, scalogram_uv_sig, folder_path, i + 1)
