def build_results_dir(model_name):
    """
    Builds and returns a directory structure for storing model results.

    Parameters:
    model_name (str): Name of the model used for creating the directory structure.

    Returns:
    str: Path to the directory where model results will be stored based on the given model_name.
    """
    import os

    # Root directory for storing results
    root_result = 'results'

    # Create root directory if it doesn't exist
    if not os.path.exists(root_result):
        os.mkdir(root_result)

    # Directory path based on model_name
    results_dir = os.path.join(root_result, model_name)

    # Create model-specific directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    return results_dir
