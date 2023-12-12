from sklearn.model_selection import train_test_split
import pickle

def preprocess_data(images, targets,is_use_from_saved_data=True):
    """
    Preprocess image data and corresponding targets for training and testing.

    Parameters:
    images (numpy.ndarray): Array of images.
    targets (numpy.ndarray): Array of target values.

    Returns:
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray: Preprocessed data split into training and testing sets.
    """
    if not is_use_from_saved_data:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42)

        # Normalize images
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # Save train data
        with open('X_train.pickle', 'wb') as f:
            pickle.dump(X_train, f)

        with open('y_train.pickle', 'wb') as f:
            pickle.dump(y_train, f)

        # Save test data
        with open('X_test.pickle', 'wb') as f:
            pickle.dump(X_test, f)

        with open('y_test.pickle', 'wb') as f:
            pickle.dump(y_test, f)

    if is_use_from_saved_data:
        # Load train data
        with open('X_train.pickle', 'rb') as f:
            X_train = pickle.load(f)

        with open('y_train.pickle', 'rb') as f:
            y_train = pickle.load(f)

        # Load test data
        with open('X_test.pickle', 'rb') as f:
            X_test = pickle.load(f)

        with open('y_test.pickle', 'rb') as f:
            y_test = pickle.load(f)

    return X_train, X_test, y_train, y_test
