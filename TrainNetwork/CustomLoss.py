import tensorflow as tf

def weighted_mae_loss_function(y_true, y_pred):
    """
    Compute a weighted Mean Absolute Error (MAE) loss across multiple outputs.

    Parameters:
    y_true (tensor-like): Ground truth values as a tensor.
    y_pred (tensor-like): Predicted values as a tensor.

    Returns:
    float: Weighted MAE loss computed across multiple outputs.
    """
    loss1 = tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))
    loss2 = tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))
    loss3 = tf.reduce_mean(tf.abs(y_true[:, 2] - y_pred[:, 2]))

    total_loss = (loss1 + 3 * loss2 + loss3) / 3
    return total_loss

def custom_loss(y_true, y_pred):
    """
    Combine various loss functions to create a custom loss, primarily incorporating a weighted MAE loss.

    Parameters:
    y_true (tensor-like): Ground truth values as a tensor.
    y_pred (tensor-like): Predicted values as a tensor.

    Returns:
    float: Custom combined loss value computed from individual loss functions.
    """
    mae_loss = weighted_mae_loss_function(y_true, y_pred)
    total_loss = mae_loss
    return total_loss

