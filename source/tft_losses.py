"""
Custom loss functions and metrics for TFT model.
File: tft_losses.py

Usage:
    from tft_losses import (
        quantile_loss_with_leakage_weight_fixed,
        masked_ia_metric,
        masked_ia_metric_leakage,
        accuracy,
        prepare_regression_targets,
        LEAKAGE_WEIGHT
    )
"""

import tensorflow as tf
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
LEAKAGE_WEIGHT = 2.0  # Default value - change this as needed


# =============================================================================
# Loss Functions
# =============================================================================
def quantile_loss_with_leakage_weight(y_true, y_pred, leakage_weight=LEAKAGE_WEIGHT):
    """
    Quantile loss with leakage weighting and -1 masking.
    
    Args:
        y_true: (batch, time, 2) - [:,:,0] is target value, [:,:,1] is leakage class
        y_pred: (batch, time, 3) - predictions for quantiles [0.05, 0.5, 0.95]
        leakage_weight: Weight multiplier for leakage samples
    """
    quantiles = tf.constant([0.05, 0.5, 0.95], dtype=y_pred.dtype)
    
    y_value = y_true[:, :, 0]
    leakage_class = y_true[:, :, 1]
    
    y_value = tf.expand_dims(y_value, axis=-1)
    y_value = tf.tile(y_value, [1, 1, 3])
    
    # Mask for -1 padding
    mask = tf.not_equal(y_value, -1)
    mask = tf.cast(mask, dtype=y_pred.dtype)
    
    # Quantile loss
    error = y_value - y_pred
    loss = tf.maximum(quantiles * error, (quantiles - 1) * error)
    loss = loss * mask
    
    # Apply leakage weighting
    leakage_weight_tensor = tf.expand_dims(leakage_class, axis=-1)
    leakage_weight_tensor = tf.tile(leakage_weight_tensor, [1, 1, 3])
    weights = 1.0 + (leakage_weight - 1.0) * leakage_weight_tensor
    loss = loss * weights
    
    # Normalize
    loss = tf.reduce_sum(loss, axis=[1, 2])
    valid_steps = tf.reduce_sum(mask, axis=[1, 2])
    normalized_loss = tf.reduce_mean(loss / (valid_steps + 1e-6))
    
    return normalized_loss


def quantile_loss_with_leakage_weight_fixed(y_true, y_pred):
    """Fixed version of quantile loss using global LEAKAGE_WEIGHT."""
    return quantile_loss_with_leakage_weight(y_true, y_pred, leakage_weight=LEAKAGE_WEIGHT)


# =============================================================================
# Metrics
# =============================================================================
def masked_ia_metric(y_true, y_pred):
    """
    Index of Agreement (IA) metric with -1 masking.
    Uses median prediction (index 1).
    """
    y_value = y_true[:, :, 0]
    mask = tf.not_equal(y_value, -1)
    y_pred_median = y_pred[:, :, 1]
    
    y_value_masked = tf.boolean_mask(y_value, mask)
    y_pred_masked = tf.boolean_mask(y_pred_median, mask)
    
    mean_true = tf.reduce_mean(y_value_masked)
    ia_numerator = tf.reduce_sum(tf.square(y_value_masked - y_pred_masked))
    ia_denominator = tf.reduce_sum(
        tf.square(tf.abs(y_pred_masked - mean_true) + tf.abs(y_value_masked - mean_true))
    )
    ia = 1 - (ia_numerator / (ia_denominator + 1e-6))
    
    return ia


def masked_ia_metric_leakage(y_true, y_pred):
    """
    Index of Agreement (IA) metric for leakage samples only.
    Only computes IA where leakage_class == 1.
    """
    y_value = y_true[:, :, 0]
    leakage_class = y_true[:, :, 1]
    
    # Mask: valid data AND leakage class
    mask = tf.logical_and(tf.not_equal(y_value, -1), tf.equal(leakage_class, 1))
    
    y_pred_median = y_pred[:, :, 1]
    
    y_value_masked = tf.boolean_mask(y_value, mask)
    y_pred_masked = tf.boolean_mask(y_pred_median, mask)
    
    def compute_ia():
        mean_true = tf.reduce_mean(y_value_masked)
        ia_numerator = tf.reduce_sum(tf.square(y_value_masked - y_pred_masked))
        ia_denominator = tf.reduce_sum(
            tf.square(tf.abs(y_pred_masked - mean_true) + tf.abs(y_value_masked - mean_true))
        )
        ia = 1 - (ia_numerator / (ia_denominator + 1e-6))
        return ia
    
    # Handle case where no leakage samples exist
    ia = tf.cond(
        tf.size(y_value_masked) > 0,
        compute_ia,
        lambda: tf.constant(0.0, dtype=y_pred.dtype)
    )
    
    return ia


def accuracy(y_true, y_pred):
    """Binary accuracy for classification output."""
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)


# =============================================================================
# Data Preparation
# =============================================================================
def prepare_regression_targets(y_values, leakage_classes):
    """
    Prepare regression targets by concatenating values with leakage classes.
    
    Args:
        y_values: (batch, time, 1) - target values
        leakage_classes: (batch,) or (batch, 1) - leakage class labels
        
    Returns:
        y_regression: (batch, time, 2) - concatenated targets
    """
    leakage_classes = np.squeeze(leakage_classes)
    leakage_classes_expanded = np.expand_dims(leakage_classes, axis=-1)
    
    time_steps = y_values.shape[1]
    leakage_classes_tiled = np.repeat(leakage_classes_expanded, time_steps, axis=1)
    leakage_classes_tiled = np.expand_dims(leakage_classes_tiled, axis=-1)
    
    y_regression = np.concatenate([y_values, leakage_classes_tiled], axis=-1)
    return y_regression


# =============================================================================
# Helper for model loading
# =============================================================================
def get_custom_losses_and_metrics():
    """
    Returns dict of custom objects for model loading.
    
    Usage:
        custom_objects = get_custom_losses_and_metrics()
        model = tf.keras.models.load_model('model.h5', custom_objects=custom_objects)
    """
    return {
        'quantile_loss_with_leakage_weight_fixed': quantile_loss_with_leakage_weight_fixed,
        'masked_ia_metric': masked_ia_metric,
        'masked_ia_metric_leakage': masked_ia_metric_leakage,
        'accuracy': accuracy,
    }
