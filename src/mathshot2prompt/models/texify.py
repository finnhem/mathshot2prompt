"""Texify model loading and inference functions."""

import torch
from texify.inference import batch_inference as texify_batch_inference
from texify.model.model import load_model as texify_load_model, GenerateVisionEncoderDecoderModel
from texify.model.processor import load_processor as texify_load_processor


def load_model():
    """Load the Texify model."""
    return texify_load_model()


def load_processor():
    """Load the Texify processor."""
    return texify_load_processor()


def batch_inference(images, model, processor):
    """Run batch inference on a list of images."""
    return texify_batch_inference(images, model, processor) 