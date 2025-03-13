import torch
import os
import json


class ThresholdManager:
    def __init__(self, initial_threshold=0.65, min_threshold=0.4, stagnation_epochs=10, save_fir=None):
        self.image_log = {}
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.stagnation_epochs = stagnation_epochs
        self.current_threshold = initial_threshold
        