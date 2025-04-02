import os
import json
import torch
import numpy as np

class ThresholdManager:
    def __init__(self, initial_threshold=0.65, min_threshold=0.4, stagnation_epochs=10, save_dir=None, lookback_distance=15):
        self.image_log = {}
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.stagnation_epochs = stagnation_epochs
        self.current_threshold = initial_threshold
        self.save_dir = save_dir
        self.lookback_distance = lookback_distance

    def initialize_log(self, image_ids):
        for img_id in image_ids:
            if img_id not in self.image_log.keys():
                self.image_log[img_id] = {
                    'true_ratio_history': [],
                    'threshold': self.initial_threshold,
                    'stagnation_count': 0,
                    'unsupervised_loss': []
                }
    
    def get_threshold(self, img_ids):
        return torch.tensor([self.image_log[img_id]['threshold'] for img_id in img_ids])
    
    def update_log(self, image_ids, true_ratio, unsupervised_loss, epoch):
        for i, img_id in enumerate(image_ids):
            # true_ratio = true_ratio.item()

            self.image_log[img_id]['true_ratio_history'].append(true_ratio)
            self.image_log[img_id]['unsupervised_loss'].append(unsupervised_loss)
            
            if len(self.image_log[img_id]['true_ratio_history']) >= self.lookback_distance:
                recent_true_ratio = self.image_log[img_id]['true_ratio_history'][-self.lookback_distance:]
                recent_unsupervised_loss = self.image_log[img_id]['unsupervised_loss'][-self.lookback_distance:]
                ratio_change = np.average(np.diff(recent_true_ratio))
                error_change = np.average(np.diff(recent_unsupervised_loss))

                # 정체 조건을 일단 이렇게 판단을 해봄.
                if ratio_change < 0.01 and error_change < 0.05:
                    self.image_log[img_id]['stagnation_count'] += 1
                else:
                    self.image_log[img_id]['stagnation_count'] = 0

                if self.image_log[img_id]['stagnation_count'] >= self.stagnation_epochs:
                    current_threshold = self.image_log[img_id]['threshold']
                    new_threshold = max(self.min_threshold, current_threshold - 0.1*(1 - true_ratio))
                    self.image_log[img_id]['threshold'] = new_threshold
                    self.image_log[img_id]['stagnation_count'] = 0
                    print(f"Epoch {epoch+1}: Adjusted threshold for {img_id} from {current_threshold:.2f} to {new_threshold:.2f}")
    
    def save_log(self):
        if self.save_dir:
            with open(os.path.join(self.save_dir, 'threshold_log.json'), 'w') as f:
                json.dump(self.image_log, f)


class EntropyThresholdManager:
    def __init__(self, initial_threshold=0.00089, max_threshold=0.00095, stagnation_epochs=10, save_dir=None, lookback_distance=15):
        self.image_log = {}
        self.initial_threshold = initial_threshold
        self.max_threshold = max_threshold
        self.stagnation_epochs = stagnation_epochs
        self.current_threshold = initial_threshold
        self.save_dir = save_dir
        self.lookback_distance = lookback_distance

    def initialize_log(self, image_ids):
        for img_id in image_ids:
            if img_id not in self.image_log.keys():
                self.image_log[img_id] = {
                    'true_ratio_history': [],
                    'threshold': self.initial_threshold,
                    'stagnation_count': 0,
                    'unsupervised_loss': []
                }
    
    def get_threshold(self, img_ids):
        return torch.tensor([self.image_log[img_id]['threshold'] for img_id in img_ids])
    
    def update_log(self, image_ids, true_ratio, unsupervised_loss, epoch):
        for i, img_id in enumerate(image_ids):
            # true_ratio = true_ratio.item()

            self.image_log[img_id]['true_ratio_history'].append(true_ratio)
            self.image_log[img_id]['unsupervised_loss'].append(unsupervised_loss)
            
            if len(self.image_log[img_id]['true_ratio_history']) >= self.lookback_distance:
                recent_true_ratio = self.image_log[img_id]['true_ratio_history'][-self.lookback_distance:]
                recent_unsupervised_loss = self.image_log[img_id]['unsupervised_loss'][-self.lookback_distance:]
                ratio_change = np.average(np.diff(recent_true_ratio))
                error_change = np.average(np.diff(recent_unsupervised_loss))

                # 정체 조건을 일단 이렇게 판단을 해봄.
                if ratio_change < 0.01 and error_change < 0.05:
                    self.image_log[img_id]['stagnation_count'] += 1
                else:
                    self.image_log[img_id]['stagnation_count'] = 0

                if self.image_log[img_id]['stagnation_count'] >= self.stagnation_epochs:
                    current_threshold = self.image_log[img_id]['threshold']
                    new_threshold = min(self.max_threshold, current_threshold + 0.00001)
                    self.image_log[img_id]['threshold'] = new_threshold
                    self.image_log[img_id]['stagnation_count'] = 0
                    print(f"Epoch {epoch+1}: Adjusted threshold for {img_id} from {current_threshold:.2f} to {new_threshold:.2f}")
    
    def save_log(self):
        if self.save_dir:
            with open(os.path.join(self.save_dir, 'threshold_log.json'), 'w') as f:
                json.dump(self.image_log, f)

