import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats


class RobustDisparityGenerator:
    """
    Multiple disparity predictions에서 statistical consensus와 RANSAC outlier detection을 통해
    robust한 pseudo label을 생성하는 클래스 (multiscale 제거)
    """
    
    def __init__(self, 
                 variance_thresh: float = 0.75,           # Statistical filtering threshold
                 ransac_inlier_thresh: float = 0.6,      # RANSAC inlier threshold
                 min_consensus_ratio: float = 0.6,       # Minimum consensus ratio for RANSAC
                 min_agreement_ratio: float = 0.1,       # Minimum final agreement ratio
                 use_median_fallback: bool = False,        # Use median when mean fails
                 confidence_weighting: bool = False,       # Enable confidence-based weighting
                 zero_outside_agreement: bool = False):
        """
        Args:
            variance_thresh: Maximum allowed variance for statistical consensus
            ransac_inlier_thresh: RANSAC inlier threshold in pixels
            min_consensus_ratio: Minimum ratio of predictions that must agree
            min_agreement_ratio: Minimum ratio of pixels that must have agreement
            use_median_fallback: Use median when variance is too high
            confidence_weighting: Weight predictions by confidence if available
            zero_outside_agreement: Set non-agreement regions to 0
        """
        self.variance_thresh = variance_thresh
        self.ransac_inlier_thresh = ransac_inlier_thresh
        self.min_consensus_ratio = min_consensus_ratio
        self.min_agreement_ratio = min_agreement_ratio
        self.use_median_fallback = use_median_fallback
        self.confidence_weighting = confidence_weighting
        self.zero_outside_agreement = zero_outside_agreement
    
    def generate_robust_disparity(self, 
                                data_batch: Dict[str, Any],
                                use_main_fallback: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Main function to generate robust averaged disparity from multiple predictions
        """
        
        # Extract main prediction
        main_pred = data_batch['pseudo_disp'][0].squeeze().detach().cpu()
        H, W = main_pred.shape
        
        # Extract random crops and coordinates
        random_crops, coords_list = self._extract_random_crops(data_batch)
        if len(random_crops) == 0:
            return self._handle_no_crops(main_pred)
        
        # print(f"Processing {len(random_crops)} random crops with Statistical + RANSAC consensus")
        
        # Convert all predictions to full size
        all_predictions = self._convert_to_full_size(main_pred, random_crops, coords_list)
        
        # Extract confidence maps if available
        confidence_maps = self._extract_confidence_maps(data_batch, all_predictions)
        
        # Stage 1: Statistical Consensus with Variance Filtering
        statistical_pred, statistical_mask, var_stats = self._compute_statistical_consensus(all_predictions)
        
        # Stage 2: RANSAC-style Outlier Detection
        ransac_pred, ransac_mask, ransac_stats = self._compute_ransac_consensus(all_predictions)
        
        # Stage 3: Combine Statistical + RANSAC consensus methods (multiscale 제거)
        final_pred, final_mask, method_info = self._combine_consensus_methods(
            all_predictions, statistical_pred, ransac_pred,
            statistical_mask, ransac_mask,
            confidence_maps, main_pred, use_main_fallback
        )
        
        # Calculate final statistics
        total_valid_pixels = (main_pred > 0).sum().item()
        agreement_pixels = final_mask.sum().item()
        agreement_ratio = agreement_pixels / max(total_valid_pixels, 1)
        
        info = {
            'num_crops': len(random_crops),
            'agreement_ratio': agreement_ratio,
            'agreement_pixels': agreement_pixels,
            'total_valid_pixels': total_valid_pixels,
            'variance_stats': var_stats,
            'ransac_stats': ransac_stats,
            'method_info': method_info,
            'final_method': method_info['primary_method']
        }
        
        return final_pred, final_mask, info
    
    def _compute_statistical_consensus(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Statistical consensus using variance-based filtering"""
        
        stacked_preds = torch.stack(predictions, dim=0)  # [N, H, W]
        
        # Compute statistics
        mean_pred = stacked_preds.mean(dim=0)
        variance = stacked_preds.var(dim=0)
        std_dev = variance.sqrt()
        
        # Create mask for low-variance regions
        variance_mask = (variance <= self.variance_thresh) & (mean_pred > 0)
        
        # Median fallback for high-variance regions
        if self.use_median_fallback:
            median_pred = stacked_preds.median(dim=0)[0]
            # Use median where variance is too high
            final_pred = torch.where(variance <= self.variance_thresh, mean_pred, median_pred)
        else:
            final_pred = mean_pred
        
        stats_info = {
            'mean_variance': variance[variance_mask].mean().item() if variance_mask.sum() > 0 else 0,
            'max_variance': variance.max().item(),
            'variance_coverage': variance_mask.sum().item() / variance_mask.numel(),
            'median_fallback_used': self.use_median_fallback
        }
        
        return final_pred, variance_mask, stats_info
    
    def _compute_ransac_consensus(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """RANSAC-inspired consensus for outlier detection"""
        
        n_predictions = len(predictions)
        H, W = predictions[0].shape
        
        best_consensus_mask = torch.zeros((H, W), dtype=torch.bool)
        best_consensus_count = 0
        best_inlier_indices = []
        
        # Try each prediction as reference
        for ref_idx in range(n_predictions):
            ref_pred = predictions[ref_idx]
            
            # Count inliers for this reference
            consensus_votes = torch.zeros((H, W), dtype=torch.float)
            inlier_indices = [ref_idx]
            
            for i, pred in enumerate(predictions):
                if i != ref_idx:
                    diff = torch.abs(pred - ref_pred)
                    is_inlier = (diff <= self.ransac_inlier_thresh) | (pred <= 0) | (ref_pred <= 0)
                    consensus_votes += is_inlier.float()
                    
                    # Track which predictions are inliers
                    if is_inlier.float().mean() > 0.8:  # If most pixels agree
                        inlier_indices.append(i)
            
            # Check if consensus ratio is met
            consensus_ratio = consensus_votes / max(n_predictions - 1, 1)
            consensus_mask = (consensus_ratio >= self.min_consensus_ratio) & (ref_pred > 0)
            consensus_count = consensus_mask.sum()
            
            if consensus_count > best_consensus_count:
                best_consensus_mask = consensus_mask
                best_consensus_count = consensus_count
                best_inlier_indices = inlier_indices
        
        # Compute robust average using only inlier predictions
        if len(best_inlier_indices) > 0:
            inlier_preds = torch.stack([predictions[i] for i in best_inlier_indices], dim=0)
            ransac_pred = inlier_preds.mean(dim=0)
        else:
            ransac_pred = torch.stack(predictions, dim=0).mean(dim=0)
        
        ransac_stats = {
            'best_consensus_count': best_consensus_count.item(),
            'num_inliers': len(best_inlier_indices),
            'inlier_ratio': len(best_inlier_indices) / n_predictions,
            'consensus_coverage': best_consensus_mask.sum().item() / best_consensus_mask.numel()
        }
        
        return ransac_pred, best_consensus_mask, ransac_stats
    
    def _combine_consensus_methods(self, all_predictions, statistical_pred, ransac_pred,
                                 statistical_mask, ransac_mask, 
                                 confidence_maps, main_pred, use_main_fallback):
        """Combine Statistical + RANSAC consensus methods (multiscale 제거)"""
        
        # Combine Statistical + RANSAC agreement masks only
        combined_mask = statistical_mask & ransac_mask
        
        # Calculate agreement ratio
        total_valid = (main_pred > 0).sum().item()
        agreement_pixels = combined_mask.sum().item()
        agreement_ratio = agreement_pixels / max(total_valid, 1)
        
        method_info = {
            'statistical_coverage': statistical_mask.sum().item() / statistical_mask.numel(),
            'ransac_coverage': ransac_mask.sum().item() / ransac_mask.numel(),
            'combined_coverage': agreement_ratio
        }
        
        if agreement_ratio >= self.min_agreement_ratio:
            # Use robust consensus
            if self.confidence_weighting and confidence_maps is not None:
                final_pred = self._compute_confidence_weighted_average(
                    all_predictions, confidence_maps, combined_mask
                )
                method_info['primary_method'] = 'confidence_weighted'
            else:
                # Choose best consensus method based on coverage
                if statistical_mask.sum() >= ransac_mask.sum():
                    final_pred = statistical_pred
                    method_info['primary_method'] = 'statistical'
                else:
                    final_pred = ransac_pred
                    method_info['primary_method'] = 'ransac'
            
            # Apply zero outside agreement if requested
            if self.zero_outside_agreement:
                final_pred = final_pred * combined_mask.float()
            
        else:
            # Fallback strategies
            if use_main_fallback and not self.zero_outside_agreement:
                final_pred = main_pred.clone()
                combined_mask = torch.ones_like(main_pred, dtype=torch.bool)
                method_info['primary_method'] = 'main_fallback'
            else:
                final_pred = torch.zeros_like(main_pred)
                combined_mask = torch.zeros_like(main_pred, dtype=torch.bool)
                method_info['primary_method'] = 'insufficient_agreement'
        
        return final_pred, combined_mask, method_info
    
    def _compute_confidence_weighted_average(self, predictions, confidence_maps, mask):
        """Compute confidence-weighted average"""
        
        if confidence_maps is None or len(confidence_maps) != len(predictions):
            # Fallback to simple average
            stacked_preds = torch.stack(predictions, dim=0)
            return stacked_preds.mean(dim=0)
        
        # Normalize confidence maps
        stacked_confs = torch.stack(confidence_maps, dim=0)
        stacked_preds = torch.stack(predictions, dim=0)
        
        # Apply softmax to confidence for proper weighting
        weights = F.softmax(stacked_confs, dim=0)
        
        # Compute weighted average
        weighted_pred = (weights * stacked_preds).sum(dim=0)
        
        return weighted_pred
    
    def _convert_to_full_size(self, main_pred, random_crops, coords_list):
        """Convert all predictions to full size"""
        
        all_predictions = [main_pred]
        
        for pred_crop, (x, y) in zip(random_crops, coords_list):
            pred_full = torch.zeros_like(main_pred)
            crop_h, crop_w = pred_crop.shape
            pred_full[y:y+crop_h, x:x+crop_w] = pred_crop
            all_predictions.append(pred_full)
        
        return all_predictions
    
    def _extract_confidence_maps(self, data_batch, all_predictions):
        """Extract confidence maps if available"""
        
        if not self.confidence_weighting:
            return None
        
        confidence_maps = []
        
        # Main confidence map
        if 'tgt_confidence_map_s' in data_batch:
            main_conf = data_batch['tgt_confidence_map_s'].squeeze().detach().cpu()
            if main_conf.shape != all_predictions[0].shape:
                main_conf = F.interpolate(
                    main_conf.unsqueeze(0).unsqueeze(0),
                    size=all_predictions[0].shape,
                    mode='bilinear', align_corners=False
                ).squeeze()
            confidence_maps.append(main_conf)
        else:
            confidence_maps.append(torch.ones_like(all_predictions[0]))
        
        # Random crop confidence maps (if available)
        for i in range(1, len(all_predictions)):
            if f'tgt_confidence_random_{i}' in data_batch:
                conf_crop = data_batch[f'tgt_confidence_random_{i}'].squeeze().detach().cpu()
                # Convert to full size similar to predictions
                conf_full = torch.zeros_like(all_predictions[0])
                # This would need crop coordinates - simplified for now
                confidence_maps.append(torch.ones_like(all_predictions[0]))
            else:
                confidence_maps.append(torch.ones_like(all_predictions[0]))
        
        return confidence_maps if len(confidence_maps) == len(all_predictions) else None
    
    def _handle_no_crops(self, main_pred):
        """Handle case when no random crops are available"""
        
        if self.zero_outside_agreement:
            return torch.zeros_like(main_pred), torch.zeros_like(main_pred, dtype=torch.bool), {
                'num_crops': 0,
                'agreement_ratio': 0.0,
                'final_method': 'no_crops_zero_output'
            }
        else:
            return main_pred, torch.ones_like(main_pred, dtype=torch.bool), {
                'num_crops': 0,
                'agreement_ratio': 1.0,
                'final_method': 'main_only'
            }
    
    def _extract_random_crops(self, data_batch: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """Extract all random crop predictions and their coordinates"""
        random_crops = []
        coords_list = []
        
        i = 1
        while f'pseudo_disp_random_{i}' in data_batch and f'tgt_random_coord_{i}' in data_batch:
            pred_random = data_batch[f'pseudo_disp_random_{i}'][0].squeeze().detach().cpu()
            x = data_batch[f'tgt_random_coord_{i}'][0]
            y = data_batch[f'tgt_random_coord_{i}'][1]
            
            random_crops.append(pred_random)
            coords_list.append((x, y))
            i += 1
        
        return random_crops, coords_list


# Backward compatibility wrapper
class AverageDisparityGenerator(RobustDisparityGenerator):
    """Backward compatibility wrapper"""
    
    def __init__(self, 
                 agreement_thresh: float = 0.6,
                 min_agreement_ratio: float = 0.1,
                 zero_outside_agreement: bool = True):
        
        # Convert old parameters to new robust parameters
        super().__init__(
            variance_thresh=agreement_thresh,
            ransac_inlier_thresh=agreement_thresh * 1.5,
            min_consensus_ratio=0.6,
            min_agreement_ratio=min_agreement_ratio,
            zero_outside_agreement=zero_outside_agreement
        )
    
    def generate_averaged_disparity(self, data_batch, use_main_fallback=True):
        """Backward compatibility method"""
        return self.generate_robust_disparity(data_batch, use_main_fallback)


class RobustPseudoLabelGenerator:
    """
    Enhanced Pseudo Label Generator using Statistical + RANSAC consensus methods
    """
    
    def __init__(self, 
                 variance_thresh: float = 1.0,
                 ransac_inlier_thresh: float = 1.2,
                 min_consensus_ratio: float = 0.6,
                 min_agreement_ratio: float = 0.1,
                 confidence_threshold: float = 0.8,
                 max_disparity: float = 192.0):
        
        self.disparity_generator = RobustDisparityGenerator(
            variance_thresh=variance_thresh,
            ransac_inlier_thresh=ransac_inlier_thresh,
            min_consensus_ratio=min_consensus_ratio,
            min_agreement_ratio=min_agreement_ratio
        )
        self.confidence_threshold = confidence_threshold
        self.max_disparity = max_disparity
    
    def generate_pseudo_labels(self, 
                             data_batch: Dict[str, Any],
                             apply_confidence_filter: bool = True) -> Dict[str, torch.Tensor]:
        """Generate robust pseudo labels with Statistical + RANSAC filtering"""
        
        # Generate robust disparity
        robust_disparity, agreement_mask, info = self.disparity_generator.generate_robust_disparity(data_batch)
        
        # Initialize confidence mask
        confidence_mask = torch.ones_like(robust_disparity, dtype=torch.bool)
        
        # Apply confidence filtering if requested
        if apply_confidence_filter and 'tgt_confidence_map_s' in data_batch:
            confidence_map = data_batch['tgt_confidence_map_s'].squeeze().detach().cpu()
            
            if confidence_map.shape != robust_disparity.shape:
                confidence_map = F.interpolate(
                    confidence_map.unsqueeze(0).unsqueeze(0), 
                    size=robust_disparity.shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
            
            confidence_mask = confidence_map > self.confidence_threshold
        
        # Apply disparity range filtering
        valid_range_mask = (robust_disparity >= 0) & (robust_disparity <= self.max_disparity)
        
        # Combine all masks
        final_mask = agreement_mask & confidence_mask & valid_range_mask
        
        # Apply mask to pseudo disparity
        pseudo_disparity = robust_disparity.clone()
        pseudo_disparity[~final_mask] = 0
        
        return {
            'pseudo_disparity': pseudo_disparity,
            'confidence_mask': confidence_mask,
            'agreement_mask': agreement_mask,
            'final_mask': final_mask,
            'info': info
        }


# 사용 예시
def example_usage_robust(data_batch):
    """Statistical + RANSAC 방법 사용 예시"""
    
    # 1. Robust disparity generation
    robust_gen = RobustDisparityGenerator(
        variance_thresh=1.0,
        ransac_inlier_thresh=1.2,
        min_consensus_ratio=0.6,
        min_agreement_ratio=0.1
    )
    
    robust_disparity, agreement_mask, info = robust_gen.generate_robust_disparity(data_batch)
    
    print(f"Generated robust disparity with {info['num_crops']} crops")
    print(f"Agreement ratio: {info['agreement_ratio']:.3f}")
    print(f"Method used: {info['final_method']}")
    print(f"RANSAC inliers: {info['ransac_stats']['num_inliers']}")
    print(f"Statistical coverage: {info['method_info']['statistical_coverage']:.3f}")
    print(f"RANSAC coverage: {info['method_info']['ransac_coverage']:.3f}")
    
    return robust_disparity, agreement_mask, info


# 간단한 인터페이스
def generate_robust_pseudo_label(data_batch: Dict[str, Any]) -> torch.Tensor:
    """간단한 인터페이스로 Statistical + RANSAC robust disparity 생성"""
    generator = RobustDisparityGenerator()
    robust_disparity, _, _ = generator.generate_robust_disparity(data_batch)
    return robust_disparity
