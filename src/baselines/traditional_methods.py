"""
Traditional Computer Vision Methods for Tamper Detection
Baseline methods for comparison with deep learning approaches
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from scipy import ndimage
from skimage.feature import local_binary_pattern


class ErrorLevelAnalysis:
    """
    Error Level Analysis (ELA) for detecting JPEG compression artifacts
    
    Principle: Tampered regions often have different compression levels
    than the rest of the image, revealing themselves through ELA.
    """
    
    def __init__(self, quality: int = 95):
        """
        Args:
            quality: JPEG quality for re-compression (default: 95)
        """
        self.quality = quality
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Perform Error Level Analysis
        
        Args:
            image: Input image (H, W, 3) in BGR
        
        Returns:
            ela_map: Error level map (H, W) normalized to [0, 1]
        """
        # Save and re-compress image
        _, encoded = cv2.imencode(
            '.jpg',
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Calculate difference
        diff = cv2.absdiff(image, recompressed)
        
        # Convert to grayscale and normalize
        ela_map = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        ela_map = ela_map.astype(np.float32) / 255.0
        
        # Enhance contrast
        ela_map = np.clip(ela_map * 10, 0, 1)
        
        return ela_map


class NoiseAnalysis:
    """
    Noise inconsistency detection
    
    Principle: Authentic images have consistent noise patterns,
    while tampered regions may have different noise characteristics.
    """
    
    def __init__(self, block_size: int = 64):
        """
        Args:
            block_size: Size of blocks for noise analysis
        """
        self.block_size = block_size
    
    def _estimate_noise(self, block: np.ndarray) -> float:
        """Estimate noise level in a block using median absolute deviation"""
        # Apply high-pass filter
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        filtered = cv2.filter2D(block, -1, kernel)
        
        # Estimate noise using MAD
        sigma = np.median(np.abs(filtered)) / 0.6745
        return sigma
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect noise inconsistencies
        
        Args:
            image: Input image (H, W, 3) in BGR
        
        Returns:
            noise_map: Noise inconsistency map (H, W) normalized to [0, 1]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calculate noise for each block
        noise_map = np.zeros((h, w), dtype=np.float32)
        noise_levels = []
        
        for y in range(0, h - self.block_size, self.block_size // 2):
            for x in range(0, w - self.block_size, self.block_size // 2):
                block = gray[y:y+self.block_size, x:x+self.block_size]
                if block.shape[0] == self.block_size and block.shape[1] == self.block_size:
                    noise = self._estimate_noise(block)
                    noise_levels.append(noise)
                    noise_map[y:y+self.block_size, x:x+self.block_size] = noise
        
        # Calculate deviation from median noise
        median_noise = np.median(noise_levels)
        noise_map = np.abs(noise_map - median_noise)
        
        # Normalize
        if noise_map.max() > 0:
            noise_map = noise_map / noise_map.max()
        
        return noise_map


class SIFTBasedDetection:
    """
    SIFT-based copy-move detection
    
    Principle: Detect duplicated regions using SIFT feature matching
    """
    
    def __init__(
        self,
        n_features: int = 1000,
        match_threshold: float = 0.7,
        min_matches: int = 10
    ):
        """
        Args:
            n_features: Maximum number of SIFT features
            match_threshold: Ratio test threshold for matching
            min_matches: Minimum matches to consider as copy-move
        """
        self.n_features = n_features
        self.match_threshold = match_threshold
        self.min_matches = min_matches
        self.sift = cv2.SIFT_create(nfeatures=n_features)
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect copy-move forgery using SIFT
        
        Args:
            image: Input image (H, W, 3) in BGR
        
        Returns:
            mask: Binary mask of suspicious regions (H, W)
            info: Dictionary with detection information
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect and compute SIFT features
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 2:
            return np.zeros(gray.shape, dtype=np.uint8), {'matches': 0}
        
        # Match features with themselves
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors, descriptors, k=2)
        
        # Apply ratio test and filter self-matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Not a self-match and passes ratio test
                if m.queryIdx != m.trainIdx and m.distance < self.match_threshold * n.distance:
                    # Check if keypoints are far apart (not just nearby features)
                    pt1 = keypoints[m.queryIdx].pt
                    pt2 = keypoints[m.trainIdx].pt
                    dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    if dist > 50:  # Minimum distance threshold
                        good_matches.append(m)
        
        # Create mask from matched keypoints
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        if len(good_matches) >= self.min_matches:
            for match in good_matches:
                pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
                pt2 = tuple(map(int, keypoints[match.trainIdx].pt))
                cv2.circle(mask, pt1, 20, 255, -1)
                cv2.circle(mask, pt2, 20, 255, -1)
        
        # Dilate to connect nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask, {
            'total_keypoints': len(keypoints),
            'good_matches': len(good_matches),
            'suspicious': len(good_matches) >= self.min_matches
        }


class LocalBinaryPatternAnalysis:
    """
    Local Binary Pattern (LBP) for texture inconsistency detection
    
    Principle: Tampered regions may have different texture patterns
    """
    
    def __init__(self, radius: int = 3, n_points: int = 24):
        """
        Args:
            radius: Radius of LBP
            n_points: Number of points in LBP
        """
        self.radius = radius
        self.n_points = n_points
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect texture inconsistencies using LBP
        
        Args:
            image: Input image (H, W, 3) in BGR
        
        Returns:
            inconsistency_map: Texture inconsistency map (H, W)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute LBP
        lbp = local_binary_pattern(
            gray,
            self.n_points,
            self.radius,
            method='uniform'
        )
        
        # Compute local LBP histograms
        h, w = gray.shape
        block_size = 32
        inconsistency_map = np.zeros((h, w), dtype=np.float32)
        
        # Reference histogram (from center region)
        center_y, center_x = h // 2, w // 2
        ref_block = lbp[
            center_y - block_size:center_y + block_size,
            center_x - block_size:center_x + block_size
        ]
        ref_hist, _ = np.histogram(ref_block, bins=self.n_points + 2, range=(0, self.n_points + 2))
        ref_hist = ref_hist.astype(np.float32)
        ref_hist /= (ref_hist.sum() + 1e-8)
        
        # Compare each block to reference
        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = lbp[y:y+block_size, x:x+block_size]
                hist, _ = np.histogram(block, bins=self.n_points + 2, range=(0, self.n_points + 2))
                hist = hist.astype(np.float32)
                hist /= (hist.sum() + 1e-8)
                
                # Chi-square distance
                distance = np.sum((hist - ref_hist)**2 / (hist + ref_hist + 1e-8))
                inconsistency_map[y:y+block_size, x:x+block_size] = distance
        
        # Normalize
        if inconsistency_map.max() > 0:
            inconsistency_map = inconsistency_map / inconsistency_map.max()
        
        return inconsistency_map


class TraditionalEnsemble:
    """
    Ensemble of traditional methods for robust detection
    """
    
    def __init__(self):
        self.ela = ErrorLevelAnalysis()
        self.noise = NoiseAnalysis()
        self.sift = SIFTBasedDetection()
        self.lbp = LocalBinaryPatternAnalysis()
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Run all traditional methods and combine results
        
        Args:
            image: Input image (H, W, 3) in BGR
        
        Returns:
            combined_map: Combined detection map (H, W) in [0, 1]
            results: Dictionary with individual results
        """
        # Run all methods
        ela_map = self.ela.detect(image)
        noise_map = self.noise.detect(image)
        sift_mask, sift_info = self.sift.detect(image)
        lbp_map = self.lbp.detect(image)
        
        # Normalize SIFT mask
        sift_map = sift_mask.astype(np.float32) / 255.0
        
        # Combine with weighted average
        weights = {
            'ela': 0.3,
            'noise': 0.2,
            'sift': 0.3,
            'lbp': 0.2
        }
        
        combined_map = (
            weights['ela'] * ela_map +
            weights['noise'] * noise_map +
            weights['sift'] * sift_map +
            weights['lbp'] * lbp_map
        )
        
        # Normalize to [0, 1]
        combined_map = np.clip(combined_map, 0, 1)
        
        results = {
            'ela_map': ela_map,
            'noise_map': noise_map,
            'sift_map': sift_map,
            'sift_info': sift_info,
            'lbp_map': lbp_map,
            'combined_map': combined_map
        }
        
        return combined_map, results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is not None:
            print(f"Analyzing: {image_path}")
            
            # Run ensemble
            ensemble = TraditionalEnsemble()
            combined_map, results = ensemble.detect(image)
            
            print(f"ELA max: {results['ela_map'].max():.3f}")
            print(f"Noise max: {results['noise_map'].max():.3f}")
            print(f"SIFT matches: {results['sift_info']['good_matches']}")
            print(f"LBP max: {results['lbp_map'].max():.3f}")
            print(f"Combined max: {combined_map.max():.3f}")
            
            # Save results
            cv2.imwrite('ela_result.png', (results['ela_map'] * 255).astype(np.uint8))
            cv2.imwrite('combined_result.png', (combined_map * 255).astype(np.uint8))
            print("Results saved!")
        else:
            print(f"Could not load image: {image_path}")
    else:
        print("Usage: python traditional_methods.py <image_path>")
