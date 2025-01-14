"""OpenCV-based image segmentation functions."""

import cv2
import numpy as np
import time


def visualize_regions(image, regions, filename='debug_regions.png'):
    """Draw boxes around detected regions for debugging."""
    debug_img = image.copy()
    for x, y, w, h in regions:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(filename, debug_img)
    print(f"Debug visualization saved to {filename}")


def segment_regions(image):
    """Segment image into regions using OpenCV."""
    start_time = time.time()
    print("🔍 Analyzing image structure...")
    
    # Get image dimensions for scaling parameters
    height, width = image.shape[:2]
    scale_factor = min(width, height) / 1000.0  # Base scale on smaller dimension
    
    # Scale parameters based on image size but with higher minimum values
    kernel_h_width = max(25, int(40 * scale_factor))  # Increased from 15 to 25
    kernel_v_height = max(10, int(20 * scale_factor))  # Increased from 7 to 10
    min_area = max(50, int(100 * scale_factor))  # Increased from 25 to 50
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Create different kernels for different purposes
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_h_width, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_v_height))
    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Increased from 3x3 to 5x5
    
    # Process horizontal and vertical components separately
    # Horizontal processing with more iterations
    h_dilate = cv2.dilate(thresh, kernel_h, iterations=3)  # Increased from 2 to 3
    h_erode = cv2.erode(h_dilate, kernel_h, iterations=2)  # Increased from 1 to 2
    
    # Vertical processing with more iterations
    v_dilate = cv2.dilate(thresh, kernel_v, iterations=3)  # Increased from 2 to 3
    v_erode = cv2.erode(v_dilate, kernel_v, iterations=2)  # Increased from 1 to 2
    
    # Combine horizontal and vertical
    combined = cv2.bitwise_or(h_erode, v_erode)
    
    # More aggressive cleanup with square kernel
    processed = cv2.dilate(combined, kernel_sq, iterations=3)  # Increased from 2 to 3
    processed = cv2.erode(processed, kernel_sq, iterations=2)  # Increased from 1 to 2
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)
    
    # Scale thresholds based on image size with higher base values
    h_gap_max = max(30, int(60 * scale_factor))  # Increased from 20 to 30
    h_gap_text = max(60, int(120 * scale_factor))  # Increased from 40 to 60
    v_gap_max = max(25, int(50 * scale_factor))  # Increased from 15 to 25
    min_height = max(20, int(35 * scale_factor))  # Increased from 15 to 20
    
    # Filter and process components
    regions = []
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # More strict filtering criteria
        aspect_ratio = h / w if w > 0 else 0
        is_vector = (aspect_ratio > 2.0 and h > min_height * 1.2) or (w < h_gap_text * 0.8 and h > min_height)
        is_bracket = (aspect_ratio > 2.5 and h > min_height * 1.5) or (w < h_gap_max * 0.8 and h > min_height * 1.2)
        
        if area > min_area or is_vector or is_bracket:
            regions.append([x, y, w, h])
    
    # Sort regions top-to-bottom, left-to-right
    regions.sort(key=lambda b: (b[1], b[0]))
    
    # More aggressive merging
    merged = []
    while regions:
        current = regions.pop(0)
        merged_with_existing = False
        
        for existing in merged:
            # Calculate distances and overlaps
            h_gap = abs((current[0] + current[2]/2) - (existing[0] + existing[2]/2))
            v_gap = abs((current[1] + current[3]/2) - (existing[1] + existing[3]/2))
            
            # Calculate vertical overlap with more tolerance
            y_overlap = max(0, min(current[1] + current[3], existing[1] + existing[3]) - 
                             max(current[1], existing[1]))
            y_overlap_ratio = y_overlap / min(current[3], existing[3])  # Changed from max to min
            
            # Calculate horizontal overlap
            x_overlap = max(0, min(current[0] + current[2], existing[0] + existing[2]) - 
                             max(current[0], existing[0]))
            
            # Calculate density of the gap region to detect text
            gap_x1 = min(current[0] + current[2], existing[0] + existing[2])
            gap_x2 = max(current[0], existing[0])
            if gap_x1 < gap_x2:  # There's a gap between components
                gap_y1 = max(current[1], existing[1])
                gap_y2 = min(current[1] + current[3], existing[1] + existing[3])
                if gap_y2 > gap_y1:  # There's vertical overlap
                    gap_region = thresh[gap_y1:gap_y2, gap_x1:gap_x2]
                    gap_density = np.sum(gap_region > 0) / (gap_region.size + 1e-6)
                else:
                    gap_density = 0
            else:
                gap_density = 0
            
            # More aggressive merging conditions
            should_merge = False
            
            # Same line merging with more tolerance
            if y_overlap_ratio > 0.2 and h_gap < h_gap_text * 3.0:  # More lenient
                # Allow more merging even with some text
                if gap_density < 0.15 or h_gap < h_gap_max * 1.5:  # More lenient
                    should_merge = True
            # Vector and bracket merging with more tolerance
            elif ((x_overlap > 0 and v_gap < v_gap_max * 3) or  # More vertical tolerance
                  (h_gap < h_gap_max * 2.0 and y_overlap > 0) or  # More horizontal tolerance
                  (h_gap < h_gap_max * 1.5 and v_gap < v_gap_max * 3.5)):  # More overall tolerance
                should_merge = True
            
            if should_merge:
                # Merge the regions
                existing[0] = min(current[0], existing[0])
                existing[1] = min(current[1], existing[1])
                existing[2] = max(current[0] + current[2], existing[0] + existing[2]) - existing[0]
                existing[3] = max(current[1] + current[3], existing[1] + existing[3]) - existing[1]
                merged_with_existing = True
                break
        
        if not merged_with_existing:
            merged.append(current)

    # Save debug visualization
    visualize_regions(image, merged)

    print(f"✨ Analysis completed ({time.time() - start_time:.2f}s)")
    print(f"📊 Found {len(merged)} regions")
    return merged 