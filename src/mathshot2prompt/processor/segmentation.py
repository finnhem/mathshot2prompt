"""OpenCV-based image segmentation functions."""

import cv2
import numpy as np
import time


def visualize_regions(image, regions, filename='debug_regions.png'):
    """Draw boxes around detected regions for debugging."""
    debug_img = image.copy()
    # Convert to BGR if grayscale for colored rectangles
    if len(debug_img.shape) == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in regions:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(filename, debug_img)
    print(f"Debug visualization saved to {filename}")


def preprocess_image(image):
    """Normalize image to black text on white background."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Determine if image has light text on dark background
    mean_value = cv2.mean(gray)[0]
    is_dark_background = mean_value < 128
    
    if is_dark_background:
        # Invert image for light text on dark background
        gray = cv2.bitwise_not(gray)
    
    # Save debug image of preprocessed result
    cv2.imwrite('debug_preprocessed.png', gray)
    print("Debug preprocessed image saved to debug_preprocessed.png")
    
    return gray


def segment_regions(image):
    """Segment regions from image."""
    print("🔍 Analyzing image structure...")
    start_time = time.time()
    
    # Preprocess image to normalize to black text on white background
    normalized_image = preprocess_image(image)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        normalized_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Invert to get white text on black for morphology
        11,  # Block size
        2    # C constant
    )
    
    # Save binary image for debugging
    cv2.imwrite('debug_binary.png', binary)
    print("Debug binary image saved to debug_binary.png")
    
    # Create kernels for morphological operations
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Apply morphological operations
    dilated_h = cv2.dilate(binary, kernel_h, iterations=2)
    dilated_v = cv2.dilate(binary, kernel_v, iterations=2)
    
    # Combine horizontal and vertical components
    combined = cv2.bitwise_or(dilated_h, dilated_v)
    
    # Clean up with small square kernel
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_sq)
    
    # Save cleaned image for debugging
    cv2.imwrite('debug_cleaned.png', cleaned)
    print("Debug cleaned image saved to debug_cleaned.png")
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and process regions
    regions = []
    min_area = 50  # Minimum area threshold
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Add padding around region
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(normalized_image.shape[1] - x, w + 2*padding)
            h = min(normalized_image.shape[0] - y, h + 2*padding)
            regions.append((x, y, w, h))
    
    # Sort regions top-to-bottom
    regions.sort(key=lambda r: r[1])
    
    # Merge overlapping or nearby regions
    merged_regions = []
    if regions:
        current_region = list(regions[0])
        
        for region in regions[1:]:
            x, y, w, h = region
            curr_x, curr_y, curr_w, curr_h = current_region
            
            # Check if regions should be merged
            vertical_overlap = (y <= curr_y + curr_h + 10) and (curr_y <= y + h + 10)
            horizontal_overlap = (x <= curr_x + curr_w + 20) and (curr_x <= x + w + 20)
            
            if vertical_overlap and horizontal_overlap:
                # Merge regions
                x1 = min(curr_x, x)
                y1 = min(curr_y, y)
                x2 = max(curr_x + curr_w, x + w)
                y2 = max(curr_y + curr_h, y + h)
                current_region = [x1, y1, x2 - x1, y2 - y1]
            else:
                merged_regions.append(tuple(current_region))
                current_region = list(region)
        
        merged_regions.append(tuple(current_region))
    
    # Save debug visualization using normalized image
    visualize_regions(normalized_image, merged_regions)
    
    print(f"✨ Analysis completed ({time.time() - start_time:.2f}s)")
    print(f"📊 Found {len(merged_regions)} regions")
    
    # Return both the regions and the normalized image for OCR
    return merged_regions, normalized_image 