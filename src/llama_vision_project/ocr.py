import cv2
from PIL import Image
import pytesseract
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
import numpy as np
import time

class OCRProcessor:
    def __init__(self):
        """Initialize OCR processor with necessary configs."""
        start_time = time.time()
        print("Initializing OCR processor...")
        
        # Configure tesseract path for Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Initialize Texify model and processor
        print("Loading Texify model...")
        self.model = load_model()
        self.processor = load_processor()
        print(f"Initialization completed in {time.time() - start_time:.2f}s")

    def visualize_regions(self, image, text_regions, math_regions, filename='debug_regions.png'):
        """Draw boxes around detected regions for debugging."""
        debug_img = image.copy()
        
        # Draw text regions in green
        for x, y, w, h in text_regions:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw math regions in red
        for x, y, w, h in math_regions:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        cv2.imwrite(filename, debug_img)
        print(f"Debug visualization saved to {filename}")

    def merge_boxes(self, boxes):
        """Merge overlapping or nearby boxes."""
        if not boxes:
            return []

        # Sort boxes by y-coordinate first, then x-coordinate
        sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        merged = []
        current_line = [list(sorted_boxes[0])]
        current_y = sorted_boxes[0][1]

        # Adjusted tolerances
        vertical_tolerance = 20  # Reduced vertical tolerance
        horizontal_tolerance = 30  # Reduced horizontal tolerance for more precise merging

        for box in sorted_boxes[1:]:
            x, y, w, h = box
            
            # Check if box is on the same line
            if abs(y - current_y) < vertical_tolerance:
                # Check if box is close to any box in current line
                merge_happened = False
                for curr_box in current_line:
                    curr_x, curr_y, curr_w, curr_h = curr_box
                    
                    # Calculate horizontal and vertical gaps
                    h_gap = x - (curr_x + curr_w) if x > curr_x + curr_w else curr_x - (x + w)
                    v_gap = abs(y - curr_y)
                    
                    # More strict merging conditions
                    if h_gap < horizontal_tolerance and v_gap < vertical_tolerance:
                        # Update current box to encompass both
                        left = min(curr_x, x)
                        top = min(curr_y, y)
                        right = max(curr_x + curr_w, x + w)
                        bottom = max(curr_y + curr_h, y + h)
                        curr_box[0] = left
                        curr_box[1] = top
                        curr_box[2] = right - left
                        curr_box[3] = bottom - top
                        merge_happened = True
                        break
                
                if not merge_happened:
                    current_line.append(list(box))
            else:
                # Add current line to merged boxes and start new line
                merged.extend(current_line)
                current_line = [list(box)]
                current_y = y

        # Add last line
        merged.extend(current_line)
        return merged

    def segment_math_text(self, image):
        """Segment image into text and math regions using OpenCV."""
        start_time = time.time()
        print("Segmenting image...")
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Perform morphological operations to connect components
        # Use a wider kernel to better connect equation parts
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        kernel_separate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        
        # Connect horizontal components (equations)
        dilated_h = cv2.dilate(thresh, kernel_connect, iterations=1)
        eroded_h = cv2.erode(dilated_h, kernel_connect, iterations=1)
        
        # Separate vertical components (different lines)
        dilated_v = cv2.dilate(eroded_h, kernel_separate, iterations=1)
        processed = cv2.erode(dilated_v, kernel_separate, iterations=1)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
        
        # Filter and process components
        boxes = []
        min_area = 100  # Minimum area to consider
        
        for i in range(1, num_labels):  # Skip background (label 0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area > min_area:
                boxes.append((x, y, w, h))
        
        # Sort boxes by position (top to bottom, left to right)
        boxes.sort(key=lambda b: (b[1], b[0]))
        
        # Merge boxes that are likely part of the same component
        merged_boxes = []
        if boxes:
            current_box = list(boxes[0])
            
            for box in boxes[1:]:
                x, y, w, h = box
                curr_x, curr_y, curr_w, curr_h = current_box
                
                # Check if boxes are close enough to merge
                h_gap = x - (curr_x + curr_w)
                v_gap = abs(y - curr_y)
                
                # More sophisticated merging conditions
                should_merge = False
                
                # Check for horizontal alignment (same line)
                same_line = v_gap < 20
                
                # Check for vertical alignment (like in matrices)
                vertical_overlap = (y < curr_y + curr_h and y + h > curr_y)
                
                # Check for matrix-like structure
                matrix_pattern = (vertical_overlap and h_gap < 50) or \
                               (same_line and h_gap < 30)
                
                # Check if boxes are part of the same equation
                equation_pattern = same_line and h_gap < 100 and \
                                 (curr_w / curr_h > 1.5 or w / h > 1.5)
                
                if matrix_pattern or equation_pattern:
                    # Update current box to encompass both
                    left = min(curr_x, x)
                    top = min(curr_y, y)
                    right = max(curr_x + curr_w, x + w)
                    bottom = max(curr_y + curr_h, y + h)
                    current_box[0] = left
                    current_box[1] = top
                    current_box[2] = right - left
                    current_box[3] = bottom - top
                else:
                    merged_boxes.append(tuple(current_box))
                    current_box = list(box)
            
            merged_boxes.append(tuple(current_box))

        text_regions = []
        math_regions = []

        # Classify regions
        for x, y, w, h in merged_boxes:
            region = image[y:y+h, x:x+w]
            region_text = pytesseract.image_to_string(region, config='--psm 6')
            
            # More precise math detection
            is_math = any([
                # Matrix patterns
                (w > 50 and h > 50 and any(c in region_text for c in ['(', ')', '[', ']', '{', '}'])),
                # Equation patterns
                ('=' in region_text or ':=' in region_text),
                # Special math symbols
                any(c in region_text for c in ['∈', '∥', '∗', '×', '√', '∫']),
                # Vector/matrix notation
                ('R' in region_text and any(c in region_text for c in ['^', '×', '2', '3'])),
                # Fraction patterns
                ('/' in region_text or '\\frac' in region_text or '\\sqrt' in region_text),
                # Matrix content patterns
                (w > 100 and h > 50 and any(str(i) in region_text for i in range(10))),
                # Equation operators
                any(c in region_text for c in ['+', '-', '*', '/', '^', '_', '||'])
            ])
            
            # Additional checks for matrices
            if not is_math and w > 50 and h > 50:
                roi = thresh[y:y+h, x:x+w]
                # Count horizontal and vertical lines
                horizontal_profile = np.sum(roi, axis=1)
                vertical_profile = np.sum(roi, axis=0)
                h_lines = len(np.where(horizontal_profile > roi.shape[1] * 0.5)[0])
                v_lines = len(np.where(vertical_profile > roi.shape[0] * 0.5)[0])
                
                if h_lines > 2 or v_lines > 2:  # Matrix-like structure detected
                    is_math = True

            if is_math:
                math_regions.append((x, y, w, h))
            else:
                text_regions.append((x, y, w, h))

        # Save debug visualization
        self.visualize_regions(image, text_regions, math_regions)

        print(f"Segmentation completed in {time.time() - start_time:.2f}s")
        print(f"Found {len(text_regions)} text regions and {len(math_regions)} math regions")
        return text_regions, math_regions

    def process_image(self, pil_image):
        """Process a PIL Image directly."""
        try:
            # Convert PIL image to OpenCV format
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Segment image into text and math regions
            text_regions, math_regions = self.segment_math_text(image)

            results = {
                'text': [],
                'equations': []
            }

            # Process text regions with Tesseract
            if text_regions:
                start_time = time.time()
                print("Processing text regions with Tesseract...")
                for x, y, w, h in text_regions:
                    region = pil_image.crop((x, y, x+w, y+h))
                    text = pytesseract.image_to_string(region)
                    if text.strip():
                        results['text'].append(text.strip())
                print(f"Text processing completed in {time.time() - start_time:.2f}s")

            # Process math regions with Texify
            if math_regions:
                start_time = time.time()
                print("Processing math regions with Texify...")
                for x, y, w, h in math_regions:
                    region = pil_image.crop((x, y, x+w, y+h))
                    results_texify = batch_inference([region], self.model, self.processor)
                    results['equations'].append(results_texify[0])
                print(f"Math processing completed in {time.time() - start_time:.2f}s")

            return results

        except Exception as e:
            print(f"❌ OCR error: {str(e)}")
            return {'text': [], 'equations': []} 