print("📚 Loading libraries...")

import cv2
print("✓ OpenCV")
from PIL import Image
print("✓ PIL")
import pytesseract
print("✓ Tesseract")
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue
from threading import Event
import torch
print("✓ PyTorch")

print("⚡ Loading Texify (this may take a moment)...")
from texify.inference import batch_inference
from texify.model.model import load_model, GenerateVisionEncoderDecoderModel
from texify.model.processor import load_processor
print("✓ Texify")

import os
from pathlib import Path
import torch.serialization

print("✨ All libraries loaded\n")

class OCRProcessor:
    # Class variable to store cached model
    _cached_model = None
    _cached_processor = None
    
    def __init__(self):
        """Initialize OCR processor with necessary configs."""
        self.initialization_done = Event()
        self.model = None
        self.processor = None
        self.tesseract_ready = False
        
        # Start only Tesseract initialization in background
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.init_future = self.executor.submit(self._init_tesseract)
        self.initialization_done.set()  # Mark basic initialization as done

    @classmethod
    def _load_or_get_cached_model(cls):
        """Load model from cache or initialize new one."""
        if cls._cached_model is not None and cls._cached_processor is not None:
            print("🚀 Using cached model from memory")
            return cls._cached_model, cls._cached_processor
            
        # Create models directory if it doesn't exist
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "texify_model.pt"
        processor_path = model_dir / "texify_processor.pt"
        
        # Try to load from disk first
        if model_path.exists() and processor_path.exists():
            try:
                print("💾 Loading model from disk...")
                torch.serialization.add_safe_globals([GenerateVisionEncoderDecoderModel])
                
                # Load with weights_only=False to handle custom classes
                model = torch.load(model_path, weights_only=False)
                processor = torch.load(processor_path, weights_only=False)
                print("✅ Model loaded from disk successfully")
                cls._cached_model = model
                cls._cached_processor = processor
                return model, processor
            except Exception as e:
                print(f"⚠️  Failed to load from disk: {str(e).split('. ')[0]}")
        
        # If not in memory or disk, load from scratch
        print("⚡ Initializing model from scratch (first time)...")
        torch.backends.cudnn.benchmark = True
        
        model = load_model()
        model = model.cuda()
        processor = load_processor()
        
        # Save to disk for faster loading next time
        try:
            print("💾 Saving model to disk...")
            torch.serialization.add_safe_globals([GenerateVisionEncoderDecoderModel])
            
            torch.save(model, model_path, pickle_protocol=5)
            torch.save(processor, processor_path, pickle_protocol=5)
            print("✅ Model saved successfully")
        except Exception as e:
            print(f"⚠️  Failed to save to disk: {str(e).split('. ')[0]}")
        
        # Cache in memory
        cls._cached_model = model
        cls._cached_processor = processor
        
        return model, processor
    
    def _init_texify(self):
        """Initialize Texify model and processor."""
        try:
            self.model, self.processor = self._load_or_get_cached_model()
            return self.model, self.processor
        except Exception as e:
            print(f"Error loading Texify model: {e}")
            raise
    
    def _init_components(self):
        """Initialize components in background."""
        print("Initializing OCR processor...")
        start_time = time.time()
        
        # Start both initializations in parallel
        texify_future = self.executor.submit(self._init_texify)
        tesseract_future = self.executor.submit(self._init_tesseract)
        
        # Wait for both to complete
        self.model, self.processor = texify_future.result()
        self.tesseract_ready = tesseract_future.result()
        
        print(f"Initialization completed in {time.time() - start_time:.2f}s")
        self.initialization_done.set()
    
    def _init_tesseract(self):
        """Initialize Tesseract OCR."""
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        try:
            pytesseract.get_languages()
            return True
        except Exception as e:
            print(f"Warning: Tesseract initialization issue: {e}")
            return False

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
        vertical_tolerance = 20  # For line detection
        horizontal_tolerance = 40  # Increased for equation parts
        equation_tolerance = 150  # For long equations with equals signs

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
                    
                    # Check for equation patterns in the regions
                    region1 = thresh[curr_y:curr_y+curr_h, curr_x:curr_x+curr_w]
                    region2 = thresh[y:y+h, x:x+w]
                    text1 = pytesseract.image_to_string(region1, config='--psm 6')
                    text2 = pytesseract.image_to_string(region2, config='--psm 6')
                    
                    # Conditions for merging
                    basic_merge = h_gap < horizontal_tolerance and v_gap < vertical_tolerance
                    equation_merge = h_gap < equation_tolerance and ('=' in text1 or '=' in text2)
                    matrix_merge = h_gap < equation_tolerance and any(c in text1 + text2 for c in ['Q', 'R', 'b']) and \
                                 any(c in text1 + text2 for c in ['(', ')', '[', ']', '{', '}'])
                    
                    if basic_merge or equation_merge or matrix_merge:
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
                # Before starting new line, check for matrix continuation
                merge_with_current = False
                for curr_box in current_line:
                    curr_x, curr_y, curr_w, curr_h = curr_box
                    # Check for vertical matrix patterns
                    if abs(x - curr_x) < horizontal_tolerance and y - (curr_y + curr_h) < vertical_tolerance:
                        # Merge vertically for matrices
                        curr_box[3] = y + h - curr_y
                        merge_with_current = True
                        break
                
                if not merge_with_current:
                    # Add current line to merged boxes and start new line
                    merged.extend(current_line)
                    current_line = [list(box)]
                    current_y = y

        # Add last line
        merged.extend(current_line)

        # Second pass to merge nearby equation components
        final_boxes = []
        for box in merged:
            merged_with_existing = False
            x, y, w, h = box
            region_text = pytesseract.image_to_string(thresh[y:y+h, x:x+w], config='--psm 6')
            
            for existing_box in final_boxes:
                ex, ey, ew, eh = existing_box
                existing_text = pytesseract.image_to_string(thresh[ey:ey+eh, ex:ex+ew], config='--psm 6')
                
                # Check if boxes might be part of the same equation
                h_gap = abs((x + w/2) - (ex + ew/2))
                v_gap = abs((y + h/2) - (ey + eh/2))
                
                should_merge = False
                # Merge equation parts
                if ('=' in region_text or '=' in existing_text) and h_gap < equation_tolerance and v_gap < vertical_tolerance:
                    should_merge = True
                # Merge matrix definitions
                elif any(c in region_text + existing_text for c in ['Q', 'R', 'b']) and h_gap < equation_tolerance:
                    should_merge = True
                
                if should_merge:
                    # Update existing box to encompass both
                    existing_box[0] = min(x, ex)
                    existing_box[1] = min(y, ey)
                    existing_box[2] = max(x + w, ex + ew) - existing_box[0]
                    existing_box[3] = max(y + h, ey + eh) - existing_box[1]
                    merged_with_existing = True
                    break
            
            if not merged_with_existing:
                final_boxes.append(list(box))

        return final_boxes

    def segment_math_text(self, image):
        """Segment image into text and math regions using OpenCV."""
        start_time = time.time()
        print("🔍 Analyzing image structure...")
        
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
                
                # Check for horizontal alignment (same line)
                same_line = v_gap < 20
                
                # Check for vertical alignment (like in matrices)
                vertical_overlap = (y < curr_y + curr_h and y + h > curr_y)
                
                # Initial merge conditions based on geometry
                should_merge = (same_line and h_gap < 30) or \
                             (vertical_overlap and h_gap < 50)
                
                if should_merge:
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

        # Initial classification based on geometric properties
        text_regions = []
        math_regions = []

        for x, y, w, h in merged_boxes:
            roi = thresh[y:y+h, x:x+w]
            
            # Calculate region properties
            aspect_ratio = w / float(h)
            density = np.sum(roi > 0) / (w * h)
            
            # Count horizontal and vertical lines
            horizontal_profile = np.sum(roi, axis=1)
            vertical_profile = np.sum(roi, axis=0)
            h_lines = len(np.where(horizontal_profile > roi.shape[1] * 0.5)[0])
            v_lines = len(np.where(vertical_profile > roi.shape[0] * 0.5)[0])
            
            # Classify based on geometric features
            is_math = any([
                # Matrix-like structure
                (h_lines > 2 or v_lines > 2),
                # Wide regions likely equations
                (w > 100 and aspect_ratio > 1.5),
                # Dense regions with specific aspect ratio (matrices)
                (density > 0.15 and 0.5 < aspect_ratio < 2.0 and w > 50 and h > 50),
                # Very wide regions (likely equations)
                (w > 200)
            ])

            if is_math:
                math_regions.append((x, y, w, h))
            else:
                text_regions.append((x, y, w, h))

        # Save debug visualization
        self.visualize_regions(image, text_regions, math_regions)

        print(f"✨ Analysis completed ({time.time() - start_time:.2f}s)")
        print(f"📊 Found {len(text_regions)} text and {len(math_regions)} math regions")
        return text_regions, math_regions

    def _ensure_model_loaded(self):
        """Ensure model is loaded before processing math regions."""
        if self.model is None or self.processor is None:
            self.model, self.processor = self._load_or_get_cached_model()

    def process_image(self, pil_image):
        """Process a PIL Image directly."""
        try:
            start_time = time.time()
            
            # Check image size and resize if too large
            max_dimension = 2000
            w, h = pil_image.size
            if w > max_dimension or h > max_dimension:
                print(f"📐 Resizing {w}x{h} → ", end='')
                scale = max_dimension / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                print(f"{new_w}x{new_h}")
            
            # Convert and segment
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            text_regions, math_regions = self.segment_math_text(image)
            
            results = {
                'text': [],
                'equations': []
            }
            
            # Process regions in parallel
            futures = []
            
            # Process text regions
            if text_regions:
                print(f"📝 Processing {len(text_regions)} text regions...")
                text_start = time.time()
                for x, y, w, h in text_regions:
                    region = pil_image.crop((x, y, x+w, y+h))
                    future = self.executor.submit(pytesseract.image_to_string, region)
                    futures.append(('text', future))

            # Process math regions
            if math_regions:
                print(f"🔢 Processing {len(math_regions)} math regions...")
                math_start = time.time()
                
                self._ensure_model_loaded()
                
                # Process in batches
                batch_size = 4
                for i in range(0, len(math_regions), batch_size):
                    batch_regions = []
                    batch_indices = []
                    
                    for j, (x, y, w, h) in enumerate(math_regions[i:i + batch_size]):
                        region = pil_image.crop((x, y, x+w, y+h))
                        batch_regions.append(region)
                        batch_indices.append(i + j)
                    
                    if batch_regions:
                        future = self.executor.submit(
                            batch_inference, batch_regions, self.model, self.processor
                        )
                        futures.append(('math_batch', (future, batch_indices)))
                        
                        progress = min((i + batch_size), len(math_regions))
                        print(f"⚡ Processing batch {i//batch_size + 1}/{(len(math_regions) + batch_size - 1)//batch_size}")
                
            # Collect results as they complete
            text_results = []
            math_results = [None] * len(math_regions)  # Pre-allocate list
            
            for item in futures:
                if item[0] == 'text':
                    result = item[1].result()
                    if result.strip():
                        text_results.append(result.strip())
                elif item[0] == 'math_batch':
                    future, indices = item[1]
                    batch_results = future.result()
                    # Store results in correct order
                    for idx, result in zip(indices, batch_results):
                        math_results[idx] = result
            
            # Remove any None values from math results (in case of errors)
            math_results = [r for r in math_results if r is not None]
            
            results['text'] = text_results
            results['equations'] = math_results

            print(f"✅ Processing completed in {time.time() - start_time:.1f}s")
            return results

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {'text': [], 'equations': []} 