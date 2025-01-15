"""Main processor for converting images to LaTeX equations."""

import cv2
import numpy as np
import time
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from .processor.segmentation import segment_regions, visualize_regions
from .models.texify import load_model, load_processor, GenerateVisionEncoderDecoderModel, batch_inference

class LatexProcessor:
    # Class variables for caching
    _cached_model = None
    _cached_processor = None
    
    def __init__(self):
        """Initialize the processor with model loading in background."""
        self.model = None
        self.processor = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.initialization_done = False
        self._init_start_time = time.time()
        
        # Start initialization in background
        self._init_components()
    
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
    
    def _init_components(self):
        """Initialize components in background."""
        print("Initializing Latex processor...")
        start_time = time.time()
        
        try:
            self.model, self.processor = self._load_or_get_cached_model()
            print(f"Initialization completed in {time.time() - start_time:.2f}s")
            self.initialization_done = True
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _ensure_model_loaded(self):
        """Ensure model is loaded before processing regions."""
        if self.model is None or self.processor is None:
            self.model, self.processor = self._load_or_get_cached_model()

    def process_image(self, image):
        """Process image and extract equations."""
        # Convert image to BGR if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # Get regions and normalized image
        regions, normalized_image = segment_regions(image)
        
        # Process regions in batches
        equations = []
        batch_size = 4  # Process 4 regions at a time for GPU efficiency
        
        print(f"🔢 Processing {len(regions)} regions...")
        for i in range(0, len(regions), batch_size):
            batch = regions[i:i + batch_size]
            print(f"⚡ Processing batch {i//batch_size + 1}/{(len(regions) + batch_size - 1)//batch_size}")
            
            # Extract region images
            region_images = []
            for x, y, w, h in batch:
                # Use normalized image for region extraction
                region = normalized_image[y:y+h, x:x+w]
                # Convert to PIL Image for model
                pil_region = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                region_images.append(pil_region)
            
            # Process batch
            self._ensure_model_loaded()
            future = self.executor.submit(batch_inference, region_images, self.model, self.processor)
            results = future.result()
            equations.extend([r for r in results if r is not None])
        
        print(f"✅ Processing completed in {time.time() - self._init_start_time:.1f}s")
        
        return {
            'equations': equations
        } 