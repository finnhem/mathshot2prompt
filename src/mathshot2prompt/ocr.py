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
            regions = segment_regions(image)
            
            results = {
                'text': [],
                'equations': []
            }
            
            # Process regions in parallel
            futures = []
            
            # Process all regions with Texify
            if regions:
                print(f"🔢 Processing {len(regions)} regions...")
                
                self._ensure_model_loaded()
                
                # Process in batches
                batch_size = 4
                for i in range(0, len(regions), batch_size):
                    batch_regions = []
                    batch_indices = []
                    
                    for j, (x, y, w, h) in enumerate(regions[i:i + batch_size]):
                        region = pil_image.crop((x, y, x+w, y+h))
                        batch_regions.append(region)
                        batch_indices.append(i + j)
                    
                    if batch_regions:
                        future = self.executor.submit(
                            batch_inference, batch_regions, self.model, self.processor
                        )
                        futures.append(('math_batch', (future, batch_indices)))
                        
                        print(f"⚡ Processing batch {i//batch_size + 1}/{(len(regions) + batch_size - 1)//batch_size}")
                
            # Collect results as they complete
            math_results = [None] * len(regions)  # Pre-allocate list
            
            for item in futures:
                future, indices = item[1]
                batch_results = future.result()
                # Store results in correct order
                for idx, result in zip(indices, batch_results):
                    math_results[idx] = result
            
            # Remove any None values from results (in case of errors)
            math_results = [r for r in math_results if r is not None]
            results['equations'] = math_results

            print(f"✅ Processing completed in {time.time() - start_time:.1f}s")
            return results

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {'text': [], 'equations': []} 