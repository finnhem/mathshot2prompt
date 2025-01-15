"""Main entry point for the image-to-LaTeX converter."""

import time
import numpy as np
from PIL import Image, ImageGrab
import cv2
from . import ocr
from .llm_corrector import LLMCorrector

print("🚀 Starting up...")

def print_debug(message: str, level: str = "info", timing: float = None) -> None:
    """Print debug information with formatting."""
    colors = {
        "info": "\033[94m",  # Blue
        "success": "\033[92m",  # Green
        "warning": "\033[93m",  # Yellow
        "error": "\033[91m",  # Red
        "reset": "\033[0m"
    }
    prefix = {
        "info": "ℹ️ ",
        "success": "✅ ",
        "warning": "⚠️ ",
        "error": "❌ "
    }
    
    if timing is not None:
        if timing >= 60:
            minutes = int(timing // 60)
            seconds = timing % 60
            time_info = f" ({minutes}min {seconds:.2f}s)"
        else:
            time_info = f" ({timing:.2f}s)"
    else:
        time_info = ""
        
    print(f"{colors[level]}{prefix[level]}{message}{time_info}{colors['reset']}")

def get_clipboard_image() -> Image.Image:
    """Get image from clipboard or load from file path."""
    start_time = time.time()
    print_debug("Checking clipboard for image...", "info")
    clipboard_content = ImageGrab.grabclipboard()
    
    if clipboard_content is None:
        raise ValueError("No image found in clipboard")
    
    # Handle case where multiple items are in clipboard
    if isinstance(clipboard_content, list):
        print_debug("File path(s) found in clipboard, using first file", "warning")
        if not clipboard_content:  # Empty list
            raise ValueError("No image found in clipboard")
        # Load image from file path
        try:
            img = Image.open(clipboard_content[0])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            print_debug("Image loaded from file", "success", time.time() - start_time)
            return img
        except Exception as e:
            raise ValueError(f"Could not load image from path: {str(e)}")
    else:
        # Direct image data from clipboard
        if clipboard_content.mode != 'RGB':
            clipboard_content = clipboard_content.convert('RGB')
        print_debug("Image loaded from clipboard", "success", time.time() - start_time)
        return clipboard_content

if __name__ == "__main__":
    total_start_time = time.time()
    print_debug("Starting LaTeX analysis...", "info")
    
    try:
        # Get image from clipboard
        clipboard_start = time.time()
        pil_image = get_clipboard_image()
        
        # Convert PIL Image to numpy array
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Process image with LaTeX processor
        ocr_start = time.time()
        print_debug("Processing image segments...", "info")
        processor = ocr.LatexProcessor()
        result = processor.process_image(image)
        print_debug("Image processing completed", "success", time.time() - ocr_start)
        
        # Combine equations into a single text
        combined_text = "\n\n".join(result['equations'])
        
        print("\nExtracted Equations:")
        print("=" * 40)
        print(combined_text)
        
        # Initialize LLM corrector and process text
        llm_start = time.time()
        print_debug("Starting LLM correction...", "info")
        corrector = LLMCorrector(model="phi4")
        corrected_text = corrector.correct_text(combined_text)
        print_debug("LLM correction completed", "success", time.time() - llm_start)
        
        # Print results
        print("\nCorrected Output:")
        print("=" * 40)
        print(corrected_text)
            
    except ValueError as e:
        print_debug(str(e), "error")
    except Exception as e:
        print_debug(f"An unexpected error occurred: {str(e)}", "error")
    finally:
        total_time = time.time() - total_start_time
        print_debug("Process completed", "success", total_time) 