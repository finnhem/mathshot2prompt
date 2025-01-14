"""Main entry point for the image-to-LaTeX converter."""

import time
from PIL import Image, ImageGrab
from . import ocr

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
        image_data = get_clipboard_image()
        
        # Process image with LaTeX processor
        ocr_start = time.time()
        print_debug("Processing image segments...", "info")
        processor = ocr.LatexProcessor()
        result = processor.process_image(image_data)
        print_debug("Image processing completed", "success", time.time() - ocr_start)
        
        # Print results
        print("\nExtracted Equations:")
        print("=" * 40)
        for equation in result['equations']:
            print(equation)
            
    except ValueError as e:
        print_debug(str(e), "error")
    except Exception as e:
        print_debug(f"An unexpected error occurred: {str(e)}", "error")
    finally:
        total_time = time.time() - total_start_time
        print_debug("Process completed", "success", total_time) 