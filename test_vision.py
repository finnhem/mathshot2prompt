import ollama
from pathlib import Path

def analyze_image(image_path: str, prompt: str = "What is in this image?") -> str:
    """Analyze an image using Llama 3.2 Vision."""
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }]
    )
    return response['message']['content']

if __name__ == "__main__":
    # Replace 'your_image.jpg' with the path to your test image
    image_path = "your_image.jpg"
    
    if Path(image_path).exists():
        result = analyze_image(image_path)
        print("Analysis result:")
        print(result)
    else:
        print(f"Error: Image file '{image_path}' not found") 