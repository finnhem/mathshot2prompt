"""Module for correcting OCR output using local LLM."""

import ollama
from .config import SYSTEM_PROMPTS

class LLMCorrector:
    def __init__(self, model="phi4", prompt_key="math_text_corrector"):
        """Initialize LLM corrector with specified model and prompt."""
        self.model = model
        self.system_prompt = SYSTEM_PROMPTS[prompt_key]
    
    def _create_prompt(self, ocr_text):
        """Create a formatted prompt combining system prompt and OCR text."""
        return f"{self.system_prompt}\n\nInput text to correct:\n{ocr_text}"
    
    def correct_text(self, ocr_text):
        """Send OCR text to LLM for correction."""
        prompt = self._create_prompt(ocr_text)
        
        try:
            # Generate response using Ollama
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for more accurate corrections
                    "top_p": 0.9
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            print(f"❌ Error during LLM correction: {str(e)}")
            return ocr_text  # Return original text if correction fails 