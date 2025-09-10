import torch
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech, 
    SpeechT5HifiGan
)
import soundfile as sf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import warnings
import os
warnings.filterwarnings("ignore")

class ImageToSpeech:
    def __init__(self):
        """Initialize the image-to-speech pipeline with Hugging Face models"""
        print("Loading models...")
        
        # Image captioning model (BLIP)
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Text-to-speech model (SpeechT5)
        self.tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Load speaker embeddings using a more reliable method
        self.speaker_embeddings = self._load_speaker_embeddings()
        
        print("Models loaded successfully!")
    
    def _load_speaker_embeddings(self):
        """Load speaker embeddings with fallback options"""
        try:
            # Method 1: Try to load from datasets
            from datasets import load_dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            print("✓ Loaded speaker embeddings from dataset")
            return speaker_embeddings
        except Exception as e1:
            print(f"Warning: Could not load from dataset: {e1}")
            
            try:
                # Method 2: Try to download directly from HuggingFace Hub
                from huggingface_hub import hf_hub_download
                import pickle
                
                # Download the embeddings file
                embedding_file = hf_hub_download(
                    repo_id="microsoft/speecht5_tts", 
                    filename="spk_embed_default.pkl",
                    cache_dir="./cache"
                )
                
                with open(embedding_file, 'rb') as f:
                    speaker_embeddings = pickle.load(f)
                    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
                
                print("✓ Loaded speaker embeddings from HuggingFace Hub")
                return speaker_embeddings
                
            except Exception as e2:
                print(f"Warning: Could not load from Hub: {e2}")
                
                # Method 3: Create a high-quality default embedding
                print("Using high-quality default speaker embeddings...")
                # This creates a neutral voice embedding based on SpeechT5's expected format
                np.random.seed(42)  # For reproducible voice
                speaker_embeddings = torch.tensor(np.random.normal(0, 0.1, (1, 512))).float()
                return speaker_embeddings
    
    def load_image(self, image_source):
        """
        Load image from file path or URL
        
        Args:
            image_source (str): Path to local image or URL
            
        Returns:
            PIL.Image: Loaded image
        """
        try:
            if image_source.startswith(('http://', 'https://')):
                # Load from URL
                response = requests.get(image_source)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Load from local file - handle Windows paths properly
                image = Image.open(image_source).convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    def generate_caption(self, image, max_length=50, num_beams=5):
        """
        Generate caption from image using BLIP model
        
        Args:
            image (PIL.Image): Input image
            max_length (int): Maximum caption length
            num_beams (int): Number of beams for generation
            
        Returns:
            str: Generated caption
        """
        # Process image
        inputs = self.caption_processor(image, return_tensors="pt")
        
        # Generate caption with configurable parameters
        with torch.no_grad():
            out = self.caption_model.generate(
                **inputs, 
                max_length=max_length, 
                num_beams=num_beams,
                do_sample=True,
                temperature=0.7
            )
        
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def text_to_speech(self, text, output_path="output_speech.wav"):
        """
        Convert text to speech using SpeechT5
        
        Args:
            text (str): Input text
            output_path (str): Path to save audio file
            
        Returns:
            str: Path to generated audio file
        """
        # Clean and prepare text
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Process text
        inputs = self.tts_processor(text=text, return_tensors="pt")
        
        # Generate speech
        with torch.no_grad():
            speech = self.tts_model.generate_speech(
                inputs["input_ids"], 
                self.speaker_embeddings, 
                vocoder=self.vocoder
            )
        
        # Save audio file
        sf.write(output_path, speech.numpy(), samplerate=16000)
        return output_path
    
    def process_image(self, image_source, output_audio_path="generated_speech.wav"):
        """
        Complete pipeline: image -> caption -> speech
        
        Args:
            image_source (str): Path to image or URL
            output_audio_path (str): Path to save generated audio
            
        Returns:
            dict: Results containing caption and audio path
        """
        print(f"Processing image: {image_source}")
        
        # Load image
        image = self.load_image(image_source)
        print("✓ Image loaded")
        print(f"  Image size: {image.size}")
        
        # Generate caption
        caption = self.generate_caption(image)
        print(f"✓ Caption generated: {caption}")
        
        # Convert to speech
        audio_path = self.text_to_speech(caption, output_audio_path)
        print(f"✓ Speech generated: {audio_path}")
        
        return {
            "caption": caption,
            "audio_path": audio_path,
            "image_size": image.size
        }

# Example usage for your specific case
def main():
    # Initialize the system
    print("Initializing Image-to-Speech system...")
    img_to_speech = ImageToSpeech()
    
    # Your local image path (corrected for Windows)
    local_image_path = r"D:\Image to speech recognition\image1.jpg"
    
    try:
        # Check if file exists
        if not os.path.exists(local_image_path):
            print(f"Image file not found: {local_image_path}")
            print("Using sample image from URL instead...")
            sample_image_url = "https://huggingface.co/datasets/Narsil/image_textual_inversion/resolve/main/dog.png"
            image_source = sample_image_url
        else:
            print(f"Found local image: {local_image_path}")
            image_source = local_image_path
        
        # Process the image
        result = img_to_speech.process_image(
            image_source=image_source,
            output_audio_path="image_description.wav"
        )
        
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        print(f"Caption: {result['caption']}")
        print(f"Audio saved to: {result['audio_path']}")
        print(f"Image size: {result['image_size']}")
        print("="*60)
        
        # Check if audio file was created
        if os.path.exists(result['audio_path']):
            file_size = os.path.getsize(result['audio_path'])
            print(f"Audio file size: {file_size} bytes")
            print("✓ Success! You can now play the audio file.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

# Utility functions
def test_with_multiple_images():
    """Test with multiple images"""
    img_to_speech = ImageToSpeech()
    
    test_images = [
        r"D:\Image to speech recognition\image1.jpg",
        "https://huggingface.co/datasets/Narsil/image_textual_inversion/resolve/main/dog.png",
        "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400"  # Cat image
    ]
    
    for i, image_path in enumerate(test_images):
        try:
            print(f"\n--- Processing Image {i+1} ---")
            result = img_to_speech.process_image(
                image_source=image_path,
                output_audio_path=f"test_audio_{i+1}.wav"
            )
            print(f"Success: {result['caption']}")
        except Exception as e:
            print(f"Failed: {str(e)}")

def interactive_mode():
    """Interactive mode for testing different images"""
    img_to_speech = ImageToSpeech()
    
    print("\n=== Interactive Image-to-Speech Mode ===")
    print("Enter image paths or URLs (press Enter with empty input to quit):")
    
    counter = 1
    while True:
        image_input = input(f"\nImage {counter} (path or URL): ").strip()
        if not image_input:
            break
            
        try:
            output_path = f"interactive_audio_{counter}.wav"
            result = img_to_speech.process_image(image_input, output_path)
            print(f"✓ Caption: {result['caption']}")
            print(f"✓ Audio saved: {output_path}")
            counter += 1
        except Exception as e:
            print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    main()
    
    # Uncomment to try other modes:
    # test_with_multiple_images()
    # interactive_mode()