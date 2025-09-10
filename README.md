# Huggingface-image-to-speech-recognition

# Image to Speech Recognition

This project converts **images into spoken descriptions** using Hugging Face’s **BLIP** (for image captioning) and **SpeechT5** (for text-to-speech) models.  
It can load an image (local or from a URL), generate a **natural language caption**, and then speak the caption out loud in a **.wav audio file**.

---

## Features
-  **Image Captioning**: Generates captions using Salesforce’s BLIP model.  
-  **Text-to-Speech**: Converts text to natural speech with Microsoft’s SpeechT5.  
-  **Speaker Embeddings**: Uses CMU Arctic x-vectors for realistic voice output.  
-  **Batch Processing**: Can process multiple images at once.  
-  **Local or Online Images**: Works with both file paths and image URLs.  

---

##  Tech Stack
- [Python 3.10+](https://www.python.org/)  
- [PyTorch](https://pytorch.org/)  
- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers/)  
- [Datasets (Hugging Face)](https://huggingface.co/docs/datasets/)  
- [SoundFile](https://pysoundfile.readthedocs.io/) – to save audio  
- [Pillow](https://pillow.readthedocs.io/) – image handling  
- [Requests](https://requests.readthedocs.io/) – fetch images from the web  

