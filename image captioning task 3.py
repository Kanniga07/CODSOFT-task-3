import torch
import numpy as np
import tensorflow as tf
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

class ImageCaptioning:
    def __init__(self):
        # Load pre-trained ResNet50 model
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model.eval()  # Set the model to evaluation mode
        
        # Define the necessary image transforms (resize, normalize)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the size ResNet50 expects
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet50 normalization
        ])
        
        # Load VisionEncoderDecoderModel for image captioning (based on ViT and GPT-2)
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Move model to device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def extract_features(self, image_path):
        # Open and transform the image
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        # Extract features using ResNet50
        with torch.no_grad():
            features = self.resnet_model(image)
        
        return features

    def generate_caption(self, image_path):
        # Extract image features using ViT
        image = Image.open(image_path)
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)

        # Generate the caption
        output_ids = self.model.generate(inputs["pixel_values"], max_length=16, num_beams=4, early_stopping=True)
        
        # Decode the generated caption
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption

    def display_image_with_caption(self, image_path, caption):
        # Display the image and its generated caption
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.title(caption)
        plt.show()

# Example usage
image_captioning = ImageCaptioning()
image_path = r"C:\Users\kanni\OneDrive\Desktop\codsoft tasks\task 3 image.jpg"

features = image_captioning.extract_features(image_path)
caption = image_captioning.generate_caption(image_path)
print("Generated Caption:", caption)

image_captioning.display_image_with_caption(image_path, caption)
