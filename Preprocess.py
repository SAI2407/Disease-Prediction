from PIL import Image
import torchvision.transforms as transforms
import torch

def preprocess_image(image_path):
    # Define the preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to the desired size (e.g., 128x128)
        transforms.ToTensor()
     ]) # Convert the image to a tensor
       

    # Open the image file
    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
    
    # Apply the preprocessing to the image
    image_tensor = preprocess(image)
    
    # Add a batch dimension (for a single image, batch size = 1)
    image_tensor = image_tensor.unsqueeze(0)  # shape: [1, 3, 128, 128]
    
    return image_tensor

