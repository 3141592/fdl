from PIL import Image
import numpy as np

def normalize_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Normalize the image array
    normalized_img_array = img_array / 255.0  # Assuming original range is [0, 255]
    
    # Convert the normalized numpy array back to an image
    normalized_img = Image.fromarray((normalized_img_array * 255).astype(np.uint8))
    
    return normalized_img

# Example usage
image_path = "images_charles.jpeg"
normalized_image = normalize_image(image_path)
normalized_image.show()  # Show the normalized image
save_path = "normalized_image.jpg"
normalized_image.save(save_path)
