from PIL import Image
import numpy as np
import cv2

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

def image_moments_color(image):
    # Split the image into color channels
    b, g, r = cv2.split(image)
    
    # Compute the moments for each color channel
    moments_b = cv2.moments(b)
    moments_g = cv2.moments(g)
    moments_r = cv2.moments(r)
    
    # Combine the moments of the color channels
    m00 = moments_b['m00'] + moments_g['m00'] + moments_r['m00']
    m10 = moments_b['m10'] + moments_g['m10'] + moments_r['m10']
    m01 = moments_b['m01'] + moments_g['m01'] + moments_r['m01']
    m20 = moments_b['m20'] + moments_g['m20'] + moments_r['m20']
    m02 = moments_b['m02'] + moments_g['m02'] + moments_r['m02']
    
    # Compute the first moments
    x_bar = m10 / m00
    y_bar = m01 / m00
    
    # Compute the second moments
    mu20 = m20 / m00 - x_bar**2
    mu02 = m02 / m00 - y_bar**2
    mu11 = (moments_b['m11'] + moments_g['m11'] + moments_r['m11']) / m00 - x_bar * y_bar
    
    return x_bar, y_bar, mu20, mu02, mu11

def min_max(image):
    # Compute the minimum and maximum pixel values for each color channel
    min_values = np.min(image, axis=(0, 1))
    max_values = np.max(image, axis=(0, 1))

    print("Minimum pixel values (BGR):", min_values)
    print("Maximum pixel values (BGR):", max_values)

    # Compute the range for each color channel
    pixel_range_per_channel = max_values - min_values
    print("Pixel value range per channel (BGR):", pixel_range_per_channel)

    # Compute the overall range
    overall_pixel_range = np.max(pixel_range_per_channel)
    print("Overall pixel value range:", overall_pixel_range)

# Example usage
image_path = "images_charles.jpeg"
normalized_image = normalize_image(image_path)
#normalized_image.show()  # Show the normalized image
save_path = "normalized_image_1.jpg"
normalized_image.save(save_path)

normalized_image = normalize_image("normalized_image_1.jpg")
#normalized_image.show()  # Show the normalized image
save_path = "normalized_image.jpg"
normalized_image.save(save_path)

# Read the color image
image = cv2.imread('images_charles.jpeg')

# Get the moments of the color image
x_bar, y_bar, mu20, mu02, mu11 = image_moments_color(image)

print("==================================")
print("First Image")
print("First moments:")
print("x_bar:", x_bar)
print("y_bar:", y_bar)

print("\nSecond moments:")
print("mu20:", mu20)
print("mu02:", mu02)
print("mu11:", mu11)

print("")
print("Min Max")
min_max(image)

print("")
print("==================================")
print("Normalized Image")
# Read the color image
image = cv2.imread('normalized_image.jpg')

# Get the moments of the color image
x_bar, y_bar, mu20, mu02, mu11 = image_moments_color(image)

print("First Image")
print("First moments:")
print("x_bar:", x_bar)
print("y_bar:", y_bar)

print("\nSecond moments:")
print("mu20:", mu20)
print("mu02:", mu02)
print("mu11:", mu11)

print("")
print("Min Max")
min_max(image)
min_max(image)

