from PIL import Image
import math
import os

def resize_image_to_min_max_pixels(image, min_pixels=256*28*28, max_pixels=1536*28*28):
    """
    The default is optimized for Qwen2-VL
    
    Resize the image while maintaining aspect ratio such that the total number of pixels stays between
    the min_pixels and max_pixels thresholds.

    Args:
        image (PIL.Image.Image): The input image.
        min_pixels (int): The minimum allowed total pixel count (width * height).
        max_pixels (int): The maximum allowed total pixel count (width * height).

    Returns:
        PIL.Image.Image: The resized image.
    """
    # Get the original dimensions of the image
    width, height = image.size
    
    # Calculate the current total number of pixels
    current_pixels = width * height
    
    # Check if the image needs to be resized based on max_pixels
    if current_pixels > max_pixels:
        # If image is too large, scale down
        scaling_factor = math.sqrt(max_pixels / current_pixels)
    elif current_pixels < min_pixels:
        # If image is too small, scale up
        scaling_factor = math.sqrt(min_pixels / current_pixels)
    else:
        # Image is within the desired range, no need to resize
        return image
    
    # Compute the new dimensions based on the scaling factor
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    
    # Resize the image with the new dimensions while maintaining aspect ratio
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image

def save_resized_image(input_path, resized_image):
    """
    Save the resized image with '_resize' appended to the original filename.

    Args:
        input_path (str): The path of the original image.
        resized_image (PIL.Image.Image): The resized image.
    """
    # Split the file path into the directory, filename, and extension
    dir_name, file_name = os.path.split(input_path)
    file_base, file_ext = os.path.splitext(file_name)
    
    # Construct the new filename with '_resize' suffix
    new_file_name = f"{file_base}_resize{file_ext}"
    
    # Construct the full path for the resized image
    output_path = os.path.join(dir_name, new_file_name)
    
    # Save the resized image
    resized_image.save(output_path)
    print(f"Resized image saved as: {output_path}")

def main():
    # Paths to the images you want to test
    image_paths = [
        'C:/Users/willwe/Desktop/Cryptocarya_botelhensis_4603317652_label.jpg',
        'C:/Users/willwe/Desktop/Acaena_elongata_1320978699_label.jpg',
        'C:/Users/willwe/Desktop/spain_label.jpg',
        'C:/Users/willwe/Desktop/stewart.jpg',
        'C:/Users/willwe/Desktop/MICH_16205594_Poaceae_Jouvea_pilosa.jpg',
        'C:/Users/willwe/Desktop/MICH_16205594_Poaceae_Jouvea_pilosa_2.jpg',
    ]
    
    # Loop through the images, resize them, and save the output
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        
        # Open the image
        image = Image.open(image_path)
        
        # Resize the image based on the min/max pixel constraints
        resized_image = resize_image_to_min_max_pixels(image)
        
        # Save the resized image with '_resize' appended to the filename
        save_resized_image(image_path, resized_image)

if __name__ == "__main__":
    main()
