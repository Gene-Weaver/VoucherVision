import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import time

# Global variables
H = 200
ROWS = 6
PADDING = 30

# Step 1: Fetch the Images from the URL Folder
def fetch_image_urls(url):
    response = requests.get(url + '?t=' + str(time.time()))
    soup = BeautifulSoup(response.content, 'html.parser')
    images = {}
    for node in soup.find_all('a'):
        href = node.get('href')
        if href.endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_index = int(href.split('__')[0])
                images[image_index] = url + '/' + href + '?t=' + str(time.time())
            except ValueError:
                print(f"Skipping invalid image: {href}")
    return images

# Step 2: Resize Images to Height H
def fetch_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def resize_images(images, target_height):
    resized_images = {}
    for index, img in images.items():
        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        resized_img = img.resize((new_width, target_height), Image.BICUBIC)
        resized_images[index] = resized_img
    return resized_images

# Step 3: Create a Collage with Efficient Placement Algorithm
def create_collage(image_urls, collage_path, H, ROWS, PADDING):
    images = {index: fetch_image(url) for index, url in image_urls.items()}
    resized_images = resize_images(images, H)  # Resize to H pixels height

    center_image = resized_images.pop(0)
    other_images = list(resized_images.items())

    # Calculate collage size based on the number of rows
    collage_width = 3000  # 16:9 aspect ratio width
    collage_height = (H + PADDING) * ROWS + 2 * PADDING  # Adjust height based on number of rows, add padding to top and bottom
    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

    # Sort images by width and height
    sorted_images = sorted(other_images, key=lambda x: x[1].width * x[1].height, reverse=True)

    # Create alternate placement list and insert the center image in the middle
    alternate_images = []
    i, j = 0, len(sorted_images) - 1
    halfway_point = (len(sorted_images) + 1) // 2
    count = 0

    while i <= j:
        if count == halfway_point:
            alternate_images.append((0, center_image))
        if i == j:
            alternate_images.append(sorted_images[i])
        else:
            alternate_images.append(sorted_images[i])
            alternate_images.append(sorted_images[j])
        i += 1
        j -= 1
        count += 2

    # Calculate number of images per row
    images_per_row = len(alternate_images) // ROWS
    extra_images = len(alternate_images) % ROWS

    # Place images in rows with only padding space between them
    def place_images_in_rows(images, collage, max_width, padding, row_height, rows, images_per_row, extra_images):
        y = padding
        for current_row in range(rows):
            row_images_count = images_per_row + (1 if extra_images > 0 else 0)
            extra_images -= 1 if extra_images > 0 else 0
            row_images = images[:row_images_count]
            row_width = sum(img.width for idx, img in row_images) + padding * (row_images_count - 1)
            x = (max_width - row_width) // 2
            for idx, img in row_images:
                collage.paste(img, (x, y))
                x += img.width + padding
            y += row_height + padding
            images = images[row_images_count:]

    place_images_in_rows(alternate_images, collage, collage_width, PADDING, H, ROWS, images_per_row, extra_images)

    collage.save(collage_path)

# Define the URL folder and other constants
url_folder = 'https://leafmachine.org/partners/'
collage_path = 'img/collage.jpg'

# Fetch, Create, and Update
image_urls = fetch_image_urls(url_folder)
create_collage(image_urls, collage_path, H, ROWS, PADDING)
