import os, io, sys, inspect
from google.cloud import vision, storage
from PIL import Image, ImageDraw

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

def draw_boxes(image, bounds, color):
    if bounds:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        line_width = int((width + height) / 2 * 0.001)  # This sets the line width as 0.5% of the average dimension

        for bound in bounds:
            draw.polygon(
                [
                    bound["vertices"][0]["x"], bound["vertices"][0]["y"],
                    bound["vertices"][1]["x"], bound["vertices"][1]["y"],
                    bound["vertices"][2]["x"], bound["vertices"][2]["y"],
                    bound["vertices"][3]["x"], bound["vertices"][3]["y"],
                ],
                outline=color,
                width=line_width
            )
    return image

def detect_text(path):
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    # Extract bounding boxes
    bounds = []
    text_to_box_mapping = {}
    for text in texts[1:]:  # Skip the first entry, as it represents the entire detected text
        # Convert BoundingPoly to dictionary
        bound_dict = {
            "vertices": [
                {"x": vertex.x, "y": vertex.y} for vertex in text.bounding_poly.vertices
            ]
        }
        bounds.append(bound_dict)
        text_to_box_mapping[str(bound_dict)] = text.description

    if texts:
        # cleaned_text = texts[0].description.replace("\n", " ").replace("\t", " ").replace("|", " ")
        cleaned_text = texts[0].description
        return cleaned_text, bounds, text_to_box_mapping
    else:
        return '', None, None
    
def overlay_boxes_on_image(path, bounds,do_create_OCR_helper_image):
    if do_create_OCR_helper_image:
        image = Image.open(path)
        draw_boxes(image, bounds, "green")
        return image
    else:
        image = Image.open(path)
        return image





















# ''' Google Vision'''
# def detect_text(path):
#     """Detects text in the file located in the local filesystem."""
#     client = vision.ImageAnnotatorClient()

#     with io.open(path, 'rb') as image_file:
#         content = image_file.read()

#     image = vision.Image(content=content)

#     response = client.document_text_detection(image=image)
#     texts = response.text_annotations

#     if response.error.message:
#         raise Exception(
#             '{}\nFor more info on error messages, check: '
#             'https://cloud.google.com/apis/design/errors'.format(
#                 response.error.message))

#     return texts[0].description if texts else ''
