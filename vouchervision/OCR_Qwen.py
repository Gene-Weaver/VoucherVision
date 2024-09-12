import random, os, base64, io, json, torch
from PIL import Image
import copy
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
from PIL import Image, ImageDraw, ImageFont 
import numpy as np
import warnings
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from vouchervision.utils_LLM import SystemLoadMonitor
except:
    from utils_LLM import SystemLoadMonitor


warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

### additionalText instead of occurrenceRemarks
PROMPT_OCR_ONLY_1 = """I cannot read the text in this image. Without explanation, please read me all of the text in this image."""
PROMPT_OCR_ONLY_2 = """Perform OCR on this image. Return only the verbatim text without any explanation."""
PROMPT_OCR_ONLY_3 = """Perform OCR on this image and return all of the text contained within the image. Return only the verbatim OCR text without any explanation."""

PROMPT_PARSE = """Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
The rules are:
1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below. 2. Map the unstructured OCR text to the appropriate JSON key and populate the field given the user-defined rules. 3. JSON key values are permitted to remain empty strings if the corresponding information is not found in the unstructured OCR text. 4. Duplicate dictionary fields are not allowed. 5. Ensure all JSON keys are in camel case. 6. Ensure new JSON field values follow sentence case capitalization. 7. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template. 8. Ensure output JSON string is valid JSON format. It should not have trailing commas or unquoted keys. 9. Only return a JSON dictionary represented as a string. You should not explain your answer.
This section provides rules for formatting each JSON value organized by the JSON key.
This is the JSON template that includes instructions for each key:
catalogNumber: Barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits. scientificName: The scientific name of the taxon including genus, specific epithet, and any lower classifications. genus: Taxonomic determination to genus. Genus must be capitalized. If genus is not present use the taxonomic family name followed by the word "indet". specificEpithet: The name of the species epithet of the scientificName. Only include the species epithet. speciesNameAuthorship: The authorship information for the scientificName formatted according to the conventions of the applicable Darwin Core nomenclatural code. collectedBy: A comma separated list of names of people, groups, or organizations responsible for observing, recording, collecting, or presenting the original specimen. The primary collector or observer should be listed first.  collectorNumber: An identifier given to the occurrence at the time it was recorded, the specimen collectors number. identifiedBy: A comma separated list of names of people, groups, or organizations who assigned the taxon to the subject organism. This is not the specimen collector.  verbatimCollectionDate: The verbatim original representation of the date and time information for when the specimen was collected. Date of collection exactly as it appears on the label. Do not change the format or correct typos. collectionDate: Date the specimen was collected formatted as year-month-day, YYYY-MM-DD. If specific components of the date are unknown, they should be replaced with zeros. Use 0000-00-00 if the entire date is unknown, YYYY-00-00 if only the year is known, and YYYY-MM-00 if year and month are known but day is not. collectionDateEnd: If a range of collection dates is provided, this is the later end date while collectionDate is the beginning date. Use the same formatting as for collectionDate. habitat: Description of the ecological habitat in which the specimen collection event occurred. cultivated: Cultivated plants are intentionally grown by humans. In text descriptions, look for planting dates, garden, cult, cultivated, ornamental, cultivar names, garden, or farm to indicate cultivated plant. Use yes if cultivated, otherwise leave blank. country: The name of the nation or country in which the specimen was originally collected. stateProvince: The name of the sub-national administrative region than country (state, province, canton, department, region, etc.) in which the specimen was originally collected. county: The full, unabbreviated name of the next smaller administrative region than stateProvince (county, shire, department, parish etc.) in which the specimen was originally collected. locality: Description of geographic location, landscape, landmarks, regional features, nearby places, municipality, city, or any contextual information aiding in pinpointing the exact origin or location of the specimen. verbatimCoordinates: Verbatim location coordinates as they appear on the label. Do not convert formats. Possible coordinate types include [Lat, Long, UTM, TRS]. decimalLatitude: Latitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format. decimalLongitude: Longitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format. minimumElevationInMeters: Minimum elevation or altitude in meters. Only if units are explicit then convert from feet ("ft" or "ft."" or "feet") to meters ("m" or "m." or "meters"). Round to integer. Values greater than 6000 are in feet and need to be converted. maximumElevationInMeters: Maximum elevation or altitude in meters. If only one elevation is present, then max_elevation should be set to the null_value. Only if units are explicit then convert from feet ("ft" or "ft." or "feet") to meters ("m" or "m." or "meters"). Round to integer. Values greater than 6000 are in feet and need to be converted. elevationUnits: Use m if the final elevation is reported in meters. If you convert from feet to meters, then use m. additionalText: All remaining OCR text and text that is not part of the main label, secondary text, background and supporting information. 
The unstructured OCR text is the verbatim text contained in the image.
Please populate the following JSON dictionary based on the rules and the unformatted OCR text:
{
"catalogNumber": "",
"scientificName": "",
"genus": "",
"specificEpithet": "",
"speciesNameAuthorship": "",
"collectedBy": "",
"collectorNumber": "",
"collector": "",
"identifiedBy": "",
"verbatimCollectionDate": "",
"collectionDate": "",
"collectionDateEnd": "",
"habitat": "",
"country": "",
"stateProvince": "",
"county": "",
"locality": "",
"verbatimCoordinates": "",
"decimalLatitude": "",
"decimalLongitude": "",
"minimumElevationInMeters": "",
"maximumElevationInMeters": "",
"elevationUnits": "",
"additionalText": "",
}
"""

PROMPT_OCR_AND_PARSE = """Perform OCR on this image. Return only the verbatim text without any explanation. Then complete the following task:
Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
The rules are:
1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below. 2. Map the unstructured OCR text to the appropriate JSON key and populate the field given the user-defined rules. 3. JSON key values are permitted to remain empty strings if the corresponding information is not found in the unstructured OCR text. 4. Duplicate dictionary fields are not allowed. 5. Ensure all JSON keys are in camel case. 6. Ensure new JSON field values follow sentence case capitalization. 7. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template. 8. Ensure output JSON string is valid JSON format. It should not have trailing commas or unquoted keys. 9. Only return a JSON dictionary represented as a string. You should not explain your answer.
This section provides rules for formatting each JSON value organized by the JSON key.
This is the JSON template that includes instructions for each key:
catalogNumber: Barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits. scientificName: The scientific name of the taxon including genus, specific epithet, and any lower classifications. genus: Taxonomic determination to genus. Genus must be capitalized. If genus is not present use the taxonomic family name followed by the word "indet". specificEpithet: The name of the species epithet of the scientificName. Only include the species epithet. speciesNameAuthorship: The authorship information for the scientificName formatted according to the conventions of the applicable Darwin Core nomenclatural code. collectedBy: A comma separated list of names of people, groups, or organizations responsible for observing, recording, collecting, or presenting the original specimen. The primary collector or observer should be listed first.  collectorNumber: An identifier given to the occurrence at the time it was recorded, the specimen collectors number. identifiedBy: A comma separated list of names of people, groups, or organizations who assigned the taxon to the subject organism. This is not the specimen collector.  verbatimCollectionDate: The verbatim original representation of the date and time information for when the specimen was collected. Date of collection exactly as it appears on the label. Do not change the format or correct typos. collectionDate: Date the specimen was collected formatted as year-month-day, YYYY-MM-DD. If specific components of the date are unknown, they should be replaced with zeros. Use 0000-00-00 if the entire date is unknown, YYYY-00-00 if only the year is known, and YYYY-MM-00 if year and month are known but day is not. collectionDateEnd: If a range of collection dates is provided, this is the later end date while collectionDate is the beginning date. Use the same formatting as for collectionDate. habitat: Description of the ecological habitat in which the specimen collection event occurred. cultivated: Cultivated plants are intentionally grown by humans. In text descriptions, look for planting dates, garden, cult, cultivated, ornamental, cultivar names, garden, or farm to indicate cultivated plant. Use yes if cultivated, otherwise leave blank. country: The name of the nation or country in which the specimen was originally collected. stateProvince: The name of the sub-national administrative region than country (state, province, canton, department, region, etc.) in which the specimen was originally collected. county: The full, unabbreviated name of the next smaller administrative region than stateProvince (county, shire, department, parish etc.) in which the specimen was originally collected. locality: Description of geographic location, landscape, landmarks, regional features, nearby places, municipality, city, or any contextual information aiding in pinpointing the exact origin or location of the specimen. verbatimCoordinates: Verbatim location coordinates as they appear on the label. Do not convert formats. Possible coordinate types include [Lat, Long, UTM, TRS]. decimalLatitude: Latitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format. decimalLongitude: Longitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format. minimumElevationInMeters: Minimum elevation or altitude in meters. Only if units are explicit then convert from feet ("ft" or "ft."" or "feet") to meters ("m" or "m." or "meters"). Round to integer. Values greater than 6000 are in feet and need to be converted. maximumElevationInMeters: Maximum elevation or altitude in meters. If only one elevation is present, then max_elevation should be set to the null_value. Only if units are explicit then convert from feet ("ft" or "ft." or "feet") to meters ("m" or "m." or "meters"). Round to integer. Values greater than 6000 are in feet and need to be converted. elevationUnits: Use m if the final elevation is reported in meters. If you convert from feet to meters, then use m. additionalText: All remaining OCR text and text that is not part of the main label, secondary text, background and supporting information. 
The unstructured OCR text is the verbatim text contained in the image.
Please populate the following JSON dictionary based on the rules and the unformatted OCR text:
{
"catalogNumber": "",
"scientificName": "",
"genus": "",
"specificEpithet": "",
"speciesNameAuthorship": "",
"collectedBy": "",
"collectorNumber": "",
"collector": "",
"identifiedBy": "",
"verbatimCollectionDate": "",
"collectionDate": "",
"collectionDateEnd": "",
"habitat": "",
"country": "",
"stateProvince": "",
"county": "",
"locality": "",
"verbatimCoordinates": "",
"decimalLatitude": "",
"decimalLongitude": "",
"minimumElevationInMeters": "",
"maximumElevationInMeters": "",
"elevationUnits": "",
"additionalText": "",
}
"""


class Qwen2VLOCR:
    def __init__(self, logger, model_id='Qwen/Qwen2-VL-7B-Instruct', ocr_text=None):
        self.MAX_TOKENS = 1024
        self.MAX_PX = 1536
        self.MIN_PX = 256

        self.logger = logger
        self.model_id = model_id

        self.monitor = SystemLoadMonitor(logger)

        try:
            from transformers import Qwen2VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            print("Qwen environment detected, using Qwen model.")
        except ImportError:
            assert False, (
                'Install the Qwen required packages before trying to use the Qwen models. '
                'Either run pip install -U "git+https://github.com/huggingface/transformers" '
                'and pip install "qwen-vl-utils" or click the Install Qwen button in the GUI.'
            )


        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
        )

        # Resize image
        min_pixels = self.MIN_PX * 28 * 28
        max_pixels = self.MAX_PX * 28 * 28
        self.processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=min_pixels, max_pixels=max_pixels)

    def encode_image_base64(self, image):
        """Encode the image to base64 format."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    def ocr_with_vlm(self, image_input, workflow_option=1):
        self.monitor.start_monitoring_usage()

        # Check if the input is a string path, a PIL Image, or an OpenCV image
        if isinstance(image_input, str):
            # Load the image if a file path is provided
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            # The input is already a PIL Image
            image = image_input
        elif isinstance(image_input, np.ndarray):
            # The input is an OpenCV (cv2) image, convert to PIL
            image = self.convert_cv2_to_pil(image_input)
        else:
            raise ValueError("Input type not supported. Please provide a string path, PIL image, or OpenCV image.")

        # Convert the image to base64 and embed it in the message
        encoded_image = self.encode_image_base64(image)
        base64_image_str = f"data:image/jpeg;base64,{encoded_image}"
        
        # Workflow selection
        if workflow_option == 1:
            # Option 1: Use TEXT_OCR and image to return OCR text
            text_prompt = PROMPT_OCR_ONLY_3
        elif workflow_option == 2:
            # Option 2: Use TEXT_OCR and image to generate OCR text, then pass the text with TEXT2 to generate JSON
            text_prompt = PROMPT_OCR_ONLY_3  # First generate the text
        elif workflow_option == 3:
            # Option 3: Use TEXT with the image to directly return structured JSON
            text_prompt = PROMPT_OCR_AND_PARSE
        else:
            raise ValueError("Invalid workflow_option. Please provide 1, 2, or 3.")

        # Prepare image and prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": base64_image_str},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        # Create inputs for model
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # Generate OCR text or structured JSON (for Option 3)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.MAX_TOKENS
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        final_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # If Option 2: Pass the generated OCR text (final_text) as input to TEXT2 for JSON generation
        if workflow_option == 2:
            text_prompt = PROMPT_PARSE
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_text},  # Using the generated OCR text
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Create inputs without image for TEXT2
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")

            # Generate JSON response based on OCR text
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_TOKENS
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            final_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

        self.monitor.stop_inference_timer()
        usage_report = self.monitor.stop_monitoring_report_usage()

        return final_text, usage_report

def main():
    
    # img_path = 'D:/D_Desktop/BR_1839468565_Ochnaceae_Campylospermum_reticulatum_label.jpg'
    # img_path = 'C:/Users/willwe/Desktop/MICH_16205594_Poaceae_Jouvea_pilosa.jpg'
    # img_path = 'C:/Users/willwe/Desktop/stewart.jpg'
    img_path = 'C:/Users/willwe/Desktop/Cryptocarya_botelhensis_4603317652_label.jpg'
    
    image = Image.open(img_path)

    # Create the HFVLMOCR object with a model ID
    # ocr = Qwen2VLOCR(logger=None, model_id='Qwen/Qwen2-VL-7B-Instruct')
    ### ocr = Qwen2VLOCR(logger=None, model_id='Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8')
    # ocr = Qwen2VLOCR(logger=None, model_id='Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4')
    ocr = Qwen2VLOCR(logger=None, model_id='Qwen/Qwen2-VL-7B-Instruct-AWQ')

    # ocr = Qwen2VLOCR(logger=None, model_id='Qwen/Qwen2-VL-2B-Instruct')
    # ocr = Qwen2VLOCR(logger=None, model_id='Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8')
    # ocr = Qwen2VLOCR(logger=None, model_id='Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4')
    # ocr = Qwen2VLOCR(logger=None, model_id='Qwen/Qwen2-VL-2B-Instruct-AWQ')
    
    final_text, usage_report = ocr.ocr_with_vlm(image, workflow_option=1)

    print("Final OCR Text:\n", final_text)
    # Check if final_text is valid JSON
    try:
        final_json = json.loads(final_text)
        print("Valid JSON:", True)
        print("Parsed JSON:\n", json.dumps(final_json, indent=4))  # Pretty-print the JSON
    except json.JSONDecodeError as e:
        print("Valid JSON:", False)
        print(e)
    print(usage_report,"\n\n")



    final_text, usage_report = ocr.ocr_with_vlm(image, workflow_option=2)

    print("Final OCR Text:\n", final_text)
    # Check if final_text is valid JSON
    try:
        final_json = json.loads(final_text)
        print("Valid JSON:", True)
        print("Parsed JSON:\n", json.dumps(final_json, indent=4))  # Pretty-print the JSON
    except json.JSONDecodeError as e:
        print("Valid JSON:", False)
        print(e)
    print(usage_report,"\n\n")



    final_text, usage_report = ocr.ocr_with_vlm(image, workflow_option=3)

    print("Final OCR Text:\n", final_text)
    # Check if final_text is valid JSON
    try:
        final_json = json.loads(final_text)
        print("Valid JSON:", True)
        print("Parsed JSON:\n", json.dumps(final_json, indent=4))  # Pretty-print the JSON
    except json.JSONDecodeError as e:
        print("Valid JSON:", False)
        print(e)
    print(usage_report,"\n\n")

if __name__ == '__main__':
    main()







