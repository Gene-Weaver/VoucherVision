import random, os, base64, io, json, torch
from PIL import Image
import copy
import numpy as np
import warnings
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

# Avoid deprecation warnings for TypedStorage
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

### PROMPTS ###
### additionalText instead of occurrenceRemarks
PROMPT_OCR_ONLY_1 = """I cannot read the text in this image. Without explanation, please read me all of the text in this image."""
PROMPT_OCR_ONLY_2 = """Perform OCR on this image. Return only the verbatim text without any explanation."""
PROMPT_OCR_ONLY_3 = """Perform OCR on this image and return all of the text contained within the image. Return only the verbatim OCR text without any explanation."""
PROMPT_OCR_ONLY_TEST = """Extract all text is in this image."""

PROMPT_PARSE_A = """Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
The rules are:
1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below. 2. Map the unstructured OCR text to the appropriate JSON key and populate the field given the user-defined rules. 3. JSON key values are permitted to remain empty strings if the corresponding information is not found in the unstructured OCR text. 4. Duplicate dictionary fields are not allowed. 5. Ensure all JSON keys are in camel case. 6. Ensure new JSON field values follow sentence case capitalization. 7. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template. 8. Ensure output JSON string is valid JSON format. It should not have trailing commas or unquoted keys. 9. Only return a JSON dictionary represented as a string. You should not explain your answer.
This section provides rules for formatting each JSON value organized by the JSON key.
This is the JSON template that includes instructions for each key:
catalogNumber: Barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits. scientificName: The scientific name of the taxon including genus, specific epithet, and any lower classifications. genus: Taxonomic determination to genus. Genus must be capitalized. If genus is not present use the taxonomic family name followed by the word "indet". specificEpithet: The name of the species epithet of the scientificName. Only include the species epithet. speciesNameAuthorship: The authorship information for the scientificName formatted according to the conventions of the applicable Darwin Core nomenclatural code. collectedBy: A comma separated list of names of people, groups, or organizations responsible for observing, recording, collecting, or presenting the original specimen. The primary collector or observer should be listed first.  collectorNumber: An identifier given to the occurrence at the time it was recorded, the specimen collectors number. identifiedBy: A comma separated list of names of people, groups, or organizations who assigned the taxon to the subject organism. This is not the specimen collector.  verbatimCollectionDate: The verbatim original representation of the date and time information for when the specimen was collected. Date of collection exactly as it appears on the label. Do not change the format or correct typos. collectionDate: Date the specimen was collected formatted as year-month-day, YYYY-MM-DD. If specific components of the date are unknown, they should be replaced with zeros. Use 0000-00-00 if the entire date is unknown, YYYY-00-00 if only the year is known, and YYYY-MM-00 if year and month are known but day is not. collectionDateEnd: If a range of collection dates is provided, this is the later end date while collectionDate is the beginning date. Use the same formatting as for collectionDate. habitat: Description of the ecological habitat in which the specimen collection event occurred. cultivated: Cultivated plants are intentionally grown by humans. In text descriptions, look for planting dates, garden, cult, cultivated, ornamental, cultivar names, garden, or farm to indicate cultivated plant. Use yes if cultivated, otherwise leave blank. country: The name of the nation or country in which the specimen was originally collected. stateProvince: The name of the sub-national administrative region than country (state, province, canton, department, region, etc.) in which the specimen was originally collected. county: The full, unabbreviated name of the next smaller administrative region than stateProvince (county, shire, department, parish etc.) in which the specimen was originally collected. locality: Description of geographic location, landscape, landmarks, regional features, nearby places, municipality, city, or any contextual information aiding in pinpointing the exact origin or location of the specimen. verbatimCoordinates: Verbatim location coordinates as they appear on the label. Do not convert formats. Possible coordinate types include [Lat, Long, UTM, TRS]. decimalLatitude: Latitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format. decimalLongitude: Longitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format. minimumElevationInMeters: Minimum elevation or altitude in meters. Only if units are explicit then convert from feet ("ft" or "ft."" or "feet") to meters ("m" or "m." or "meters"). Round to integer. Values greater than 6000 are in feet and need to be converted. maximumElevationInMeters: Maximum elevation or altitude in meters. If only one elevation is present, then max_elevation should be set to the null_value. Only if units are explicit then convert from feet ("ft" or "ft." or "feet") to meters ("m" or "m." or "meters"). Round to integer. Values greater than 6000 are in feet and need to be converted. elevationUnits: Use m if the final elevation is reported in meters. Use ft if the final elevation is in feet. Units should match minimumElevationInMeters and maximumElevationInMeters. additionalText: All remaining OCR text and text that is not part of the main label, secondary text, background and supporting information. 
This is the unstructured OCR text:
"""
PROMPT_PARSE_B = """
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
catalogNumber: Barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits. scientificName: The scientific name of the taxon including genus, specific epithet, and any lower classifications. genus: Taxonomic determination to genus. Genus must be capitalized. If genus is not present use the taxonomic family name followed by the word "indet". specificEpithet: The name of the species epithet of the scientificName. Only include the species epithet. speciesNameAuthorship: The authorship information for the scientificName formatted according to the conventions of the applicable Darwin Core nomenclatural code. collectedBy: A comma separated list of names of people, groups, or organizations responsible for observing, recording, collecting, or presenting the original specimen. The primary collector or observer should be listed first.  collectorNumber: An identifier given to the occurrence at the time it was recorded, the specimen collectors number. identifiedBy: A comma separated list of names of people, groups, or organizations who assigned the taxon to the subject organism. This is not the specimen collector.  verbatimCollectionDate: The verbatim original representation of the date and time information for when the specimen was collected. Date of collection exactly as it appears on the label. Do not change the format or correct typos. collectionDate: Date the specimen was collected formatted as year-month-day, YYYY-MM-DD. If specific components of the date are unknown, they should be replaced with zeros. Use 0000-00-00 if the entire date is unknown, YYYY-00-00 if only the year is known, and YYYY-MM-00 if year and month are known but day is not. collectionDateEnd: If a range of collection dates is provided, this is the later end date while collectionDate is the beginning date. Use the same formatting as for collectionDate. habitat: Description of the ecological habitat in which the specimen collection event occurred. cultivated: Cultivated plants are intentionally grown by humans. In text descriptions, look for planting dates, garden, cult, cultivated, ornamental, cultivar names, garden, or farm to indicate cultivated plant. Use yes if cultivated, otherwise leave blank. country: The name of the nation or country in which the specimen was originally collected. stateProvince: The name of the sub-national administrative region than country (state, province, canton, department, region, etc.) in which the specimen was originally collected. county: The full, unabbreviated name of the next smaller administrative region than stateProvince (county, shire, department, parish etc.) in which the specimen was originally collected. locality: Description of geographic location, landscape, landmarks, regional features, nearby places, municipality, city, or any contextual information aiding in pinpointing the exact origin or location of the specimen. verbatimCoordinates: Verbatim location coordinates as they appear on the label. Do not convert formats. Possible coordinate types include [Lat, Long, UTM, TRS]. decimalLatitude: Latitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format. decimalLongitude: Longitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format. minimumElevationInMeters: Minimum elevation or altitude in meters. Only if units are explicit then convert from feet ("ft" or "ft."" or "feet") to meters ("m" or "m." or "meters"). Round to integer. Values greater than 6000 are in feet and need to be converted. maximumElevationInMeters: Maximum elevation or altitude in meters. If only one elevation is present, then max_elevation should be set to the null_value. Only if units are explicit then convert from feet ("ft" or "ft." or "feet") to meters ("m" or "m." or "meters"). Round to integer. Values greater than 6000 are in feet and need to be converted. elevationUnits: Use m if the final elevation is reported in meters. Use ft if the final elevation is in feet. Units should match minimumElevationInMeters and maximumElevationInMeters. additionalText: All remaining OCR text and text that is not part of the main label, secondary text, background and supporting information. 
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

# Phi-3.5 model class
class Phi35VisionOCR:
    def __init__(self, logger=None, model_id='microsoft/Phi-3.5-vision-instruct', attn_implementation='eager'):
        self.MAX_TOKENS = 1024
        self.MAX_PX = 1536  # Max size for images
        self.MIN_PX = 256  # Min size for images
        
        self.logger = logger
        self.model_id = model_id

        # Load the model and processor
        # If on Linux AND using Ada generation GPU (NVIDIA A100, NVIDIA A6000 Ada, NVIDIA H100), can use 'flash_attention_2' to improve efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="cuda", 
            trust_remote_code=True, _attn_implementation=attn_implementation
        )
        # Resize image based on constraints
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, num_crops=4
        )

    def encode_image_base64(self, image):
        """Encode the image to base64 format."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def resize_image(self, image):
        """Resize image to fit within the min/max resolution range."""
        width, height = image.size  # Extract width and height directly
        # Calculate scaling factor
        scale_factor = min(self.MAX_PX / max(width, height), self.MIN_PX / min(width, height))
        # Compute the new size
        new_size = (int(width * scale_factor), int(height * scale_factor))
        # Resize and return the image
        return image.resize(new_size, Image.LANCZOS)


    def ocr_with_vlm(self, image_input, workflow_option=1):
        ocr_text = None
        json_out = None
        # Load the image if it's a path, PIL image, or numpy array
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            raise ValueError("Unsupported input type, provide a path, PIL image, or numpy array.")

        # Resize the image if necessary
        image = self.resize_image(image)

        # Use the image placeholder for the Phi-3.5 model format
        image_str = "<|image_1|>\n"

        # Workflow selection
        if workflow_option == 1:
            prompt = PROMPT_OCR_ONLY_TEST  # Perform OCR only
        elif workflow_option == 2:
            prompt = PROMPT_OCR_ONLY_TEST  # Perform OCR, then chain to parsing afterward
        elif workflow_option == 3:
            prompt = PROMPT_OCR_AND_PARSE  # Perform both OCR and JSON parsing directly
        else:
            raise ValueError("Invalid workflow_option. Please provide 1, 2, or 3.")

        # Create message by appending the prompt after the image placeholder
        input_content = image_str + prompt
        messages = [{"role": "user", "content": input_content}]

        # Apply the chat template to format the prompt
        text_prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Pass the text and image inputs to the processor
        inputs = self.processor(
            text=text_prompt,  # Formatted text prompt
            images=[image],      # Pass the image(s) as a list
            return_tensors="pt"  # Convert to PyTorch tensors
        ).to("cuda")

        # Generate a response from the model
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.MAX_TOKENS, eos_token_id=self.processor.tokenizer.eos_token_id
        )

        # Remove the input tokens from the generated response
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]

        # Decode the model's output
        ocr_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if workflow_option == 3:
            json_out = ocr_text
            ocr_text = None

        # If the second workflow is selected, chain the OCR result to the parsing step
        if workflow_option == 2:
            # Pass the OCR output to the parsing prompt
            messages = [
                {"role": "user", "content": PROMPT_PARSE_A + ocr_text + PROMPT_PARSE_B}  # Parsing rules prompt
            ]
            text_prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Re-process for parsing
            inputs = self.processor(
                text=[text_prompt], return_tensors="pt"
            ).to("cuda")

            # Generate structured JSON output
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.MAX_TOKENS, eos_token_id=self.processor.tokenizer.eos_token_id
            )
            generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
            json_out = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

        return ocr_text, json_out


# Usage Example
def main():
    img_path = 'D:/D_Desktop/BR_1839468565_Ochnaceae_Campylospermum_reticulatum_label.jpg'
    # img_path = 'C:/Users/willwe/Desktop/MICH_16205594_Poaceae_Jouvea_pilosa.jpg'
    # img_path = 'C:/Users/willwe/Desktop/stewart.jpg'

    image = Image.open(img_path)

    # Create the Phi model object, set eager mode for non-flash attention
    ocr = Phi35VisionOCR(logger=None, model_id='microsoft/Phi-3.5-vision-instruct', attn_implementation='eager')

    # Run OCR workflow
    ocr_text, json_out = ocr.ocr_with_vlm(image, workflow_option=1)
    print("Final OCR Text:", ocr_text)
    print("JSON Out:", json_out)

    # Check if ocr_text is valid JSON for workflows 2 and 3
    try:
        if json_out:
            final_json = json.loads(json_out)
            print("Parsed JSON:", json.dumps(final_json, indent=4))  # Pretty-print the JSON
        print("No JSON")
    except json.JSONDecodeError:
        print("Invalid JSON")

if __name__ == '__main__':
    main()
