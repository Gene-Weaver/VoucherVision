import os
import base64
import requests
from io import BytesIO
from PIL import Image

from OCR_resize_for_VLMs import resize_image_to_min_max_pixels
from general_utils import calculate_cost

PROMPT_SIMPLE = """Please perform OCR on this scientific image and extract all of the words and text verbatim. Do not explain your answer, only return the verbatim text:"""
PROMPT_OCR_AND_PARSE = """Perform OCR on this image. Return only the verbatim text without any explanation. Then complete the following task:
Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
The rules are:
1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below. 2. Map the unstructured OCR text to the appropriate JSON key and populate the field given the user-defined rules. 3. JSON key values are permitted to remain empty strings if the corresponding information is not found in the unstructured OCR text. 4. Duplicate dictionary fields are not allowed. 5. Ensure all JSON keys are in camel case. 6. Ensure new JSON field values follow sentence case capitalization. 7. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template. 8. Ensure output JSON string is valid JSON format. It should not have trailing commas or unquoted keys. 9. Only return a JSON dictionary represented as a string. You should not explain your answer.
This section provides rules for formatting each JSON value organized by the JSON key.
This is the JSON template that includes instructions for each key:
catalogNumber: Barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits. scientificName: The scientific name of the taxon including genus, specific epithet, and any lower classifications. genus: Taxonomic determination to genus. Genus must be capitalized. If genus is not present use the taxonomic family name followed by the word "indet". specificEpithet: The name of the species epithet of the scientificName. Only include the species epithet. speciesNameAuthorship: The authorship information for the scientificName formatted according to the conventions of the applicable Darwin Core nomenclatural code. collectedBy: A comma separated list of names of people, groups, or organizations responsible for observing, recording, collecting, or presenting the original specimen. The primary collector or observer should be listed first.  collectorNumber: An identifier given to the occurrence at the time it was recorded, the specimen collectors number. identifiedBy: A comma separated list of names of people, groups, or organizations who assigned the taxon to the subject organism. This is not the specimen collector.  identifiedDate: Date that the most recent determination was given, in the following format. YYYY-MM-DD (zeros may be used if only partial date). identifiedConfidence: The determiner may wish to indicate additional information related to their determination. Record this information if it is present. Some examples are '?', 'cf.', 'aff.', 'sensu lato', 's.l.', 'sensu stricto', 's.s.', 'probably', '? hybrid'. The Confidence field is for short comments about the determination (16 characters maximum). Periods are used in abbreviations in this field. identificationHistory: If there are multiple species or genus names, provide the date and determiner for each name and determination. verbatimCollectionDate: The verbatim original representation of the date and time information for when the specimen was collected. Date of collection exactly as it appears on the label. Do not change the format or correct typos. collectionDate: Date the specimen was collected formatted as year-month-day, YYYY-MM-DD. If specific components of the date are unknown, they should be replaced with zeros. Use 0000-00-00 if the entire date is unknown, YYYY-00-00 if only the year is known, and YYYY-MM-00 if year and month are known but day is not. collectionDateEnd: If a range of collection dates is provided, this is the later end date while collectionDate is the beginning date. Use the same formatting as for collectionDate. habitat: Description of the ecological habitat in which the specimen collection event occurred. specimenDescription: Verbatim text describing the specimen itself, including color, measurements not specifically tied to a determination remark, observations of reproductive characters, growth form, taste, smell, etc.  cultivated: Cultivated plants are intentionally grown by humans. In text descriptions, look for planting dates, garden, cult, cultivated, ornamental, cultivar names, garden, or farm to indicate cultivated plant. Use yes if cultivated, otherwise leave blank. continent: Use your knowledge to infer the continent where the natural history museum specimen was originally collected. country: Use your knowledge and the OCR text to infer the country where the natural history museum specimen was originally collected. stateProvince: The name of the sub-national administrative region than country (state, province, canton, department, region, etc.) in which the specimen was originally collected. county: The full, unabbreviated name of the next smaller administrative region than stateProvince (county, shire, department, parish etc.) in which the specimen was originally collected. locality: Description of geographic location, landscape, landmarks, regional features, nearby places, municipality, city, or any contextual information aiding in pinpointing the exact origin or location of the specimen. verbatimCoordinates: Verbatim location coordinates as they appear on the label. Do not convert formats. Possible coordinate types include [Lat, Long, UTM, TRS]. decimalLatitude: Latitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format. decimalLongitude: Longitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format. minimumElevationInMeters: Minimum elevation or altitude in meters. Only if units are explicit then convert from feet ("ft" or "ft."" or "feet") to meters ("m" or "m." or "meters"). Round to integer. Values greater than 6000 are in feet and need to be converted. maximumElevationInMeters: Maximum elevation or altitude in meters. If only one elevation is present, then max_elevation should be set to the null_value. Only if units are explicit then convert from feet ("ft" or "ft." or "feet") to meters ("m" or "m." or "meters"). Round to integer. Values greater than 6000 are in feet and need to be converted. elevationUnits: Use m if the final elevation is reported in meters. If you convert from feet to meters, then use m. additionalText: All remaining OCR text and text that is not part of the main label, secondary text, background and supporting information. 
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
"identifiedBy": "",
"identifiedDate": "",
"identifiedConfidence": "",
"identifiedRemarks": "",
"identificationHistory": "",
"verbatimCollectionDate": "",
"collectionDate": "",
"collectionDateEnd": "",
"habitat": "",
"specimenDescription": "",
"cultivated": "",
"continent": "",
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

class HyperbolicOCR:
    def __init__(self, api_key, model_id):
        self.api_key = api_key
        self.api_url = "https://api.hyperbolic.xyz/v1/chat/completions"
        # self.PROMPT = PROMPT_OCR_AND_PARSE
        self.PROMPT = PROMPT_SIMPLE
        self.model_id = model_id
        self.path_api_cost = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_cost', 'api_cost.yaml')

    def encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def ocr_hyperbolic(self, image_path, max_tokens=2048, temperature=0.1, top_p=0.001):
        # Open and resize the image
        image = Image.open(image_path)
        resized_image = resize_image_to_min_max_pixels(image)

        # Encode the resized image
        base64_image = self.encode_image(resized_image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                # "detail": resolution,
                            },
                        },
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        try:
            response_json = response.json()

            if "choices" in response_json:
                parsed_answer = response_json["choices"][0]["message"]["content"]
            else:
                parsed_answer = None

            if "usage" in response_json:
                usage = response_json["usage"]
                tokens_in = usage["prompt_tokens"]
                tokens_out = usage["completion_tokens"]
            else:
                usage = None
                tokens_in = 0
                tokens_out = 0
            

            if self.model_id == "mistralai/Pixtral-12B-2409": 
                total_cost = calculate_cost('Hyperbolic_VLM_Pixtral_12B', self.path_api_cost, tokens_in, tokens_out)
            elif self.model_id == "Llama-3.2-90B-Vision-Instruct":
                total_cost = calculate_cost('Hyperbolic_VLM_Llama_3_2_90B_Vision_Instruct', self.path_api_cost, tokens_in, tokens_out)
            elif self.model_id == "Qwen/Qwen2-VL-72B-Instruct":
                total_cost = calculate_cost('Hyperbolic_VLM_Qwen2_VL_72B_Instruct', self.path_api_cost, tokens_in, tokens_out)
            elif self.model_id == "Qwen/Qwen2-VL-7B-Instruct":
                total_cost = calculate_cost('Hyperbolic_VLM_Qwen2_VL_7B_Instruct', self.path_api_cost, tokens_in, tokens_out)
            else:
                print("invalid model_id for HyperbolicOCR") # Very impressive OCR

            cost_in, cost_out, total_cost, rates_in, rates_out = total_cost

            return parsed_answer, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out
        except Exception as e:
            print(f"Error with OCR: {e}")
            return None, 0, 0, 0, None, None, 0, 0

def main():
    # Example image path
    img_path = 'D:/D_Desktop/BR_1839468565_Ochnaceae_Campylospermum_reticulatum_label.jpg'
    
    # Replace with your actual Hyperbolic API key
    API_KEY = ""

    model_id = "mistralai/Pixtral-12B-2409" # Very impressive single shot performance
    # model_id = "Llama-3.2-90B-Vision-Instruct" # Not yet hosted
    # model_id = "Qwen/Qwen2-VL-72B-Instruct" # Very impressive OCR
    # model_id = "Qwen/Qwen2-VL-7B-Instruct" # Very impressive OCR
    
    ocr = HyperbolicOCR(API_KEY,model_id=model_id)
    
    parsed_answer, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr.ocr_hyperbolic(img_path, 
                                                                                                                max_tokens=1024)
    print(f"Parsed Answer:\n{parsed_answer}")
    print(f"Usage:\ntokens_in:{tokens_in}\ntokens_out:{tokens_out}\n\n")

if __name__ == '__main__':
    main()
