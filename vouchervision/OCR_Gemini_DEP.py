from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.pydantic_v1 import BaseModel, Field
import requests, logging, os, vertexai, json
from PIL import Image as PILImage
from io import BytesIO
import http.client
import typing
import urllib.request
from google.oauth2 import service_account
from vertexai.preview.generative_models import GenerativeModel, Image

class Transcription(BaseModel):
    Transcription_Printed_Text: str = Field(description="The transcription of all printed text in the image.")
    Transcription_Handwritten_Text: str = Field(description="The transcription of all handwritten text in the image.")

class OCRGeminiProVision:
    def __init__(self, logger, model_name="gemini-pro-vision"):
        self.logger = logger
        self.llm = GenerativeModel(model_name)

        # self.llm = ChatGoogleGenerativeAI(model=model_name)
        self.parser = JsonOutputParser(pydantic_object=Transcription)   

    def image_to_vertex_image(self, image_path: str) -> Image:
        """Converts a local image or image URL to a Vertex AI Image object."""
        if image_path.startswith("http"):
            # Load image from URL
            with urllib.request.urlopen(image_path) as response:
                response = typing.cast(http.client.HTTPResponse, response)
                image_bytes = response.read()
        else:
            # Load image from local file
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()
        
        return Image.from_bytes(image_bytes)
    
    def combine_json_values(self, data, separator=" "):
        """
        Recursively traverses through a JSON-like dictionary or list,
        combining all the values into a single string with a given separator.
        
        :return: A single string containing all values from the input.
        """
        # Base case for strings, directly return the string
        if isinstance(data, str):
            return data
        
        # If the data is a dictionary, iterate through its values
        elif isinstance(data, dict):
            combined_string = separator.join(self.combine_json_values(v, separator) for v in data.values())
        
        # If the data is a list, iterate through its elements
        elif isinstance(data, list):
            combined_string = separator.join(self.combine_json_values(item, separator) for item in data)
        
        # For other data types (e.g., numbers), convert to string directly
        else:
            combined_string = str(data)
        
        return combined_string

    def transcribe_image(self, image_file, prompt):
        # Load the image
        image = self.image_to_vertex_image(image_file)
        
        # Convert the image to base64

        # Construct the message
        # message = HumanMessage(
        #     content=[
        #         {"type": "text", 
        #          "text": prompt},
        #         {"type": "image", "image": image_base64},
        #         # {"type": "image", "image": f"data:image/png;base64,{image_base64}"},
        #     ]
        # )

        # Invoke the model
        # direct_output = self.llm.invoke([message])
        response = self.llm.generate_content(
            [prompt, image]
        )
        direct_output = response.text[1:]
        print(direct_output)

        # Parse the output to JSON format using the specified schema.
        try:
            json_output = self.parser.parse(direct_output)
        except:
            json_output = direct_output

        try:
            str_output = self.combine_json_values(json_output)
        except:
            str_output = direct_output

        return image, json_output, direct_output, str_output, None


PROMPT_OCR = """I need you to transcribe all of the text in this image. 
        Place the transcribed text into a JSON dictionary with this form {"Transcription_Printed_Text": "text","Transcription_Handwritten_Text": "text"}"""
PROMPT_ALL = """1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below.
2. Map the unstructured OCR text to the appropriate JSON key and populate the field given the user-defined rules.
3. JSON key values are permitted to remain empty strings if the corresponding information is not found in the unstructured OCR text.
4. Duplicate dictionary fields are not allowed.
5. Ensure all JSON keys are in camel case.
6. Ensure new JSON field values follow sentence case capitalization.
7. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
8. Ensure output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
9. Only return a JSON dictionary represented as a string. You should not explain your answer.
This section provides rules for formatting each JSON value organized by the JSON key.
{catalogNumber Barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits., order The full scientific name of the order in which the taxon is classified. Order must be capitalized., family The full scientific name of the family in which the taxon is classified. Family must be capitalized., scientificName The scientific name of the taxon including genus, specific epithet, and any lower classifications., scientificNameAuthorship The authorship information for the scientificName formatted according to the conventions of the applicable Darwin Core nomenclaturalCode., genus Taxonomic determination to genus. Genus must be capitalized. If genus is not present use the taxonomic family name followed by the word 'indet'., subgenus The full scientific name of the subgenus in which the taxon is classified. Values should include the genus to avoid homonym confusion., specificEpithet The name of the first or species epithet of the scientificName. Only include the species epithet., infraspecificEpithet The name of the lowest or terminal infraspecific epithet of the scientificName, excluding any rank designation., identifiedBy A comma separated list of names of people, groups, or organizations who assigned the taxon to the subject organism. This is not the specimen collector., recordedBy A comma separated list of names of people, groups, or organizations responsible for observing, recording, collecting, or presenting the original specimen. The primary collector or observer should be listed first., recordNumber An identifier given to the occurrence at the time it was recorded. Often serves as a link between field notes and an occurrence record, such as a specimen collector's number., verbatimEventDate The verbatim original representation of the date and time information for when the specimen was collected. Date of collection exactly as it appears on the label. Do not change the format or correct typos., eventDate Date the specimen was collected formatted as year-month-day, YYYY-MM_DD. If specific components of the date are unknown, they should be replaced with zeros. Examples \0000-00-00\ if the entire date is unknown, \YYYY-00-00\ if only the year is known, and \YYYY-MM-00\ if year and month are known but day is not., habitat A category or description of the habitat in which the specimen collection event occurred., occurrenceRemarks Text describing the specimen's geographic location. Text describing the appearance of the specimen. A statement about the presence or absence of a taxon at a the collection location. Text describing the significance of the specimen, such as a specific expedition or notable collection. Description of plant features such as leaf shape, size, color, stem texture, height, flower structure, scent, fruit or seed characteristics, root system type, overall growth habit and form, any notable aroma or secretions, presence of hairs or bristles, and any other distinguishing morphological or physiological characteristics., country The name of the country or major administrative unit in which the specimen was originally collected., stateProvince The name of the next smaller administrative region than country (state, province, canton, department, region, etc.) in which the specimen was originally collected., county The full, unabbreviated name of the next smaller administrative region than stateProvince (county, shire, department, parish etc.) in which the specimen was originally collected., municipality The full, unabbreviated name of the next smaller administrative region than county (city, municipality, etc.) in which the specimen was originally collected., locality Description of geographic location, landscape, landmarks, regional features, nearby places, or any contextual information aiding in pinpointing the exact origin or location of the specimen., degreeOfEstablishment Cultivated plants are intentionally grown by humans. In text descriptions, look for planting dates, garden locations, ornamental, cultivar names, garden, or farm to indicate cultivated plant. Use either - unknown or cultivated., decimalLatitude Latitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format., decimalLongitude Longitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format., verbatimCoordinates Verbatim location coordinates as they appear on the label. Do not convert formats. Possible coordinate types include [Lat, Long, UTM, TRS]., minimumElevationInMeters Minimum elevation or altitude in meters. Only if units are explicit then convert from feet (\ft\ or \ft.\\ or \feet\) to meters (\m\ or \m.\ or \meters\). Round to integer., maximumElevationInMeters Maximum elevation or altitude in meters. If only one elevation is present, then max_elevation should be set to the null_value. Only if units are explicit then convert from feet (\ft\ or \ft.\ or \feet\) to meters (\m\ or \m.\ or \meters\). Round to integer.}
Please populate the following JSON dictionary based on the rules and the unformatted OCR text
{
catalogNumber ,
order ,
family ,
scientificName ,
scientificNameAuthorship ,
genus ,
subgenus ,
specificEpithet ,
infraspecificEpithet ,
identifiedBy ,
recordedBy ,
recordNumber ,
verbatimEventDate ,
eventDate ,
habitat ,
occurrenceRemarks ,
country ,
stateProvince ,
county ,
municipality ,
locality ,
degreeOfEstablishment ,
decimalLatitude ,
decimalLongitude ,
verbatimCoordinates ,
minimumElevationInMeters ,
maximumElevationInMeters 
}
  """
def _get_google_credentials():
    with open('', 'r') as file:
        data = json.load(file)
        creds_json_str = json.dumps(data)
        credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json_str))
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_json_str
        os.environ['GOOGLE_API_KEY'] = ''
        return credentials
    
if __name__ == '__main__':
    vertexai.init(project='', location='', credentials=_get_google_credentials())

    logger = logging.getLogger('LLaVA')
    logger.setLevel(logging.DEBUG)
    
    OCR_Gemini = OCRGeminiProVision(logger)
    image, json_output, direct_output, str_output, usage_report = OCR_Gemini.transcribe_image(
        # "C:/Users/Will/Downloads/gallery_short_gpt4t_trOCRhand/Cropped_Images/By_Class/label/MICH_7574789_Cyperaceae_Carex_scoparia.jpg",
        # "D:/D_Desktop/usda_out/usda/Original_Images/4.jpg",
        "D:/Dropbox/VoucherVision/demo/demo_images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg",
                                                                                            PROMPT_OCR)
    print('json_output')
    print(json_output)
    print('direct_output')
    print(direct_output)
    print('str_output')
    print(str_output)
    print('usage_report')
    print(usage_report)
