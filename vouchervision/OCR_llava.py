import os, re, logging
import requests
from PIL import Image
from io import BytesIO
import torch
# from transformers import AutoTokenizer, BitsAndBytesConfig, TextStreamer

# from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# from vouchervision.LLaVA.llava.model import LlavaLlamaForCausalLM
from vouchervision.LLaVA.llava.model.builder import load_pretrained_model
from vouchervision.LLaVA.llava.conversation import conv_templates#, SeparatorStyle
from vouchervision.LLaVA.llava.utils import disable_torch_init
from vouchervision.LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from vouchervision.LLaVA.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images #KeywordsStoppingCriteria

from vouchervision.utils_LLM import SystemLoadMonitor

'''
Performance expectations system: 
    GPUs:
        2x RTX6000 Ada 
    CPU:
        AMD Ryzen threadripper pro 5975wx 32-cores x64 threads
    RAM:
        512 GB
    OS:
        Ubuntu 22.04.3 LTS

LLaVA Models:
    "liuhaotian/llava-v1.6-mistral-7b" --- Model is 20 GB in size --- Mistral-7B
        --- Full
            --- Inference time ~6 sec
            --- VRAM ~18 GB

        --- 8bit (don't use. author says there is a problem right now, 2024-02-08) anecdotally worse results too
            --- Inference time ~37 sec
            --- VRAM ~18 GB

        --- 4bit
            --- Inference time ~15 sec
            --- VRAM ~9 GB

            
    "liuhaotian/llava-v1.6-34b" --- Model is 100 GB in size --- Hermes-Yi-34B
        --- Full
            --- Inference time ~21 sec
            --- VRAM ~70 GB

        --- 8bit (don't use. author says there is a problem right now, 2024-02-08) anecdotally worse results too
            --- Inference time ~52 sec
            --- VRAM ~42 GB

        --- 4bit
            --- Inference time ~23 sec
            --- VRAM ~25GB

            
    "liuhaotian/llava-v1.6-vicuna-13b" --- Model is 30 GB in size --- Vicuna-13B
        --- Full
            --- Inference time ~8 sec
            --- VRAM ~33 GB

        --- 8bit (don't use. author says there is a problem right now, 2024-02-08) anecdotally worse results too, has lots of ALL CAPS and mistakes
            --- Inference time ~32 sec
            --- VRAM ~23 GB

        --- 4bit
            --- Inference time ~12 sec
            --- VRAM ~15 GB

            
    "liuhaotian/llava-v1.6-vicuna-7b" --- Model is 15 GB in size --- Vicuna-7B
        --- Full
            --- Inference time ~7 sec
            --- VRAM ~20 GB

        --- 8bit (don't use. author says there is a problem right now, 2024-02-08) anecdotally worse results too
            --- Inference time ~27 sec
            --- VRAM ~14 GB

        --- 4bit
            --- Inference time ~10 sec
            --- VRAM ~10 GB


'''

# OCR_Llava = OCRLlava()
# image, caption = OCR_Llava.transcribe_image("path/to/image.jpg", "Describe this image.")
# print(caption)

# Define the desired data structure for the transcription.
class Transcription(BaseModel):
    Transcription_Printed_Text: str = Field(description="The transcription of all printed text in the image.")
    Transcription_Handwritten_Text: str = Field(description="The transcription of all handwritten text in the image.")

class OCRllava:
    def __init__(self, logger, model_path="liuhaotian/llava-v1.6-34b",load_in_4bit=False, load_in_8bit=False):
        self.monitor = SystemLoadMonitor(logger)

        # self.model_path = "liuhaotian/llava-v1.6-mistral-7b"
        # self.model_path = "liuhaotian/llava-v1.6-34b"
        # self.model_path = "liuhaotian/llava-v1.6-vicuna-13b"

        self.model_path = model_path

        # kwargs = {"device_map": "auto", "load_in_4bit": load_in_4bit, "quantization_config": BitsAndBytesConfig(
        #     load_in_4bit=load_in_4bit,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=load_in_4bit,
        #     bnb_4bit_quant_type='nf4'
        # )}

        

        if "llama-2" in self.model_path.lower(): # this is borrowed from def eval_model(args): in run_llava.py
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_path.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_path.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_path.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_path.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, None, 
                                                                               model_name = get_model_name_from_path(self.model_path),
                                                                               load_8bit=load_in_8bit, load_4bit=load_in_4bit)

        # self.model = LlavaLlamaForCausalLM.from_pretrained(self.model_path, low_cpu_mem_usage=True, **kwargs)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        # self.vision_tower = self.model.get_vision_tower()
        # if not self.vision_tower.is_loaded:
            # self.vision_tower.load_model()
        # self.vision_tower.to(device='cuda')
        # self.image_processor = self.vision_tower.image_processor
        self.parser = JsonOutputParser(pydantic_object=Transcription)   

    def image_parser(self):
        sep = ","
        out = self.image_file.split(sep)
        return out
    
    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
    
    def load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)
            out.append(image)
        return out
    
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

    def transcribe_image(self, image_file, prompt, max_new_tokens=512, temperature=0.1, top_p=None, num_beams=1):
        self.monitor.start_monitoring_usage()
        
        self.image_file = image_file
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        disable_torch_init()

        qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_files = self.image_parser()
        images = self.load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                # top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        direct_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Parse the output to JSON format using the specified schema.
        try:
            json_output = self.parser.parse(direct_output)
        except:
            json_output = direct_output

        try:
            str_output = self.combine_json_values(json_output)
        except:
            str_output = direct_output

        self.monitor.stop_inference_timer() # Starts tool timer too
        usage_report = self.monitor.stop_monitoring_report_usage()    
        

        return image, json_output, direct_output, str_output, usage_report


PROMPT_OCR = """I need you to transcribe all of the text in this image. Place the transcribed text into a JSON dictionary with this form {"Transcription": "text"}"""
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
if __name__ == '__main__':
    logger = logging.getLogger('LLaVA')
    logger.setLevel(logging.DEBUG)
    
    OCR_Llava = OCRllava(logger)
    image, json_output, direct_output, str_output, usage_report = OCR_Llava.transcribe_image("/home/brlab/Dropbox/VoucherVision/demo/demo_images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg",
                                                                                            PROMPT_OCR)
    print('json_output')
    print(json_output)
    print('direct_output')
    print(direct_output)
    print('str_output')
    print(str_output)
    print('usage_report')
    print(usage_report)
