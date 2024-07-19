import random, os
from PIL import Image
import copy
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
from PIL import Image, ImageDraw, ImageFont 
import numpy as np
import warnings
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from vouchervision.utils_LLM import SystemLoadMonitor
except:
    from utils_LLM import SystemLoadMonitor


warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

class FlorenceOCR:
    # def __init__(self, logger, model_id='microsoft/Florence-2-base'):
    def __init__(self, logger, model_id='microsoft/Florence-2-large'):
        self.MAX_TOKENS = 1024
        self.logger = logger
        self.model_id = model_id

        self.monitor = SystemLoadMonitor(logger)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # self.model_id_clean = "mistralai/Mistral-7B-v0.3"
        self.model_id_clean = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
        self.tokenizer_clean = AutoTokenizer.from_pretrained(self.model_id_clean)
        # Configuring the BitsAndBytesConfig for quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            quant_method="bnb",
        )
        self.model_clean = AutoModelForCausalLM.from_pretrained(
            self.model_id_clean,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,)
        

    def ocr_florence(self, image, task_prompt='<OCR>', text_input=None):
        self.monitor.start_monitoring_usage()

        # Open image if a path is provided
        if isinstance(image, str):
            image = Image.open(image)

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # Move input_ids and pixel_values to the same device as the model
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.MAX_TOKENS,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer_dict = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )

        parsed_answer_text = parsed_answer_dict[task_prompt]

        # Prepare input for the second model
        inputs_clean = self.tokenizer_clean(
            f"Insert spaces into this text to make all the words valid. This text contains scientific names of plants, locations, habitat, coordinate words: {parsed_answer_text}", 
            return_tensors="pt"
        )
        inputs_clean = {key: value.to(self.model_clean.device) for key, value in inputs_clean.items()}

        outputs_clean = self.model_clean.generate(**inputs_clean, max_new_tokens=self.MAX_TOKENS)
        text_with_spaces = self.tokenizer_clean.decode(outputs_clean[0], skip_special_tokens=True)

        # Extract only the LLM response from the decoded text
        response_start = text_with_spaces.find(parsed_answer_text)
        if response_start != -1:
            text_with_spaces = text_with_spaces[response_start + len(parsed_answer_text):].strip()

        print(text_with_spaces)

        self.monitor.stop_inference_timer() # Starts tool timer too
        usage_report = self.monitor.stop_monitoring_report_usage()   

        return text_with_spaces, parsed_answer_text, parsed_answer_dict, usage_report


def main():
    # img_path = '/home/brlab/Downloads/gem_2024_06_26__02-26-02/Cropped_Images/By_Class/label/1.jpg'
    img_path = 'D:/D_Desktop/BR_1839468565_Ochnaceae_Campylospermum_reticulatum_label.jpg'

    image = Image.open(img_path)

    # ocr = FlorenceOCR(logger = None, model_id='microsoft/Florence-2-base')
    ocr = FlorenceOCR(logger = None, model_id='microsoft/Florence-2-large')
    results_text, results_all, results_dirty, usage_report = ocr.ocr_florence(image, task_prompt='<OCR>', text_input=None)
    print(results_text)

if __name__ == '__main__':
    main()
