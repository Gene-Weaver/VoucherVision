import random, os
from PIL import Image
import copy
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
from PIL import Image, ImageDraw, ImageFont 
import numpy as np
import warnings
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from vouchervision.utils_LLM import SystemLoadMonitor

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

class FlorenceOCR:
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
        self.model_clean = AutoModelForCausalLM.from_pretrained(self.model_id_clean)
        

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
        parsed_answer_dirty = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )

        inputs = self.tokenizer_clean(f"Insert spaces into this text to make all the words valid. This text contains scientific names of plants, locations, habitat, coordinate words: {parsed_answer_dirty[task_prompt]}", return_tensors="pt")
        inputs = {key: value.to(self.model_clean.device) for key, value in inputs.items()}

        outputs = self.model_clean.generate(**inputs, max_new_tokens=self.MAX_TOKENS)
        parsed_answer = self.tokenizer_clean.decode(outputs[0], skip_special_tokens=True)
        print(parsed_answer_dirty)
        print(parsed_answer)

        self.monitor.stop_inference_timer() # Starts tool timer too
        usage_report = self.monitor.stop_monitoring_report_usage()   

        return parsed_answer, parsed_answer_dirty[task_prompt], parsed_answer_dirty, usage_report


def main():
    img_path = '/home/brlab/Downloads/gem_2024_06_26__02-26-02/Cropped_Images/By_Class/label/1.jpg'
    # img = 'D:/D_Desktop/BR_1839468565_Ochnaceae_Campylospermum_reticulatum_label.jpg'

    image = Image.open(img_path)

    ocr = FlorenceOCR(logger = None)
    results_text, results, usage_report = ocr.ocr_florence(image, task_prompt='<OCR>', text_input=None)
    print(results_text)

if __name__ == '__main__':
    main()
