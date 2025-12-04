import os, io, sys, inspect, statistics, json, cv2
from statistics import mean 
# from google.cloud import vision, storage
from google.cloud import vision
from google.cloud import vision_v1p3beta1 as vision_beta
from PIL import Image, ImageDraw, ImageFont
import colorsys
from tqdm import tqdm
from google.oauth2 import service_account
### LLaVA should only be installed if the user will actually use it.
### It requires the most recent pytorch/Python and can mess with older systems

from OCR_sanitize import sanitize_for_storage

'''
@misc{li2021trocr,
      title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models}, 
      author={Minghao Li and Tengchao Lv and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
      year={2021},
      eprint={2109.10282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@inproceedings{baek2019character,
  title={Character Region Awareness for Text Detection},
  author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9365--9374},
  year={2019}
}
'''

class OCREngine:

    BBOX_COLOR = "black"

    def __init__(self, logger, json_report, dir_home, is_hf, cfg, trOCR_model_version, trOCR_model, trOCR_processor, device):
        self.is_hf = is_hf
        self.logger = logger

        self.json_report = json_report

        self.cfg = cfg
        self.do_use_trOCR = self.cfg['leafmachine']['project']['do_use_trOCR']
        # self.do_use_florence = self.cfg['leafmachine']['project']['do_use_florence']
        self.OCR_option = self.cfg['leafmachine']['project']['OCR_option']
        self.double_OCR = self.cfg['leafmachine']['project']['double_OCR']
        self.dir_home = dir_home

        # Initialize TrOCR components
        self.trOCR_model_version = trOCR_model_version
        self.trOCR_processor = trOCR_processor
        self.trOCR_model = trOCR_model
        self.device = device

        self.OCR_JSON_to_file = {}

        # for paid vLM OCR like GPT-vision
        self.cost = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_in = 0
        self.cost_out = 0

        self.ocr_method = set()

        self.hand_cleaned_text = None
        self.hand_organized_text = None
        self.hand_bounds = None
        self.hand_bounds_word = None
        self.hand_bounds_flat = None
        self.hand_text_to_box_mapping = None
        self.hand_height = None
        self.hand_confidences = None
        self.hand_characters = None

        self.normal_cleaned_text = None
        self.normal_organized_text = None
        self.normal_bounds = None
        self.normal_bounds_word = None
        self.normal_text_to_box_mapping = None
        self.normal_bounds_flat = None
        self.normal_height = None
        self.normal_confidences = None
        self.normal_characters = None

        self.trOCR_texts = None
        self.trOCR_text_to_box_mapping = None
        self.trOCR_bounds_flat = None
        self.trOCR_height = None
        self.trOCR_confidences = None
        self.trOCR_characters = None
        self.set_client()
        self.init_florence()
        self.init_gpt_4o_mini()
        self.init_gemini_pro()
        self.init_hyperbolic()
        self.init_Qwen2VL()
        self.init_craft()

        self.multimodal_prompt = """I need you to transcribe all of the text in this image. 
        Place the transcribed text into a JSON dictionary with this form {"Transcription_Printed_Text": "text","Transcription_Handwritten_Text": "text"}"""
        self.init_llava()

        
    def set_client(self):
        # Only init Google Vision if it is needed
        if 'hand' in self.OCR_option or 'normal' in self.OCR_option:
            if self.is_hf:
                self.client_beta = vision_beta.ImageAnnotatorClient(credentials=self.get_google_credentials())
                self.client = vision.ImageAnnotatorClient(credentials=self.get_google_credentials())
            else:
                self.client_beta = vision_beta.ImageAnnotatorClient(credentials=self.get_google_credentials()) 
                self.client = vision.ImageAnnotatorClient(credentials=self.get_google_credentials())


    def get_google_credentials(self):
        creds_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json_str))
        return credentials
    
    def init_craft(self):
        if 'CRAFT' in self.OCR_option:
            from craft_text_detector import load_craftnet_model, load_refinenet_model

            try:
                self.refine_net = load_refinenet_model(cuda=True)
                self.use_cuda = True
            except:
                self.refine_net = load_refinenet_model(cuda=False)
                self.use_cuda = False

            if self.use_cuda:
                self.craft_net = load_craftnet_model(weight_path=os.path.join(self.dir_home,'vouchervision','craft','craft_mlt_25k.pth'), cuda=True)
            else:
                self.craft_net = load_craftnet_model(weight_path=os.path.join(self.dir_home,'vouchervision','craft','craft_mlt_25k.pth'), cuda=False)

    def init_florence(self):
        if 'LOCAL Florence-2' in self.OCR_option:
            from OCR_Florence_2 import FlorenceOCR
            self.Florence = FlorenceOCR(logger=self.logger, model_id=self.cfg['leafmachine']['project']['florence_model_path'])

    def init_gpt_4o_mini(self):
        if 'GPT-4o-mini' in self.OCR_option:
            from vouchervision.OCR_GPT4o import GPT4oOCR
            self.GPTmini = GPT4oOCR(api_key = os.getenv('OPENAI_API_KEY'))
        if 'GPT-4o' in self.OCR_option:
            from vouchervision.OCR_GPT4o import GPT4oOCR
            self.GPT4o = GPT4oOCR(api_key = os.getenv('OPENAI_API_KEY'))

    def init_gemini_pro(self):
        if any(model in self.OCR_option for model in [
            "Gemini-3-Pro-Preview", 
            "gemini-3-pro-preview", 
            "Gemini-2.5-Pro", 
            "Gemini-2.5-Flash", 
            "Gemini-2.0-Flash", 
            "Gemini-1.5-Pro", 
            "Gemini-1.5-Flash", 
            "Gemini-1.5-Flash-8B"
        ]):
            from OCR_Gemini import OCRGeminiProVision

        if "Gemini-3-Pro-Preview" in self.OCR_option:
            self.GeminiProVision_3_Pro = OCRGeminiProVision(api_key = os.getenv('GOOGLE_API_KEY'),model_name="gemini-3-pro-preview") # TODO update to non-preview
        elif "gemini-3-pro-preview" in self.OCR_option:
            self.GeminiProVision_3_Pro = OCRGeminiProVision(api_key = os.getenv('GOOGLE_API_KEY'),model_name="gemini-3-pro-preview") # TODO update to non-preview
        if "Gemini-2.5-Pro" in self.OCR_option:
            self.GeminiProVision_2_5_Pro = OCRGeminiProVision(api_key = os.getenv('GOOGLE_API_KEY'),model_name="gemini-2.5-pro")
        if "Gemini-2.5-Flash" in self.OCR_option:  
            self.GeminiProVision_2_5_Flash = OCRGeminiProVision(api_key = os.getenv('GOOGLE_API_KEY'),model_name="gemini-2.5-flash") 
        if "Gemini-2.0-Flash" in self.OCR_option:
            self.GeminiProVision_2_0_Flash = OCRGeminiProVision(api_key = os.getenv('GOOGLE_API_KEY'),model_name="gemini-2.0-flash")
        if 'Gemini-1.5-Pro' in self.OCR_option:
            self.GeminiProVision_1_5_Pro = OCRGeminiProVision(api_key = os.getenv('GOOGLE_API_KEY'),model_name="gemini-1.5-pro")
        if "Gemini-1.5-Flash" in self.OCR_option:
            self.GeminiProVision_1_5_Flash = OCRGeminiProVision(api_key = os.getenv('GOOGLE_API_KEY'),model_name="gemini-1.5-flash")
        if "Gemini-1.5-Flash-8B" in self.OCR_option:
            self.GeminiProVision_1_5_Flash_8B = OCRGeminiProVision(api_key = os.getenv('GOOGLE_API_KEY'),model_name="gemini-1.5-flash-8b")

    def init_hyperbolic(self):
        if any(model in self.OCR_option for model in [
                'Pixtral-12B-2409', 
                'Llama-3.2-90B-Vision-Instruct', 
                'Qwen2-VL-72B-Instruct', 
                'Qwen2-VL-7B-Instruct'
            ]):
                from OCR_Hyperbolic_VLMs import HyperbolicOCR
        if 'Pixtral-12B-2409' in self.OCR_option:
            self.Hyperbolic_Pixtral_12B = HyperbolicOCR(api_key = os.getenv('HYPERBOLIC_API_KEY'), model_id="mistralai/Pixtral-12B-2409")
        if 'Llama-3.2-90B-Vision-Instruct' in self.OCR_option:
            self.Hyperbolic_Llama_2_2_90B_Vision = HyperbolicOCR(api_key = os.getenv('HYPERBOLIC_API_KEY'), model_id="Llama-3.2-90B-Vision-Instruct")
        if 'Qwen2-VL-72B-Instruct' in self.OCR_option:
            self.Hyperbolic_Qwen2_VL_72B = HyperbolicOCR(api_key = os.getenv('HYPERBOLIC_API_KEY'), model_id="Qwen/Qwen2-VL-72B-Instruct")
        if 'Qwen2-VL-7B-Instruct' in self.OCR_option:
            self.Hyperbolic_Qwen2_VL_7B = HyperbolicOCR(api_key = os.getenv('HYPERBOLIC_API_KEY'), model_id="Qwen/Qwen2-VL-7B-Instruct")

    def init_Qwen2VL(self):
        if 'LOCAL Qwen-2-VL' in self.OCR_option:
            from OCR_Qwen import Qwen2VLOCR
            self.Qwen2VL = Qwen2VLOCR(logger=self.logger, model_id=self.cfg['leafmachine']['project']['qwen_model_path'])
            

    def init_llava(self):
        if 'LLaVA' in self.OCR_option:
            from vouchervision.OCR_llava import OCRllava

            self.model_path = "liuhaotian/" + self.cfg['leafmachine']['project']['OCR_option_llava']
            self.model_quant = self.cfg['leafmachine']['project']['OCR_option_llava_bit']
            
            if self.json_report:
                self.json_report.set_text(text_main=f'Loading LLaVA model: {self.model_path} Quantization: {self.model_quant}')

            if self.model_quant == '4bit':
                use_4bit = True
            elif self.model_quant == 'full':
                use_4bit = False
            else:
                self.logger.info(f"Provided model quantization invlid. Using 4bit.")
                use_4bit = True

            self.Llava = OCRllava(self.logger, model_path=self.model_path, load_in_4bit=use_4bit, load_in_8bit=False)

    def init_gemini_vision(self):
        pass

    def init_gpt4_vision(self):
        pass
            

    def detect_text_craft(self):
        from craft_text_detector import read_image, get_prediction

        # Perform prediction using CRAFT
        image = read_image(self.path)

        link_threshold = 0.85
        text_threshold = 0.4
        low_text = 0.4

        if self.use_cuda:
            self.prediction_result = get_prediction(
                image=image,
                craft_net=self.craft_net,
                refine_net=self.refine_net,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                low_text=low_text,
                cuda=True,
                long_size=1280
            )
        else:
            self.prediction_result = get_prediction(
                image=image,
                craft_net=self.craft_net,
                refine_net=self.refine_net,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                low_text=low_text,
                cuda=False,
                long_size=1280
            )

        # Initialize metadata structures
        bounds = []
        bounds_word = []  # CRAFT gives bounds for text regions, not individual words
        text_to_box_mapping = []
        bounds_flat = []
        height_flat = []
        confidences = []  # CRAFT does not provide confidences per character, so this might be uniformly set or estimated
        characters = []  # Simulating as CRAFT doesn't provide character-level details
        organized_text = ""
        
        total_b = len(self.prediction_result["boxes"])
        i=0
        # Process each detected text region
        for box in self.prediction_result["boxes"]:
            i+=1
            if self.json_report:
                self.json_report.set_text(text_main=f'Locating text using CRAFT --- {i}/{total_b}')

            vertices = [{"x": int(vertex[0]), "y": int(vertex[1])} for vertex in box]
            
            # Simulate a mapping for the whole detected region as a word
            text_to_box_mapping.append({
                "vertices": vertices,
                "text": "detected_text"  # Placeholder, as CRAFT does not provide the text content directly
            })

            # Assuming each box is a word for the sake of this example
            bounds_word.append({"vertices": vertices})

            # For simplicity, we're not dividing text regions into characters as CRAFT doesn't provide this
            # Instead, we create a single large 'character' per detected region
            bounds.append({"vertices": vertices})
            
            # Simulate flat bounds and height for each detected region
            x_positions = [vertex["x"] for vertex in vertices]
            y_positions = [vertex["y"] for vertex in vertices]
            min_x, max_x = min(x_positions), max(x_positions)
            min_y, max_y = min(y_positions), max(y_positions)
            avg_height = max_y - min_y
            height_flat.append(avg_height)
            
            # Assuming uniform confidence for all detected regions
            confidences.append(1.0)  # Placeholder confidence
            
            # Adding dummy character for each box
            characters.append("X")  # Placeholder character
            
            # Organize text as a single string (assuming each box is a word)
            # organized_text += "detected_text "  # Placeholder text    

        # Update class attributes with processed data
        self.normal_bounds = bounds
        self.normal_bounds_word = bounds_word
        self.normal_text_to_box_mapping = text_to_box_mapping
        self.normal_bounds_flat = bounds_flat  # This would be similar to bounds if not processing characters individually
        self.normal_height = height_flat
        self.normal_confidences = confidences
        self.normal_characters = characters
        self.normal_organized_text = organized_text.strip() 
    

    def detect_text_with_trOCR_using_google_bboxes(self, do_use_trOCR, logger):
        CONFIDENCES = 0.80
        MAX_NEW_TOKENS = 50
        
        ocr_parts = ''
        if not do_use_trOCR:
            if 'normal' in self.OCR_option:
                self.OCR_JSON_to_file['OCR_printed'] = self.normal_organized_text
                # logger.info(f"Google_OCR_Standard:\n{self.normal_organized_text}")
                # ocr_parts = ocr_parts + f"Google_OCR_Standard:\n{self.normal_organized_text}"
                ocr_parts = self.normal_organized_text
            
            if 'hand' in self.OCR_option:
                self.OCR_JSON_to_file['OCR_handwritten'] = self.hand_organized_text
                # logger.info(f"Google_OCR_Handwriting:\n{self.hand_organized_text}")
                # ocr_parts = ocr_parts +  f"Google_OCR_Handwriting:\n{self.hand_organized_text}"
                ocr_parts = self.hand_organized_text

            # if self.OCR_option in ['both',]:
            #     logger.info(f"Google_OCR_Standard:\n{self.normal_organized_text}\n\nGoogle_OCR_Handwriting:\n{self.hand_organized_text}")
            #     return f"Google_OCR_Standard:\n{self.normal_organized_text}\n\nGoogle_OCR_Handwriting:\n{self.hand_organized_text}"
            return ocr_parts
        else:
            logger.info(f'Supplementing with trOCR')

            self.trOCR_texts = []
            original_image = Image.open(self.path).convert("RGB")

            if 'normal' in self.OCR_option or 'CRAFT' in self.OCR_option:
                available_bounds = self.normal_bounds_word
            elif 'hand' in self.OCR_option:
                available_bounds = self.hand_bounds_word 
            # elif self.OCR_option in ['both',]:
            #     available_bounds = self.hand_bounds_word 
            else:
                raise

            text_to_box_mapping = []
            characters = []
            height = []
            confidences = []
            total_b = len(available_bounds)
            i=0
            for bound in tqdm(available_bounds, desc="Processing words using Google Vision bboxes"):
                i+=1
                if self.json_report:
                    self.json_report.set_text(text_main=f'Working on trOCR :construction: {i}/{total_b}')

                vertices = bound["vertices"]

                left = min([v["x"] for v in vertices])
                top = min([v["y"] for v in vertices])
                right = max([v["x"] for v in vertices])
                bottom = max([v["y"] for v in vertices])

                # Crop image based on Google's bounding box
                cropped_image = original_image.crop((left, top, right, bottom))
                pixel_values = self.trOCR_processor(cropped_image, return_tensors="pt").pixel_values

                # Move pixel values to the appropriate device
                pixel_values = pixel_values.to(self.device)

                generated_ids = self.trOCR_model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
                extracted_text = self.trOCR_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                self.trOCR_texts.append(extracted_text)

                # For plotting 
                word_length = max(vertex.get('x') for vertex in vertices) - min(vertex.get('x') for vertex in vertices)
                num_symbols = len(extracted_text)

                Yw = max(vertex.get('y') for vertex in vertices)
                Yo = Yw - min(vertex.get('y') for vertex in vertices)
                X = word_length / num_symbols if num_symbols > 0 else 0
                H = int(X+(Yo*0.1))
                height.append(H)

                map_dict = {
                    "vertices": vertices,
                    "text": extracted_text  # Use the text extracted by trOCR
                    }
                text_to_box_mapping.append(map_dict)

                characters.append(extracted_text)
                confidences.append(CONFIDENCES)
            
            median_height = statistics.median(height) if height else 0
            median_heights = [median_height * 1.5] * len(characters)

            self.trOCR_texts = ' '.join(self.trOCR_texts)

            self.trOCR_text_to_box_mapping = text_to_box_mapping
            self.trOCR_bounds_flat = available_bounds
            self.trOCR_height = median_heights
            self.trOCR_confidences = confidences
            self.trOCR_characters = characters

            if 'normal' in self.OCR_option:
                self.OCR_JSON_to_file['OCR_printed'] = self.normal_organized_text
                self.OCR_JSON_to_file['OCR_trOCR'] = self.trOCR_texts
                # logger.info(f"Google_OCR_Standard:\n{self.normal_organized_text}\n\ntrOCR:\n{self.trOCR_texts}")
                # ocr_parts = ocr_parts +  f"\nGoogle_OCR_Standard:\n{self.normal_organized_text}\n\ntrOCR:\n{self.trOCR_texts}"
                ocr_parts = self.trOCR_texts
            if 'hand' in self.OCR_option:
                self.OCR_JSON_to_file['OCR_handwritten'] = self.hand_organized_text
                self.OCR_JSON_to_file['OCR_trOCR'] = self.trOCR_texts
                # logger.info(f"Google_OCR_Handwriting:\n{self.hand_organized_text}\n\ntrOCR:\n{self.trOCR_texts}")
                # ocr_parts = ocr_parts +  f"\nGoogle_OCR_Handwriting:\n{self.hand_organized_text}\n\ntrOCR:\n{self.trOCR_texts}"
                ocr_parts = self.trOCR_texts
            # if self.OCR_option in ['both',]:
            #     self.OCR_JSON_to_file['OCR_printed'] = self.normal_organized_text
            #     self.OCR_JSON_to_file['OCR_handwritten'] = self.hand_organized_text
            #     self.OCR_JSON_to_file['OCR_trOCR'] = self.trOCR_texts
            #     logger.info(f"Google_OCR_Standard:\n{self.normal_organized_text}\n\nGoogle_OCR_Handwriting:\n{self.hand_organized_text}\n\ntrOCR:\n{self.trOCR_texts}")
            #     ocr_parts = ocr_parts +  f"\nGoogle_OCR_Standard:\n{self.normal_organized_text}\n\nGoogle_OCR_Handwriting:\n{self.hand_organized_text}\n\ntrOCR:\n{self.trOCR_texts}"
            if 'CRAFT' in self.OCR_option:
                # self.OCR_JSON_to_file['OCR_printed'] = self.normal_organized_text
                self.OCR_JSON_to_file['OCR_CRAFT_trOCR'] = self.trOCR_texts
                # logger.info(f"CRAFT_trOCR:\n{self.trOCR_texts}")
                # ocr_parts = ocr_parts +  f"\nCRAFT_trOCR:\n{self.trOCR_texts}"
                ocr_parts = self.trOCR_texts
            return ocr_parts

    @staticmethod
    def confidence_to_color(confidence):
        hue = (confidence - 0.5) * 120 / 0.5
        r, g, b = colorsys.hls_to_rgb(hue/360, 0.5, 1)
        return (int(r*255), int(g*255), int(b*255))


    def render_text_on_black_image(self, option):
        bounds_flat = getattr(self, f'{option}_bounds_flat', [])
        heights = getattr(self, f'{option}_height', [])
        confidences = getattr(self, f'{option}_confidences', [])
        characters = getattr(self, f'{option}_characters', [])

        original_image = Image.open(self.path)
        width, height = original_image.size
        black_image = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(black_image)

        for bound, confidence, char_height, character in zip(bounds_flat, confidences, heights, characters):
            font_size = int(char_height)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default().font_variant(size=font_size)
            if option == 'trOCR':
                color = (0, 170, 255)
            else:
                color = OCREngine.confidence_to_color(confidence)
            position = (bound["vertices"][0]["x"], bound["vertices"][0]["y"] - char_height)
            draw.text(position, character, fill=color, font=font)

        return black_image


    def merge_images(self, image1, image2):
        width1, height1 = image1.size
        width2, height2 = image2.size
        merged_image = Image.new("RGB", (width1 + width2, max([height1, height2])))
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (width1, 0))
        return merged_image


    def draw_boxes(self, option):
        bounds = getattr(self, f'{option}_bounds', [])
        bounds_word = getattr(self, f'{option}_bounds_word', [])
        confidences = getattr(self, f'{option}_confidences', [])

        draw = ImageDraw.Draw(self.image)
        width, height = self.image.size
        if min([width, height]) > 4000:
            line_width_thick = int((width + height) / 2 * 0.0025)  # Adjust line width for character level
            line_width_thin = 1
        else:
            line_width_thick = int((width + height) / 2 * 0.005)  # Adjust line width for character level
            line_width_thin = 1 #int((width + height) / 2 * 0.001)

        for bound in bounds_word:
            draw.polygon(
                [
                    bound["vertices"][0]["x"], bound["vertices"][0]["y"],
                    bound["vertices"][1]["x"], bound["vertices"][1]["y"],
                    bound["vertices"][2]["x"], bound["vertices"][2]["y"],
                    bound["vertices"][3]["x"], bound["vertices"][3]["y"],
                ],
                outline=OCREngine.BBOX_COLOR,
                width=line_width_thin
            )

        # Draw a line segment at the bottom of each handwritten character
        for bound, confidence in zip(bounds, confidences):  
            color = OCREngine.confidence_to_color(confidence)
            # Use the bottom two vertices of the bounding box for the line
            bottom_left = (bound["vertices"][3]["x"], bound["vertices"][3]["y"] + line_width_thick)
            bottom_right = (bound["vertices"][2]["x"], bound["vertices"][2]["y"] + line_width_thick)
            draw.line([bottom_left, bottom_right], fill=color, width=line_width_thick)

        return self.image


    def detect_text(self):
        self.ocr_method.add("Google OCR Printed")
        
        with io.open(self.path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = self.client.document_text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        bounds = []
        bounds_word = []
        text_to_box_mapping = []
        bounds_flat = []
        height_flat = []
        confidences = []
        characters = []
        organized_text = ""
        paragraph_count = 0
        
        for text in texts[1:]:
            vertices = [{"x": vertex.x, "y": vertex.y} for vertex in text.bounding_poly.vertices]
            map_dict = {
                "vertices": vertices,
                "text": text.description
            }
            text_to_box_mapping.append(map_dict)

        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                # paragraph_count += 1
                # organized_text += f'OCR_paragraph_{paragraph_count}:\n'  # Add paragraph label
                for paragraph in block.paragraphs:

                    avg_H_list = []
                    for word in paragraph.words:
                        Yw = max(vertex.y for vertex in word.bounding_box.vertices)
                        # Calculate the width of the word and divide by the number of symbols
                        word_length = max(vertex.x for vertex in word.bounding_box.vertices) - min(vertex.x for vertex in word.bounding_box.vertices)
                        num_symbols = len(word.symbols)
                        if num_symbols <= 3:
                            H = int(Yw - min(vertex.y for vertex in word.bounding_box.vertices))
                        else:
                            Yo = Yw - min(vertex.y for vertex in word.bounding_box.vertices)
                            X = word_length / num_symbols if num_symbols > 0 else 0
                            H = int(X+(Yo*0.1))
                        avg_H_list.append(H)
                    avg_H = int(mean(avg_H_list))

                    words_in_para = []
                    for word in paragraph.words:
                        # Get word-level bounding box
                        bound_word_dict = {
                            "vertices": [
                                {"x": vertex.x, "y": vertex.y} for vertex in word.bounding_box.vertices
                            ]
                        }
                        bounds_word.append(bound_word_dict)
                        
                        Y = max(vertex.y for vertex in word.bounding_box.vertices)
                        word_x_start = min(vertex.x for vertex in word.bounding_box.vertices)
                        word_x_end = max(vertex.x for vertex in word.bounding_box.vertices)
                        num_symbols = len(word.symbols)
                        symbol_width = (word_x_end - word_x_start) / num_symbols if num_symbols > 0 else 0

                        current_x_position = word_x_start

                        characters_ind = []
                        for symbol in word.symbols:
                            bound_dict = {
                                "vertices": [
                                    {"x": vertex.x, "y": vertex.y} for vertex in symbol.bounding_box.vertices
                                ]
                            }
                            bounds.append(bound_dict)

                            # Create flat bounds with adjusted x position
                            bounds_flat_dict = {
                                "vertices": [
                                    {"x": current_x_position, "y": Y}, 
                                    {"x": current_x_position + symbol_width, "y": Y}
                                ]
                            }
                            bounds_flat.append(bounds_flat_dict)
                            current_x_position += symbol_width

                            height_flat.append(avg_H)
                            confidences.append(round(symbol.confidence, 4))

                            characters_ind.append(symbol.text)
                            characters.append(symbol.text)

                        words_in_para.append(''.join(characters_ind))
                    paragraph_text = ' '.join(words_in_para)  # Join words in paragraph
                    organized_text += paragraph_text + ' ' #+ '\n' 

        # median_height = statistics.median(height_flat) if height_flat else 0
        # median_heights = [median_height] * len(characters)

        self.normal_cleaned_text = texts[0].description if texts else ''
        self.normal_organized_text = organized_text
        self.normal_bounds = bounds
        self.normal_bounds_word = bounds_word
        self.normal_text_to_box_mapping = text_to_box_mapping
        self.normal_bounds_flat = bounds_flat
        # self.normal_height = median_heights #height_flat
        self.normal_height = height_flat
        self.normal_confidences = confidences
        self.normal_characters = characters
        return self.normal_cleaned_text


    def detect_handwritten_ocr(self):
        self.ocr_method.add("Google OCR Handwritten")

        
        with open(self.path, "rb") as image_file:
            content = image_file.read()

        image = vision_beta.Image(content=content)
        image_context = vision_beta.ImageContext(language_hints=["en-t-i0-handwrit"])
        response = self.client_beta.document_text_detection(image=image, image_context=image_context)
        texts = response.text_annotations
        
        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        bounds = []
        bounds_word = []
        bounds_flat = []
        height_flat = []
        confidences = []
        characters = []
        organized_text = ""
        paragraph_count = 0
        text_to_box_mapping = []

        for text in texts[1:]:
            vertices = [{"x": vertex.x, "y": vertex.y} for vertex in text.bounding_poly.vertices]
            map_dict = {
                "vertices": vertices,
                "text": text.description
            }
            text_to_box_mapping.append(map_dict)

        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                # paragraph_count += 1
                # organized_text += f'\nOCR_paragraph_{paragraph_count}:\n'  # Add paragraph label
                for paragraph in block.paragraphs:
                    
                    avg_H_list = []
                    for word in paragraph.words:
                        Yw = max(vertex.y for vertex in word.bounding_box.vertices)
                        # Calculate the width of the word and divide by the number of symbols
                        word_length = max(vertex.x for vertex in word.bounding_box.vertices) - min(vertex.x for vertex in word.bounding_box.vertices)
                        num_symbols = len(word.symbols)
                        if num_symbols <= 3:
                            H = int(Yw - min(vertex.y for vertex in word.bounding_box.vertices))
                        else:
                            Yo = Yw - min(vertex.y for vertex in word.bounding_box.vertices)
                            X = word_length / num_symbols if num_symbols > 0 else 0
                            H = int(X+(Yo*0.1))
                        avg_H_list.append(H)
                    avg_H = int(mean(avg_H_list))

                    words_in_para = []
                    for word in paragraph.words:
                        # Get word-level bounding box
                        bound_word_dict = {
                            "vertices": [
                                {"x": vertex.x, "y": vertex.y} for vertex in word.bounding_box.vertices
                            ]
                        }
                        bounds_word.append(bound_word_dict)

                        Y = max(vertex.y for vertex in word.bounding_box.vertices)
                        word_x_start = min(vertex.x for vertex in word.bounding_box.vertices)
                        word_x_end = max(vertex.x for vertex in word.bounding_box.vertices)
                        num_symbols = len(word.symbols)
                        symbol_width = (word_x_end - word_x_start) / num_symbols if num_symbols > 0 else 0

                        current_x_position = word_x_start

                        characters_ind = []
                        for symbol in word.symbols:
                            bound_dict = {
                                "vertices": [
                                    {"x": vertex.x, "y": vertex.y} for vertex in symbol.bounding_box.vertices
                                ]
                            }
                            bounds.append(bound_dict)

                            # Create flat bounds with adjusted x position
                            bounds_flat_dict = {
                                "vertices": [
                                    {"x": current_x_position, "y": Y}, 
                                    {"x": current_x_position + symbol_width, "y": Y}
                                ]
                            }
                            bounds_flat.append(bounds_flat_dict)
                            current_x_position += symbol_width

                            height_flat.append(avg_H)
                            confidences.append(round(symbol.confidence, 4))

                            characters_ind.append(symbol.text)
                            characters.append(symbol.text)

                        words_in_para.append(''.join(characters_ind))
                    paragraph_text = ' '.join(words_in_para)  # Join words in paragraph
                    organized_text += paragraph_text + ' ' #+ '\n' 

        # median_height = statistics.median(height_flat) if height_flat else 0
        # median_heights = [median_height] * len(characters)

        self.hand_cleaned_text = response.text_annotations[0].description if response.text_annotations else ''
        self.hand_organized_text = organized_text
        self.hand_bounds = bounds
        self.hand_bounds_word = bounds_word
        self.hand_bounds_flat = bounds_flat
        self.hand_text_to_box_mapping = text_to_box_mapping
        # self.hand_height = median_heights #height_flat
        self.hand_height = height_flat
        self.hand_confidences = confidences
        self.hand_characters = characters
        return self.hand_cleaned_text

    def hyperbolic_ocr(self, ocr_option, ocr_helper, json_key, logger_message):
        self.ocr_method.add(ocr_option)

        # Update the JSON report
        if self.json_report:
            self.json_report.set_text(text_main=f'Working on {ocr_option} OCR :construction:')

        # Log usage
        self.logger.info(f"{logger_message} Usage Report")

        # Perform OCR
        results_text, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr_helper.ocr_hyperbolic(
            self.path, max_tokens=1024
        )

        self.cost += total_cost
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_in += cost_in
        self.cost_out += cost_out

        results_text_sanitized = sanitize_for_storage(results_text)

        # Store results in JSON
        self.OCR_JSON_to_file[json_key] = results_text_sanitized

        # Update the OCR string
        if self.double_OCR:
            self.OCR += f"\n{ocr_option} OCR:\n{results_text}" * 2
        else:
            self.OCR += f"\n{ocr_option} OCR:\n{results_text}"

    def gemini_ocr(self, ocr_option, ocr_helper, json_key, logger_message):
        self.ocr_method.add(ocr_option)

        # Update the JSON report
        if self.json_report:
            self.json_report.set_text(text_main=f'Working on {ocr_option} OCR :construction:')

        # Log usage
        self.logger.info(f"{logger_message} Usage Report")

        # Perform OCR
        results_text, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr_helper.ocr_gemini(self.path)

        self.cost += total_cost
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_in += cost_in
        self.cost_out += cost_out

        results_text_sanitized = sanitize_for_storage(results_text)

        # Store results in JSON
        self.OCR_JSON_to_file[json_key] = results_text_sanitized

        # Update the OCR string
        if self.double_OCR:
            self.OCR += f"\n{ocr_option} OCR:\n{results_text}" * 2
        else:
            self.OCR += f"\n{ocr_option} OCR:\n{results_text}"

    def process_image(self, do_create_OCR_helper_image, path_to_crop, logger):
        self.path = path_to_crop
        if 'hand' not in self.OCR_option and 'normal' not in self.OCR_option:
            do_create_OCR_helper_image = False
            
        # Can stack options, so solitary if statements
        self.OCR = 'OCR:\n'
        if 'CRAFT' in self.OCR_option:
            self.ocr_method.add("CRAFT trOCR")

            self.do_use_trOCR = True
            self.detect_text_craft()
            ### Optionally add trOCR to the self.OCR for additional context
            if self.double_OCR:
                part_OCR = "\CRAFT trOCR:\n" + self.detect_text_with_trOCR_using_google_bboxes(self.do_use_trOCR, logger)
                self.OCR = self.OCR + part_OCR + part_OCR
            else:
                self.OCR = self.OCR + "\\CRAFT trOCR:\n" + self.detect_text_with_trOCR_using_google_bboxes(self.do_use_trOCR, logger)

            # logger.info(f"CRAFT trOCR:\n{self.OCR}")

        if 'LLaVA' in self.OCR_option: # This option does not produce an OCR helper image
            self.ocr_method.add("LLaVA")
            
            if self.json_report:
                self.json_report.set_text(text_main=f'Working on LLaVA {self.Llava.model_path} OCR :construction:')

            image, json_output, direct_output, str_output, usage_report = self.Llava.transcribe_image(self.path, self.multimodal_prompt)
            self.logger.info(f"LLaVA Usage Report for Model {self.Llava.model_path}:\n{usage_report}")

            self.OCR_JSON_to_file['OCR_LLaVA'] = str_output

            if self.double_OCR:
                self.OCR = self.OCR + f"\nLLaVA OCR:\n{str_output}" + f"\nLLaVA OCR:\n{str_output}"
            else:
                self.OCR = self.OCR + f"\nLLaVA OCR:\n{str_output}"
            # logger.info(f"LLaVA OCR:\n{self.OCR}")

        if 'Florence-2' in self.OCR_option: # This option does not produce an OCR helper image
            self.ocr_method.add("Florence-2")

            if self.json_report:
                self.json_report.set_text(text_main=f'Working on Florence-2 [{self.Florence.model_id}] OCR :construction:')

            self.logger.info(f"Florence-2 Usage Report for Model [{self.Florence.model_id}]")
            results_text, results_text_dirty, results, usage_report = self.Florence.ocr_florence(self.path, task_prompt='<OCR>', text_input=None)
            results_text_sanitized = sanitize_for_storage(results_text)

            self.OCR_JSON_to_file['OCR_Florence'] = results_text_sanitized

            if self.double_OCR:
                self.OCR = self.OCR + f"\nFlorence-2 OCR:\n{results_text}" + f"\nFlorence-2 OCR:\n{results_text}"
            else:
                self.OCR = self.OCR + f"\nFlorence-2 OCR:\n{results_text}"

        if 'Qwen-2-VL' in self.OCR_option: # This option does not produce an OCR helper image
            self.ocr_method.add("Qwen-2-VL")

            if self.json_report:
                self.json_report.set_text(text_main=f'Working on Qwen-2-VL [{self.Qwen2VL.model_id}] OCR :construction:')

            self.logger.info(f"Qwen-2-VL Usage Report for Model [{self.Qwen2VL.model_id}]")
            results_text, usage_report = self.Qwen2VL.ocr_with_vlm(self.path, workflow_option=1)
            results_text_sanitized = sanitize_for_storage(results_text)

            self.OCR_JSON_to_file['OCR_Qwen2VL'] = results_text_sanitized

            if self.double_OCR:
                self.OCR = self.OCR + f"\nQwen2VL OCR:\n{results_text}" + f"\nQwen2VL OCR:\n{results_text}"
            else:
                self.OCR = self.OCR + f"\nQwen2VL OCR:\n{results_text}"

        if 'GPT-4o-mini' in self.OCR_option: # This option does not produce an OCR helper image
            self.ocr_method.add("GPT-4o-mini")
            if self.json_report:
                self.json_report.set_text(text_main=f'Working on GPT-4o-mini OCR :construction:')

            self.logger.info(f"GPT-4o-mini Usage Report")
            results_text, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = self.GPTmini.ocr_gpt4o(self.path, model_name="gpt-4o-mini",
                                                                                                                            resolution=self.cfg['leafmachine']['project']['OCR_GPT_4o_mini_resolution'],
                                                                                                                            max_tokens=1024)
            self.cost += total_cost
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            self.cost_in += cost_in
            self.cost_out += cost_out

            results_text_sanitized = sanitize_for_storage(results_text)

            self.OCR_JSON_to_file['OCR_GPT_4o_mini'] = results_text_sanitized

            if self.double_OCR:
                self.OCR = self.OCR + f"\nGPT-4o-mini OCR:\n{results_text}" + f"\nGPT-4o-mini OCR:\n{results_text}"
            else:
                self.OCR = self.OCR + f"\nGPT-4o-mini OCR:\n{results_text}"
        
        if 'GPT-4o' in self.OCR_option: # This option does not produce an OCR helper image
            self.ocr_method.add("GPT-4o")
            if self.json_report:
                self.json_report.set_text(text_main=f'Working on GPT-4o OCR :construction:')

            self.logger.info(f"GPT-4o Usage Report")
            results_text, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = self.GPT4o.ocr_gpt4o(self.path, model_name="gpt-4o",
                                                                                                                            resolution=self.cfg['leafmachine']['project']['OCR_GPT_4o_mini_resolution'], 
                                                                                                                            max_tokens=1024)
            self.cost += total_cost
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            self.cost_in += cost_in
            self.cost_out += cost_out

            results_text_sanitized = sanitize_for_storage(results_text)

            self.OCR_JSON_to_file['OCR_GPT_4o'] = results_text_sanitized

            if self.double_OCR:
                self.OCR = self.OCR + f"\nGPT-4o OCR:\n{results_text}" + f"\nGPT-4o OCR:\n{results_text}"
            else:
                self.OCR = self.OCR + f"\nGPT-4o OCR:\n{results_text}"

        # Gemini options
        if "Gemini-3-Pro-Preview" in self.OCR_option: # This option does not produce an OCR helper image
            self.gemini_ocr(ocr_option="Gemini-3-Pro-Preview",
                ocr_helper=self.GeminiProVision_3_Pro,
                json_key='OCR_Gemini_3_Pro_Preview',
                logger_message="Gemini-3-Pro-Preview"
            )
        elif "gemini-3-pro-preview" in self.OCR_option: # This option does not produce an OCR helper image
            self.gemini_ocr(ocr_option="gemini-3-pro-preview",
                ocr_helper=self.GeminiProVision_3_Pro,
                json_key='OCR_gemini_3_pro_preview',
                logger_message="gemini-3-pro-preview"
            )

        if "Gemini-2.5-Pro" in self.OCR_option: # This option does not produce an OCR helper image
            self.gemini_ocr(ocr_option="Gemini-2.5-Pro",
                ocr_helper=self.GeminiProVision_2_5_Pro,
                json_key='OCR_Gemini_2_5_Pro',
                logger_message="Gemini-2.5-Pro"
            )

        if "Gemini-2.5-Flash" in self.OCR_option: # This option does not produce an OCR helper image
            self.gemini_ocr(ocr_option="Gemini-2.5-Flash",
                ocr_helper=self.GeminiProVision_2_5_Flash,
                json_key='OCR_Gemini_2_5_Flash',
                logger_message="Gemini-2.5-Flash"
            )

        if "Gemini-2.0-Flash" in self.OCR_option: # This option does not produce an OCR helper image
            self.gemini_ocr(ocr_option="Gemini-2.0-Flash",
                ocr_helper=self.GeminiProVision_2_0_Flash,
                json_key='OCR_Gemini_2_0_Flash',
                logger_message="Gemini-2.0-Flash"
            )

        if "Gemini-1.5-Pro" in self.OCR_option: # This option does not produce an OCR helper image
            self.gemini_ocr(ocr_option="Gemini-1.5-Pro",
                ocr_helper=self.GeminiProVision_1_5_Pro,
                json_key='OCR_Gemini_1_5_Pro',
                logger_message="Gemini-1.5-Pro"
            )

        if "Gemini-1.5-Flash" in self.OCR_option: # This option does not produce an OCR helper image
            self.gemini_ocr(ocr_option="Gemini-1.5-Flash",
                ocr_helper=self.GeminiProVision_1_5_Flash,
                json_key='OCR_Gemini_1_5_Flash',
                logger_message="Gemini-1.5-Flash"
            )

        if "Gemini-1.5-Flash-8B" in self.OCR_option: # This option does not produce an OCR helper image
            self.gemini_ocr(ocr_option="Gemini-1.5-Flash-8B",
                ocr_helper=self.GeminiProVision_1_5_Flash_8B,
                json_key='OCR_Gemini_1_5_Flash_8B',
                logger_message="Gemini-1.5-Flash-8B"
            )

        # Process each OCR option dynamically
        if 'Qwen2-VL-7B-Instruct' in self.OCR_option: # This option does not produce an OCR helper image
            self.hyperbolic_ocr(ocr_option='Qwen2-VL-7B-Instruct',
                ocr_helper=self.Hyperbolic_Qwen2_VL_7B,
                json_key='OCR_Hyperbolic_VLM_Qwen2_VL_7B_Instruct',
                logger_message='Qwen2-VL-7B-Instruct'
            )

        if 'Qwen2-VL-72B-Instruct' in self.OCR_option: # This option does not produce an OCR helper image
            self.hyperbolic_ocr(
                ocr_option='Qwen2-VL-72B-Instruct',
                ocr_helper=self.Hyperbolic_Qwen2_VL_72B,
                json_key='OCR_Hyperbolic_VLM_Qwen2_VL_72B_Instruct',
                logger_message='Qwen2-VL-72B-Instruct'
            )

        if 'Llama-3.2-90B-Vision-Instruct' in self.OCR_option: # This option does not produce an OCR helper image
            self.hyperbolic_ocr(
                ocr_option='Llama-3.2-90B-Vision-Instruct',
                ocr_helper=self.Hyperbolic_Llama_2_2_90B_Vision,
                json_key='OCR_Hyperbolic_VLM_Llama_3_2_90B_Vision_Instruct',
                logger_message='Llama-3.2-90B-Vision-Instruct'
            )

        if 'Pixtral-12B-2409' in self.OCR_option: # This option does not produce an OCR helper image
            self.hyperbolic_ocr(
                ocr_option='Pixtral-12B-2409',
                ocr_helper=self.Hyperbolic_Pixtral_12B,
                json_key='OCR_Hyperbolic_VLM_Pixtral_12B',
                logger_message='Pixtral-12B-2409'
            )

        if 'normal' in self.OCR_option or 'hand' in self.OCR_option:
            if 'normal' in self.OCR_option:
                if self.double_OCR:
                    part_OCR = self.OCR + "\nGoogle Printed OCR:\n" + self.detect_text()
                    self.OCR = self.OCR + part_OCR + part_OCR
                else:
                    self.OCR = self.OCR + "\nGoogle Printed OCR:\n" + self.detect_text()
            if 'hand' in self.OCR_option:
                if self.double_OCR:
                    part_OCR = self.OCR + "\nGoogle Handwritten OCR:\n" + self.detect_handwritten_ocr()
                    self.OCR = self.OCR + part_OCR + part_OCR
                else:
                    self.OCR = self.OCR + "\nGoogle Handwritten OCR:\n" + self.detect_handwritten_ocr()
            # if self.OCR_option not in ['normal', 'hand', 'both']:
            #     self.OCR_option = 'both'
            #     self.detect_text()
            #     self.detect_handwritten_ocr()

            ### Optionally add trOCR to the self.OCR for additional context
            if self.do_use_trOCR:
                if self.double_OCR:
                    part_OCR = "\ntrOCR:\n" + self.detect_text_with_trOCR_using_google_bboxes(self.do_use_trOCR, logger)
                    self.OCR = self.OCR + part_OCR + part_OCR
                else:
                    self.OCR = self.OCR + "\ntrOCR:\n" + self.detect_text_with_trOCR_using_google_bboxes(self.do_use_trOCR, logger)
            # logger.info(f"OCR:\n{self.OCR}")
            else:
                # populate self.OCR_JSON_to_file = {}
                _ = self.detect_text_with_trOCR_using_google_bboxes(self.do_use_trOCR, logger)


        if do_create_OCR_helper_image and ('LLaVA' not in self.OCR_option):
            self.image = Image.open(self.path)

            if 'normal' in self.OCR_option:
                image_with_boxes_normal = self.draw_boxes('normal')
                text_image_normal = self.render_text_on_black_image('normal')
                self.merged_image_normal = self.merge_images(image_with_boxes_normal, text_image_normal)

            if 'hand' in self.OCR_option:
                image_with_boxes_hand = self.draw_boxes('hand')
                text_image_hand = self.render_text_on_black_image('hand')
                self.merged_image_hand = self.merge_images(image_with_boxes_hand, text_image_hand)

            if self.do_use_trOCR:
                text_image_trOCR = self.render_text_on_black_image('trOCR') 

            if 'CRAFT' in self.OCR_option:
                image_with_boxes_normal = self.draw_boxes('normal')
                self.merged_image_normal = self.merge_images(image_with_boxes_normal, text_image_trOCR)

            ### Merge final overlay image
            ### [original, normal bboxes, normal text]
            if 'hand' in self.OCR_option or 'normal' in self.OCR_option:
                if 'CRAFT' in self.OCR_option or 'normal' in self.OCR_option:
                    self.overlay_image = self.merge_images(Image.open(self.path), self.merged_image_normal)
                ### [original, hand bboxes, hand text]
                elif 'hand' in self.OCR_option:
                    self.overlay_image = self.merge_images(Image.open(self.path), self.merged_image_hand)
                ### [original, normal bboxes, normal text, hand bboxes, hand text]
                else:
                    self.overlay_image = self.merge_images(Image.open(self.path), self.merge_images(self.merged_image_normal, self.merged_image_hand))
                
            
            if self.do_use_trOCR:
                if 'CRAFT' in self.OCR_option:
                    heat_map_text = Image.fromarray(cv2.cvtColor(self.prediction_result["heatmaps"]["text_score_heatmap"], cv2.COLOR_BGR2RGB))
                    heat_map_link = Image.fromarray(cv2.cvtColor(self.prediction_result["heatmaps"]["link_score_heatmap"], cv2.COLOR_BGR2RGB))
                    self.overlay_image = self.merge_images(self.overlay_image, heat_map_text)
                    self.overlay_image = self.merge_images(self.overlay_image, heat_map_link)

                else:
                    self.overlay_image = self.merge_images(self.overlay_image, text_image_trOCR)

        else:
            self.merged_image_normal = None
            self.merged_image_hand = None
            self.overlay_image = Image.open(self.path)
        
        try:
            from craft_text_detector import empty_cuda_cache
            empty_cuda_cache()
        except:
            pass

class SafetyCheck():
    def __init__(self, is_hf) -> None:
        self.is_hf = is_hf
        self.set_client()

    def set_client(self):
        if self.is_hf:
            self.client = vision.ImageAnnotatorClient(credentials=self.get_google_credentials())
        else:
            self.client = vision.ImageAnnotatorClient(credentials=self.get_google_credentials())


    def get_google_credentials(self):
        creds_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json_str))
        return credentials
    
    def check_for_inappropriate_content(self, file_stream):
        LEVEL = 2
        content = file_stream.read()
        image = vision.Image(content=content)
        response = self.client.safe_search_detection(image=image)
        safe = response.safe_search_annotation

        likelihood_name = (
            "UNKNOWN",
            "VERY_UNLIKELY",
            "UNLIKELY",
            "POSSIBLE",
            "LIKELY",
            "VERY_LIKELY",
        )
        print("Safe search:")

        print(f"    adult*: {likelihood_name[safe.adult]}")
        print(f"    medical*: {likelihood_name[safe.medical]}")
        print(f"    spoofed: {likelihood_name[safe.spoof]}")
        print(f"    violence*: {likelihood_name[safe.violence]}")
        print(f"    racy: {likelihood_name[safe.racy]}")

        # Check the levels of adult, violence, racy, etc. content.
        if (safe.adult > LEVEL or
            safe.medical > LEVEL or
            # safe.spoof > LEVEL or
            safe.violence > LEVEL #or
            # safe.racy > LEVEL
            ):
            print("Found violation")
            return True  # The image violates safe search guidelines.

        print("Found NO violation")
        return False  # The image is considered safe.