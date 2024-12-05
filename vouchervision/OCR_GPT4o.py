import os, base64, requests
from io import BytesIO
from PIL import Image
from openai import OpenAI

from general_utils import calculate_cost
from OCR_Prompt_Catalog import OCRPromptCatalog
from OCR_resize_for_VLMs import resize_image_to_min_max_pixels

# PROMPT = """Please perform OCR on this scientific image and extract the printed and handwritten text verbatim. Do not explain your answer, only return the verbatim text in this JSON dictionary format: {'printed_text': '', 'handwritten_text': ''}"""
PROMPT = """Please perform OCR on this scientific image and extract all of the words and text verbatim. Do not explain your answer, only return the verbatim text:"""

class GPT4oOCR:
    def __init__(self, api_key):
        self.api_key = api_key
        self.path_api_cost = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_cost', 'api_cost.yaml')

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def encode_image_resize(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def ocr_gpt4o(self, image_path, model_name="gpt-4o-mini", resolution="high", max_tokens=1024, do_resize=True, prompt=None,
                  temperature=0.1, top_p=None, seed=123456):
        
        # Temp is used first
        if top_p:
            temperature = None

        self.model_name = model_name
        overall_cost_in = 0
        overall_cost_out = 0
        overall_total_cost = 0
        overall_tokens_in = 0
        overall_tokens_out = 0
        overall_response = ""
        
        # Getting the base64 string
        if do_resize:
            image = Image.open(image_path)
            resized_image = resize_image_to_min_max_pixels(image, min_pixels=260000, max_pixels=1500000)

            # Encode the resized image
            base64_image = self.encode_image_resize(resized_image)
        else:
            base64_image = self.encode_image(image_path)

        

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }

        if prompt is None:
            # keys = ["default", "default_plus_minorcorrect", "default_plus_minorcorrect_idhandwriting", "default_plus_minorcorrect_excludestricken_gpt4",
            #         "handwriting_only", "species_only", "detailed_metadata"]

            # keys = ["default_plus_minorcorrect_excludestricken_gpt4",]
            # keys = ["default_plus_minorcorrect_excludestricken_gpt4", "species_only",]
            keys = ["default_plus_minorcorrect_excludestricken_idhandwriting",]
            # keys = ["default_plus_minorcorrect_excludestricken_idhandwriting", "species_only",]

            prompts = OCRPromptCatalog().get_prompts_by_keys(keys)
            for key, prompt in zip(keys, prompts):

                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                        "role": "user",
                        "content": [
                            {
                            "type": "text",
                            "text": prompt,
                            },
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": resolution,
                            }
                            }
                        ]
                        }
                    ],
                    "max_completion_tokens": max_tokens,
                    "temperature": temperature,
                    "seed": seed,
                    "top_p": top_p,
                    }

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response_json  = response.json()

                if "choices" in response_json :
                    parsed_answer = response_json["choices"][0]["message"]["content"]
                    usage_report = response_json.get('usage', {})
                    tokens_in = usage_report["prompt_tokens"]
                    tokens_out = usage_report["completion_tokens"]

                    if self.model_name == "gpt-4o-mini":
                        total_cost = calculate_cost('GPT_4o_mini_2024_07_18', self.path_api_cost, tokens_in, tokens_out)
                    elif self.model_name == "gpt-4o":
                        total_cost = calculate_cost('GPT_4o_2024_08_06', self.path_api_cost, tokens_in, tokens_out)

                    cost_in, cost_out, total_cost, rates_in, rates_out = total_cost

                    overall_cost_in += cost_in
                    overall_cost_out += cost_out
                    overall_total_cost += total_cost
                    overall_tokens_in += tokens_in
                    overall_tokens_out += tokens_out

                    # if key == "species_only":
                    #     parsed_answer = f"Based on context, determine which of these scientific names is the primary name: {parsed_answer}"

                    if len(keys) > 1:
                        overall_response += (parsed_answer + "\n\n")
                    else:
                        overall_response = parsed_answer
                else:
                    print(f"OCR failed")


        try:
            return overall_response, overall_cost_in, overall_cost_out, overall_total_cost, rates_in, rates_out, overall_tokens_in, overall_tokens_out
        except:
            return "", overall_cost_in, overall_cost_out, overall_total_cost, rates_in, rates_out, overall_tokens_in, overall_tokens_out

        

def main():
    run_sweep = True
    # img_path = '/home/brlab/Downloads/gem_2024_06_26__02-26-02/Cropped_Images/By_Class/label/1.jpg'
    # img_path = 'D:/D_Desktop/BR_1839468565_Ochnaceae_Campylospermum_reticulatum_label.jpg'
    img_path = 'C:/Users/willwe/Downloads/test_2024_12_04__13-49-56/Original_Images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg'
    
    # $env:OPENAI_API_KEY="KEY"
    API_KEY = ""

    
    # ocr = GPT4oOCR(API_KEY, model_name="gpt-4o-mini")
    ocr = GPT4oOCR(API_KEY)
    
    # parsed_answer, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr.ocr_gpt4o(img_path, resolution="low", max_tokens=1024)
    # print(f"Parsed Answer: {parsed_answer}")
    # print(f"Total Cost: {total_cost}")
    
    # parsed_answer, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr.ocr_gpt4o(img_path, resolution="high", max_tokens=1024)
    # print(f"Parsed Answer: {parsed_answer}")
    # print(f"Total Cost: {total_cost}")

    if run_sweep:
        import csv

        success_counts = {}
        fail_counts = {}

        reps = [0,]
        temps = [0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.2,]
        ps = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1]
        # reps = [0,]
        # temps = [0, ]
        # ps = [0.1,]

        for rep in reps:
            for t in temps:
                response, _, _, _, _, _, _, _ = ocr.ocr_gpt4o(img_path, model_name="gpt-4o",
                                                              resolution="high", max_tokens=1024, temperature=t, seed=123456)
                param_set = (t, 'N/A', response)  # N/A signifies no top_p used

                if "jouvea pilosa" in response.lower():
                    print("<<<< SUCCESS >>>>")
                    print(f"t {t}, p 'N/A'")
                    success_counts[param_set] = success_counts.get(param_set, 0) + 1
                else:
                    print("                     <<<< FAIL >>>>")
                    print(f"                     t {t}, p 'N/A'")
                    fail_counts[param_set] = fail_counts.get(param_set, 0) + 1

            for p in ps:
                response, _, _, _, _, _, _, _ = ocr.ocr_gpt4o(img_path, model_name="gpt-4o",
                                                              resolution="high", max_tokens=1024, top_p=p, seed=123456)
                param_set = ('N/A', p, response)  # N/A signifies no temperature used

                if "jouvea pilosa" in response.lower():
                    print("<<<< SUCCESS >>>>")
                    print(f"t 'N/A', p {p}")
                    success_counts[param_set] = success_counts.get(param_set, 0) + 1
                else:
                    print("                     <<<< FAIL >>>>")
                    print(f"                     t 'N/A', p {p}")
                    fail_counts[param_set] = fail_counts.get(param_set, 0) + 1

        # Display the results
        # print("Success counts:", success_counts)
        # print("Fail counts:", fail_counts)

        # Save to CSV
        with open('GPT4o_OCR_parameter_sweep_results_wSpecies_notGPT4version.csv', mode='w', newline='',encoding='ISO-8859-1', errors='replace') as file:
            writer = csv.writer(file)
            writer.writerow(['Temperature', 'Top P', 'Success Count', 'Fail Count', 'Response'])
            
            all_params = set(success_counts.keys()).union(set(fail_counts.keys()))
            for params in all_params:
                t, p, response = params
                success = success_counts.get(params, 0)
                fail = fail_counts.get(params, 0)
                response = response.encode('utf-8', errors='replace').decode('utf-8')
                writer.writerow([t, p, success, fail, response])

    


if __name__ == '__main__':
    main()