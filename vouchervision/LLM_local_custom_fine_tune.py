import os, re, json, yaml, torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

import json, torch, transformers, gc
from transformers import BitsAndBytesConfig
from langchain.output_parsers.retry import RetryOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from huggingface_hub import hf_hub_download
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, count_tokens, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template

# MODEL_NAME = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
# sltp_version = 'HLT_MICH_Angiospermae_SLTPvA_v1-0_medium__OCR-C25-L25-E50-R05'
# LORA = "phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05"

TEXT = "HERBARIUM OF MARCUS W. LYON , JR . Tracaulon sagittatum Indiana : Porter Co. Mincral Springs edge wet subdural woods 1927 TX 11 Flowers pink UNIVERSIT HERBARIUM MICHIGAN MICH University of Michigan Herbarium 1439649 copyright reserved PERSICARIA FEB 26 1965 cm "
PARENT_MODEL = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"

class LocalFineTuneHandler: 
    RETRY_DELAY = 2  # Wait 2 seconds before retrying
    MAX_RETRIES = 5  # Maximum number of retries
    STARTING_TEMP = 0.001
    TOKENIZER_NAME = None
    VENDOR = 'mistral'
    MAX_GPU_MONITORING_INTERVAL = 2  # seconds

    

    def __init__(self, cfg, logger, model_name, JSON_dict_structure, config_vals_for_permutation=None):
        # self.model_id = f"phyloforfun/{self.model_name}"
        # model_name = LORA #######################################################
        
        # self.JSON_dict_structure = JSON_dict_structure
        # self.JSON_dict_structure_str = json.dumps(self.JSON_dict_structure, sort_keys=False, indent=4)
        
        self.JSON_dict_structure_str = """{"catalogNumber": "", "scientificName": "", "genus": "", "specificEpithet": "", "scientificNameAuthorship": "", "collector": "", "recordNumber": "", "identifiedBy": "", "verbatimCollectionDate": "", "collectionDate": "", "occurrenceRemarks": "", "habitat": "", "locality": "", "country": "", "stateProvince": "", "county": "", "municipality": "", "verbatimCoordinates": "", "decimalLatitude": "", "decimalLongitude": "",  "minimumElevationInMeters": "", "maximumElevationInMeters": ""}"""


        self.cfg = cfg
        self.print_output = True
        self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
        self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
        self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']

        self.logger = logger

        self.has_GPU = torch.cuda.is_available()
        if self.has_GPU:
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.monitor = SystemLoadMonitor(logger)

        self.model_name = model_name.split("/")[1]
        self.model_id = model_name

        # self.model_path = hf_hub_download(repo_id=self.model_id, repo_type="model",filename="config.json")

        
        self.starting_temp = float(self.STARTING_TEMP)
        self.temp_increment = float(0.2)
        self.adjust_temp = self.starting_temp 

        self.load_in_4bit = False

        self.parser = JsonOutputParser()

        self._load_model()
        self._create_prompt()
        self._set_config()
        self._build_model_chain_parser()

    def _set_config(self):
        # self._clear_VRAM()
        self.config = {'max_new_tokens': 1024,
                'temperature': self.starting_temp,
                'seed': 2023,
                'top_p': 1,
                # 'top_k': 1,
                # 'top_k': 40,
                'do_sample': False,
                'n_ctx':4096,

                # Activate 4-bit precision base model loading
                # 'use_4bit': True,
                # # Compute dtype for 4-bit base models
                # 'bnb_4bit_compute_dtype': "float16",
                # # Quantization type (fp4 or nf4)
                # 'bnb_4bit_quant_type': "nf4",
                # # Activate nested quantization for 4-bit base models (double quantization)
                # 'use_nested_quant': False,
                }
        
    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        if self.json_report:
            self.json_report.set_text(text_main=f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.adjust_temp += self.temp_increment

    
    def _reset_config(self):
        if self.json_report:
            self.json_report.set_text(text_main=f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.logger.info(f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.adjust_temp = self.starting_temp


    def _load_model(self):
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_id, # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit = self.load_in_4bit,
            low_cpu_mem_usage=True,
            
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL)
        self.eos_token_id = self.tokenizer.eos_token_id


    # def _build_model_chain_parser(self):
    #     self.local_model_pipeline = transformers.pipeline("text-generation", 
    #                 model=self.model_id,
    #                 max_new_tokens=self.config.get('max_new_tokens'),
    #                 # top_k=self.config.get('top_k'),
    #                 top_p=self.config.get('top_p'),
    #                 do_sample=self.config.get('do_sample'),
    #                 model_kwargs={"load_in_4bit": self.load_in_4bit})
    #     self.local_model = HuggingFacePipeline(pipeline=self.local_model_pipeline)
    #     # Set up the retry parser with the runnable
    #     # self.retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.parser, llm=self.local_model, max_retries=self.MAX_RETRIES)
    #     self.retry_parser = RetryOutputParser(parser=self.parser, llm=self.local_model, max_retries=self.MAX_RETRIES)

    #     # Create an llm chain with LLM and prompt
    #     self.chain = self.prompt | self.local_model  # LCEL
    def _build_model_chain_parser(self):
        self.local_model_pipeline = transformers.pipeline(
            "text-generation", 
            model=self.model_id,
            max_new_tokens=self.config.get('max_new_tokens'),
            top_k=self.config.get('top_k', None),
            top_p=self.config.get('top_p'),
            do_sample=self.config.get('do_sample'),
            model_kwargs={"load_in_4bit": self.load_in_4bit},
        )
        self.local_model = HuggingFacePipeline(pipeline=self.local_model_pipeline)
        self.retry_parser = RetryOutputParser(parser=self.parser, llm=self.local_model, max_retries=self.MAX_RETRIES)



    def _create_prompt(self):
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""   
        
        self.template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}""".format("{instructions}", "{OCR_text}", "{empty}")

        self.instructions_text = """Refactor the unstructured text into a valid JSON dictionary. The key names follow the Darwin Core Archive Standard. If a key lacks content, then insert an empty string. Fill in the following JSON structure as required: """
        self.instructions_json = self.JSON_dict_structure_str.replace("\n    ", " ").strip().replace("\n", " ")
        self.instructions = ''.join([self.instructions_text, self.instructions_json])


        # Create a prompt from the template so we can use it with Langchain
        self.prompt = PromptTemplate(template=self.template, input_variables=["instructions", "OCR_text", "empty"])

        # Set up a parser
        self.parser = JsonOutputParser()
    

    def extract_json(self, response_text):
        # Assuming the response is a list with a single string entry
        # response_text = response[0]

        response_pattern = re.compile(r'### Response:(.*)', re.DOTALL)
        response_match = response_pattern.search(response_text)
        if not response_match:
            raise ValueError("No '### Response:' section found in the provided text")
        
        response_text = response_match.group(1)
        
        # Use a regular expression to find JSON objects in the response text
        json_objects = re.findall(r'\{.*?\}', response_text, re.DOTALL)
        
        if json_objects:
            # Assuming you want the first JSON object if there are multiple
            json_str = json_objects[0]
            # Convert the JSON string to a Python dictionary
            json_dict = json.loads(json_str)
            return json_str, json_dict
        else:
            raise ValueError("No JSON object found in the '### Response:' section")


    def call_llm_local_custom_fine_tune(self, OCR_text, json_report, paths):
        _____, ____, _, __, ___, json_file_path_wiki, txt_file_path_ind_prompt = paths
        self.json_report = json_report
        if self.json_report:
            self.json_report.set_text(text_main=f'Sending request to {self.model_name}')
        self.monitor.start_monitoring_usage()
        
        nt_in = 0
        nt_out = 0

        self.inputs = self.tokenizer(
        [
            self.alpaca_prompt.format(
                self.instructions, # instruction
                OCR_text, # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to(self.device)

        ind = 0
        while ind < self.MAX_RETRIES:
            ind += 1
            try:
                # Fancy
                # Dynamically set the temperature for this specific request
                model_kwargs = {"temperature": self.adjust_temp}
                
                # Invoke the chain to generate prompt text
                # results = self.chain.invoke({"instructions": self.instructions, "OCR_text": OCR_text, "empty": "", "model_kwargs": model_kwargs})

                # Use retry_parser to parse the response with retry logic
                # output = self.retry_parser.parse_with_prompt(results, prompt_value=OCR_text)
                results = self.local_model.invoke(OCR_text)
                output = self.retry_parser.parse_with_prompt(results, prompt_value=OCR_text)


                # Should work:
                # output = self.model.generate(**self.inputs, eos_token_id=self.eos_token_id, max_new_tokens=512)  # Adjust max_length as needed

                # Decode the generated text
                # generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

                # json_str, json_dict = self.extract_json(generated_text)
                if self.print_output:
                    # print("\nJSON String:")
                    # print(json_str)
                    print("\nJSON Dictionary:")
                    print(output)


                
                if output is None:
                    self.logger.error(f'Failed to extract JSON from:\n{results}')
                    self._adjust_config()
                    del results

                else:
                    nt_in = count_tokens(self.instructions+OCR_text, self.VENDOR, self.TOKENIZER_NAME)
                    nt_out = count_tokens(results, self.VENDOR, self.TOKENIZER_NAME)

                    output = validate_and_align_JSON_keys_with_template(output, json.loads(self.JSON_dict_structure_str))
                    
                    if output is None:
                        self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{results}')
                        self._adjust_config()
                    else:
                        self.monitor.stop_inference_timer() # Starts tool timer too
                        
                        if self.json_report:
                            self.json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
                        output_WFO, WFO_record, output_GEO, GEO_record = run_tools(output, self.tool_WFO, self.tool_GEO, self.tool_wikipedia, json_file_path_wiki)

                        save_individual_prompt(sanitize_prompt(self.instructions+OCR_text), txt_file_path_ind_prompt)

                        self.logger.info(f"Formatted JSON:\n{json.dumps(output,indent=4)}")

                        usage_report = self.monitor.stop_monitoring_report_usage()    

                        if self.adjust_temp != self.starting_temp:            
                            self._reset_config()

                        if self.json_report:
                            self.json_report.set_text(text_main=f'LLM call successful')
                        del results
                        return output, nt_in, nt_out, WFO_record, GEO_record, usage_report

            except Exception as e:
                self.logger.error(f'{e}')


        self.logger.info(f"Failed to extract valid JSON after [{ind}] attempts")
        if self.json_report:
            self.json_report.set_text(text_main=f'Failed to extract valid JSON after [{ind}] attempts')

        self.monitor.stop_inference_timer() # Starts tool timer too
        usage_report = self.monitor.stop_monitoring_report_usage()                
        if self.json_report:
            self.json_report.set_text(text_main=f'LLM call failed')

        return None, nt_in, nt_out, None, None, usage_report



        # # Create a prompt from the template so we can use it with Langchain
        # self.prompt = PromptTemplate(template=template, input_variables=["query"])

        # # Set up a parser
        # self.parser = JsonOutputParser()




















model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
sltp_version = 'HLT_MICH_Angiospermae_SLTPvA_v1-0_medium__OCR-C25-L25-E50-R05'
lora_name = "phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvA_v1-0_medium__OCR-C25-L25-E50-R05"

OCR_test = "HERBARIUM OF MARCUS W. LYON , JR . Tracaulon sagittatum Indiana : Porter Co. Mincral Springs edge wet subdural woods 1927 TX 11 Flowers pink UNIVERSIT HERBARIUM MICHIGAN MICH University of Michigan Herbarium 1439649 copyright reserved PERSICARIA FEB 26 1965 cm "





# model.merge_and_unload()



# Generate the output

