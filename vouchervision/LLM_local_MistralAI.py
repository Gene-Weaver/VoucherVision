import json, os
import torch
import transformers
import gc
from transformers import BitsAndBytesConfig
from langchain_classic.output_parsers.retry import RetryOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFacePipeline
from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, count_tokens, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template

'''
Local Pipielines:
https://python.langchain.com/docs/integrations/llms/huggingface_pipelines
'''

class LocalMistralHandler:
    RETRY_DELAY = 2  # Wait 2 seconds before retrying
    MAX_RETRIES = 5  # Maximum number of retries
    STARTING_TEMP = 0.1
    TOKENIZER_NAME = None
    VENDOR = 'mistral'
    MAX_GPU_MONITORING_INTERVAL = 2  # seconds

    def __init__(self, cfg, logger, model_name, JSON_dict_structure, config_vals_for_permutation):
        self.cfg = cfg
        self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
        self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
        self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']
        
        self.logger = logger
        self.has_GPU = torch.cuda.is_available()
        self.monitor = SystemLoadMonitor(logger)

        self.model_name = model_name
        self.model_id = f"mistralai/{self.model_name}"
        huggingface_token = os.getenv("HUGGING_FACE_KEY")
        if not huggingface_token:
            self.logger.error("Hugging Face token is not set. Please set it using the HUGGING_FACE_KEY environment variable.")
            raise ValueError("Hugging Face token is not set.")
        
        self.model_path = hf_hub_download(repo_id=self.model_id, repo_type="model", filename="config.json", token=huggingface_token)

        self.JSON_dict_structure = JSON_dict_structure
        self.starting_temp = float(self.STARTING_TEMP)
        self.temp_increment = float(0.2)
        self.adjust_temp = self.starting_temp

        system_prompt = "You are a helpful AI assistant who answers queries by returning a JSON dictionary as specified by the user."
        template = "<s>[INST]{}[/INST]</s>[INST]{}[/INST]".format(system_prompt, "{query}")

        # Create a prompt from the template so we can use it with Langchain
        self.prompt = PromptTemplate(template=template, input_variables=["query"])

        # Set up a parser
        self.parser = JsonOutputParser()

        self._set_config()

    def _set_config(self):
        self.config = {
            'max_new_tokens': 1024,
            'temperature': self.starting_temp,
            'seed': 2023,
            'top_p': 1,
            'top_k': 40,
            'do_sample': True,
            'n_ctx': 4096,
            'use_4bit': True,
            'bnb_4bit_compute_dtype': "float16",
            'bnb_4bit_quant_type': "nf4",
            'use_nested_quant': False,
        }

        compute_dtype = getattr(torch, self.config.get('bnb_4bit_compute_dtype'))

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.get('use_4bit'),
            bnb_4bit_quant_type=self.config.get('bnb_4bit_quant_type'),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.get('use_nested_quant'),
        )

        if compute_dtype == torch.float16 and self.config.get('use_4bit'):
            major, _ = torch.cuda.get_device_capability()
            self.b_float_opt = torch.bfloat16 if major >= 8 else torch.float16

        self._build_model_chain_parser()

    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.adjust_temp += self.temp_increment

    def _reset_config(self):
        self.logger.info(f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.adjust_temp = self.starting_temp

    def _build_model_chain_parser(self):
        self.local_model_pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            max_new_tokens=self.config.get('max_new_tokens'),
            top_k=self.config.get('top_k'),
            top_p=self.config.get('top_p'),
            do_sample=self.config.get('do_sample'),
            model_kwargs={"torch_dtype": self.b_float_opt, "quantization_config": self.bnb_config},
        )
        self.local_model = HuggingFacePipeline(pipeline=self.local_model_pipeline)

        # Set up the retry parser with the runnable
        self.retry_parser = RetryOutputParser(parser=self.parser, llm=self.local_model, max_retries=self.MAX_RETRIES)
        # Create an llm chain with LLM and prompt
        self.chain = self.prompt | self.local_model

    def call_llm_local_MistralAI(self, prompt_template, json_report, paths):
        json_file_path_wiki, txt_file_path_ind_prompt = paths[-2:]
        self.json_report = json_report
        if self.json_report:
            self.json_report.set_text(text_main=f'Sending request to {self.model_name}')
        self.monitor.start_monitoring_usage()

        nt_in = 0
        nt_out = 0

        for ind in range(self.MAX_RETRIES):
            try:
                model_kwargs = {"temperature": self.adjust_temp}
                results = self.chain.invoke({"query": prompt_template, "model_kwargs": model_kwargs})

                output = self.retry_parser.parse_with_prompt(results, prompt_value=prompt_template)

                if output is None:
                    self.logger.error(f'Failed to extract JSON from:\n{results}')
                    self._adjust_config()
                    del results
                else:
                    nt_in = count_tokens(prompt_template, self.VENDOR, self.TOKENIZER_NAME)
                    nt_out = count_tokens(results, self.VENDOR, self.TOKENIZER_NAME)

                    output = validate_and_align_JSON_keys_with_template(output, self.JSON_dict_structure)

                    if output is None:
                        self.logger.error(f'[Attempt {ind + 1}] Failed to extract JSON from:\n{results}')
                        self._adjust_config()
                    else:
                        self.monitor.stop_inference_timer()  # Starts tool timer too

                        if self.json_report:
                            self.json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
                        output_WFO, WFO_record, output_GEO, GEO_record = run_tools(
                            output, self.tool_WFO, self.tool_GEO, self.tool_wikipedia, json_file_path_wiki
                        )

                        save_individual_prompt(sanitize_prompt(prompt_template), txt_file_path_ind_prompt)

                        self.logger.info(f"Formatted JSON Pre-Sanitize:\n{json.dumps(output, indent=4)}")

                        usage_report = self.monitor.stop_monitoring_report_usage()

                        if self.adjust_temp != self.starting_temp:
                            self._reset_config()

                        if self.json_report:
                            self.json_report.set_text(text_main=f'LLM call successful')
                        del results
                        return output, nt_in, nt_out, WFO_record, GEO_record, usage_report
            except Exception as e:
                self.logger.error(f'{e}')
                self._adjust_config()

        self.logger.info(f"Failed to extract valid JSON after [{self.MAX_RETRIES}] attempts")
        if self.json_report:
            self.json_report.set_text(text_main=f'Failed to extract valid JSON after [{self.MAX_RETRIES}] attempts')

        self.monitor.stop_inference_timer()  # Starts tool timer too
        usage_report = self.monitor.stop_monitoring_report_usage()
        if self.json_report:
            self.json_report.set_text(text_main=f'LLM call failed')

        self._reset_config()
        return None, nt_in, nt_out, None, None, usage_report