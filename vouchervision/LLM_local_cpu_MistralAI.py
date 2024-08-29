import os, json, gc
import time
import torch
import transformers
import random
from transformers import BitsAndBytesConfig#, AutoModelForCausalLM, AutoTokenizer
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_experimental.llms import JsonFormer
from langchain.tools import tool
# from langchain_community.llms import CTransformers
# from ctransformers import AutoModelForCausalLM, AutoConfig, Config

from langchain_community.llms import LlamaCpp
# from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from huggingface_hub import hf_hub_download


from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, count_tokens, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template

class LocalCPUMistralHandler: 
    RETRY_DELAY = 2  # Wait 2 seconds before retrying
    MAX_RETRIES = 5  # Maximum number of retries
    STARTING_TEMP = 0.1
    TOKENIZER_NAME = None
    VENDOR = 'mistral'
    SEED = 2023


    def __init__(self, cfg, logger, model_name, JSON_dict_structure, config_vals_for_permutation):
        self.cfg = cfg
        self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
        self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
        self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']

        self.logger = logger
        self.monitor = SystemLoadMonitor(logger)
        self.has_GPU = torch.cuda.is_available()
        self.JSON_dict_structure = JSON_dict_structure

        self.model_file = None
        self.model_name = model_name

        # https://medium.com/@scholarly360/mistral-7b-complete-guide-on-colab-129fa5e9a04d
        self.model_name = "Mistral-7B-Instruct-v0.2-GGUF"  #huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir /home/brlab/.cache --local-dir-use-symlinks False
        self.model_id = f"TheBloke/{self.model_name}"
        name_parts = self.model_name.split('-')

        huggingface_token = os.getenv("HUGGING_FACE_KEY")
        if not huggingface_token:
            self.logger.error("Hugging Face token is not set. Please set it using the HUGGING_FACE_KEY environment variable.")
            raise ValueError("Hugging Face token is not set.")
        
        if self.model_name == "Mistral-7B-Instruct-v0.2-GGUF":
            self.model_file = 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'
            self.model_path = hf_hub_download(repo_id=self.model_id,
                                 filename=self.model_file,
                                 repo_type="model",
                                 token=huggingface_token)
        else:
            raise f"Unsupported GGUF model name"

        # self.model_id = f"mistralai/{self.model_name}"
        self.starting_temp = float(self.STARTING_TEMP)
        self.temp_increment = float(0.2)
        self.adjust_temp = self.starting_temp 

        system_prompt = "You are a helpful AI assistant who answers queries with JSON objects and no explanations."
        template = """
            <s>[INST]{}[/INST]</s>

            [INST]{}[/INST]
            """.format(system_prompt, "{query}")

        # Create a prompt from the template so we can use it with Langchain
        self.prompt = PromptTemplate(template=template, input_variables=["query"])

        # Set up a parser
        self.parser = JsonOutputParser()

        self._set_config()


    # def _clear_VRAM(self):
    #     # Clear CUDA cache if it's being used
    #     if self.has_GPU:
    #         self.local_model = None
    #         del self.local_model
    #         gc.collect()  # Explicitly invoke garbage collector
    #         torch.cuda.empty_cache()
    #     else:
    #         self.local_model = None
    #         del self.local_model
    #         gc.collect()  # Explicitly invoke garbage collector


    def _set_config(self): 
        # self._clear_VRAM()
        self.config = {'max_new_tokens': 1024,
                'temperature': self.starting_temp,
                'seed': self.SEED,
                'top_p': 1,
                'top_k': 40,
                'n_ctx': 4096,
                'do_sample': True,
                }       
        self._build_model_chain_parser()
        

    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        if self.json_report:
            self.json_report.set_text(text_main=f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.adjust_temp += self.temp_increment
        self.config['temperature'] = self.adjust_temp      

    def _reset_config(self):
        if self.json_report:
            self.json_report.set_text(text_main=f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.logger.info(f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.adjust_temp = self.starting_temp
        self.config['temperature'] = self.starting_temp        
        

    def _build_model_chain_parser(self):
        self.local_model = LlamaCpp(
            model_path=self.model_path,
            max_tokens=self.config.get('max_new_tokens'),
            top_p=self.config.get('top_p'),
            # callback_manager=callback_manager,
            # n_gpu_layers=1,
            # n_batch=512,
            n_ctx=self.config.get('n_ctx'),
            stop=["[INST]"],
            verbose=False,
            streaming=False,
            )
        # Set up the retry parser with the runnable
        self.retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.parser, llm=self.local_model, max_retries=self.MAX_RETRIES)
        # Create an llm chain with LLM and prompt
        self.chain = self.prompt | self.local_model


    def call_llm_local_cpu_MistralAI(self, prompt_template, json_report, paths):
        _____, ____, _, __, ___, json_file_path_wiki, txt_file_path_ind_prompt = paths
        self.json_report = json_report
        if self.json_report:
            self.json_report.set_text(text_main=f'Sending request to {self.model_name}')
        self.monitor.start_monitoring_usage()

        nt_in = 0
        nt_out = 0

        ind = 0
        while ind < self.MAX_RETRIES:
            ind += 1
            try:
                ### BELOW IS BASIC MISTRAL CALL
                # mistral_prompt = f"<s>[INST] {prompt_template} [/INST]"
                # results = self.local_model(mistral_prompt, temperature = 0.7, 
                #             repetition_penalty = 1.15,
                #             max_new_tokens = 2048)
                # print(results)

                model_kwargs = {"temperature": self.adjust_temp}
                
                # Invoke the chain to generate prompt text
                results = self.chain.invoke({"query": prompt_template, "model_kwargs": model_kwargs})

                # Use retry_parser to parse the response with retry logic
                output = self.retry_parser.parse_with_prompt(results, prompt_value=prompt_template)

                if output is None:
                    self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{results}')
                    self._adjust_config()

                else:
                    nt_in = count_tokens(prompt_template, self.VENDOR, self.TOKENIZER_NAME)
                    nt_out = count_tokens(results, self.VENDOR, self.TOKENIZER_NAME)

                    output = validate_and_align_JSON_keys_with_template(output, self.JSON_dict_structure)
                    if output is None:
                        self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{results}')
                        self._adjust_config()
                    else:
                        self.monitor.stop_inference_timer() # Starts tool timer too

                        if self.json_report:            
                            self.json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
                        output_WFO, WFO_record, output_GEO, GEO_record = run_tools(output, self.tool_WFO, self.tool_GEO, self.tool_wikipedia, json_file_path_wiki)

                        save_individual_prompt(sanitize_prompt(prompt_template), txt_file_path_ind_prompt)

                        self.logger.info(f"Formatted JSON:\n{json.dumps(output,indent=4)}")

                        usage_report = self.monitor.stop_monitoring_report_usage()    

                        if self.adjust_temp != self.starting_temp:            
                            self._reset_config()

                        if self.json_report:            
                            self.json_report.set_text(text_main=f'LLM call successful')
                        return output, nt_in, nt_out, WFO_record, GEO_record, usage_report
                    
            except Exception as e:
                self.logger.error(f'{e}')
                self._adjust_config()  
        
        self.logger.info(f"Failed to extract valid JSON after [{ind}] attempts")
        if self.json_report:            
            self.json_report.set_text(text_main=f'Failed to extract valid JSON after [{ind}] attempts')

        self.monitor.stop_inference_timer() # Starts tool timer too
        usage_report = self.monitor.stop_monitoring_report_usage()                
        self._reset_config()

        if self.json_report:            
            self.json_report.set_text(text_main=f'LLM call failed')
        return None, nt_in, nt_out, None, None, usage_report


