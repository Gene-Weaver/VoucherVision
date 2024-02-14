import json, torch, transformers, gc
from transformers import BitsAndBytesConfig
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from huggingface_hub import hf_hub_download
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from vouchervision.utils_LLM import SystemLoadMonitor, count_tokens, save_individual_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template
from vouchervision.utils_taxonomy_WFO import validate_taxonomy_WFO
from vouchervision.utils_geolocate_HERE import validate_coordinates_here
from vouchervision.tool_wikipedia import WikipediaLinks

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

    def __init__(self, logger, model_name, JSON_dict_structure):
        self.logger = logger
        self.has_GPU = torch.cuda.is_available()
        self.monitor = SystemLoadMonitor(logger)

        self.model_name = model_name
        self.model_id = f"mistralai/{self.model_name}"
        name_parts = self.model_name.split('-')

        self.model_path = hf_hub_download(repo_id=self.model_id, repo_type="model",filename="config.json")


        self.JSON_dict_structure = JSON_dict_structure
        self.starting_temp = float(self.STARTING_TEMP)
        self.temp_increment = float(0.2)
        self.adjust_temp = self.starting_temp 

        system_prompt = "You are a helpful AI assistant who answers queries a JSON dictionary as specified by the user."
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
    #         self.local_model_pipeline = None
    #         del self.local_model
    #         del self.local_model_pipeline
    #         gc.collect()  # Explicitly invoke garbage collector
    #         torch.cuda.empty_cache()
    #     else:
    #         self.local_model_pipeline = None
    #         self.local_model = None
    #         del self.local_model_pipeline
    #         del self.local_model
    #         gc.collect()  # Explicitly invoke garbage collector


    def _set_config(self):
        # self._clear_VRAM()
        self.config = {'max_new_tokens': 1024,
                'temperature': self.starting_temp,
                'seed': 2023,
                'top_p': 1,
                'top_k': 40,
                'do_sample': True,
                'n_ctx':4096,

                # Activate 4-bit precision base model loading
                'use_4bit': True,
                # Compute dtype for 4-bit base models
                'bnb_4bit_compute_dtype': "float16",
                # Quantization type (fp4 or nf4)
                'bnb_4bit_quant_type': "nf4",
                # Activate nested quantization for 4-bit base models (double quantization)
                'use_nested_quant': False,
                }
        
        compute_dtype = getattr(torch,self.config.get('bnb_4bit_compute_dtype') )

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.get('use_4bit'),
            bnb_4bit_quant_type=self.config.get('bnb_4bit_quant_type'),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.get('use_nested_quant'),
        )

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and self.config.get('use_4bit'):
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                # print("=" * 80)
                # print("Your GPU supports bfloat16: accelerate training with bf16=True")
                # print("=" * 80)
                self.b_float_opt =  torch.bfloat16

            else:
                self.b_float_opt =  torch.float16
        self._build_model_chain_parser()
    

    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        self.json_report.set_text(text_main=f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.adjust_temp += self.temp_increment

    
    def _reset_config(self):
        self.json_report.set_text(text_main=f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.logger.info(f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.adjust_temp = self.starting_temp
    

    def _build_model_chain_parser(self):
        self.local_model_pipeline = transformers.pipeline("text-generation", 
                    model=self.model_id,
                    max_new_tokens=self.config.get('max_new_tokens'),
                    top_k=self.config.get('top_k'),
                    top_p=self.config.get('top_p'),
                    do_sample=self.config.get('do_sample'),
                    model_kwargs={"torch_dtype": self.b_float_opt, 
                                "load_in_4bit": True, 
                                "quantization_config": self.bnb_config})
        self.local_model = HuggingFacePipeline(pipeline=self.local_model_pipeline)
        # Set up the retry parser with the runnable
        self.retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.parser, llm=self.local_model, max_retries=self.MAX_RETRIES)
        # Create an llm chain with LLM and prompt
        self.chain = self.prompt | self.local_model  # LCEL


    def call_llm_local_MistralAI(self, prompt_template, json_report, paths):
        _____, ____, _, __, ___, json_file_path_wiki, txt_file_path_ind_prompt = paths
        self.json_report = json_report
        self.json_report.set_text(text_main=f'Sending request to {self.model_name}')
        self.monitor.start_monitoring_usage()
        
        nt_in = 0
        nt_out = 0

        ind = 0
        while ind < self.MAX_RETRIES:
            ind += 1
            try:
                # Dynamically set the temperature for this specific request
                model_kwargs = {"temperature": self.adjust_temp}
                
                # Invoke the chain to generate prompt text
                results = self.chain.invoke({"query": prompt_template, "model_kwargs": model_kwargs})

                # Use retry_parser to parse the response with retry logic
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
                        self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{results}')
                        self._adjust_config()
                    else:
                        self.monitor.stop_inference_timer() # Starts tool timer too

                        json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
                        output, WFO_record = validate_taxonomy_WFO(output, replace_if_success_wfo=False) ###################################### make this configurable
                        output, GEO_record = validate_coordinates_here(output, replace_if_success_geo=False) ###################################### make this configurable

                        Wiki = WikipediaLinks(json_file_path_wiki)
                        Wiki.gather_wikipedia_results(output)

                        save_individual_prompt(Wiki.sanitize(prompt_template), txt_file_path_ind_prompt)

                        self.logger.info(f"Formatted JSON:\n{json.dumps(output,indent=4)}")

                        usage_report = self.monitor.stop_monitoring_report_usage()    

                        if self.adjust_temp != self.starting_temp:            
                            self._reset_config()

                        json_report.set_text(text_main=f'LLM call successful')
                        del results
                        return output, nt_in, nt_out, WFO_record, GEO_record, usage_report

            except Exception as e:
                self.logger.error(f'{e}')
                self._adjust_config()           
                
        self.logger.info(f"Failed to extract valid JSON after [{ind}] attempts")
        self.json_report.set_text(text_main=f'Failed to extract valid JSON after [{ind}] attempts')

        usage_report = self.monitor.stop_monitoring_report_usage()                
        json_report.set_text(text_main=f'LLM call failed')

        self._reset_config()
        return None, nt_in, nt_out, None, None, usage_report

