import json, torch, transformers, gc
from transformers import BitsAndBytesConfig
from langchain_classic.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from huggingface_hub import hf_hub_download
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from utils_LLM import validate_and_align_JSON_keys_with_template, count_tokens, validate_taxonomy_WFO, validate_coordinates_here, remove_colons_and_double_apostrophes, SystemLoadMonitor

'''
https://python.langchain.com/docs/integrations/llms/huggingface_pipelines
'''

from torch.utils.data import Dataset, DataLoader
# Dataset for handling prompts
class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]
    
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


    def _clear_VRAM(self):
        # Clear CUDA cache if it's being used
        if self.has_GPU:
            self.local_model = None
            self.local_model_pipeline = None
            del self.local_model
            del self.local_model_pipeline
            gc.collect()  # Explicitly invoke garbage collector
            torch.cuda.empty_cache()
        else:
            self.local_model_pipeline = None
            self.local_model = None
            del self.local_model_pipeline
            del self.local_model
            gc.collect()  # Explicitly invoke garbage collector


    def _set_config(self):
        self._clear_VRAM()
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
        self.logger.info(f'Incrementing temperature and reloading model')
        self._clear_VRAM()
        self.adjust_temp += self.temp_increment
        self.config['temperature'] = self.adjust_temp
        self._build_model_chain_parser()
    

    def _build_model_chain_parser(self):
        self.local_model_pipeline = transformers.pipeline("text-generation", 
                        model=self.model_id,
                        max_new_tokens=self.config.get('max_new_tokens'),
                        temperature=self.config.get('temperature'),
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

    '''
    def call_llm_local_MistralAI(self, prompt_template):
        self.monitor.start_monitoring_usage()
        
        nt_in = 0
        nt_out = 0

        ind = 0
        while (ind < self.MAX_RETRIES):
            ind += 1
            # Invoke the chain to generate prompt text
            results = self.chain.invoke({"query": prompt_template})

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
                output, WFO_record = validate_taxonomy_WFO(output, replace_if_success_wfo=False) # Make configurable if needed
                output, GEO_record = validate_coordinates_here(output, replace_if_success_geo=False) # Make configurable if needed

                self.logger.info(f"Formatted JSON:\n{json.dumps(output,indent=4)}")

                self.monitor.stop_monitoring_report_usage()                

                if self.adjust_temp != self.starting_temp:
                    self._set_config()
                
                del results
                return output, nt_in, nt_out, WFO_record, GEO_record
        
        self.logger.info(f"Failed to extract valid JSON after [{ind}] attempts")

        self.monitor.stop_monitoring_report_usage()                

        self._set_config()
        return None, nt_in, nt_out, None, None
    '''
    def call_llm_local_MistralAI(self, prompts, batch_size=4):
        self.monitor.start_monitoring_usage()

        dataset = PromptDataset(prompts)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_results = []
        for batch_prompts in data_loader:
            batch_results = self._process_batch(batch_prompts)
            all_results.extend(batch_results)

        self.monitor.stop_monitoring_report_usage()

        if self.adjust_temp != self.starting_temp:
            self._set_config()

        return all_results

    def _process_batch(self, batch_prompts):
        batch_results = []
        for prompt in batch_prompts:
            output, nt_in, nt_out, WFO_record, GEO_record = self._process_single_prompt(prompt)
            if output is not None:
                batch_results.append({
                    "output": output,
                    "nt_in": nt_in,
                    "nt_out": nt_out,
                    "WFO_record": WFO_record,
                    "GEO_record": GEO_record
                })
        return batch_results

    def _process_single_prompt(self, prompt_template):
        nt_in = nt_out = 0
        ind = 0
        while ind < self.MAX_RETRIES:
            ind += 1
            results = self.chain.invoke({"query": prompt_template})
            output = self.retry_parser.parse_with_prompt(results, prompt_value=prompt_template)

            if output is None:
                self.logger.error(f'Failed to extract JSON from:\n{results}')
                self._adjust_config()
                del results
            else:
                nt_in = count_tokens(prompt_template, self.VENDOR, self.TOKENIZER_NAME)
                nt_out = count_tokens(results, self.VENDOR, self.TOKENIZER_NAME)

                output = validate_and_align_JSON_keys_with_template(output, self.JSON_dict_structure)
                output, WFO_record = validate_taxonomy_WFO(output, replace_if_success_wfo=False)
                output, GEO_record = validate_coordinates_here(output, replace_if_success_geo=False)

                self.logger.info(f"Formatted JSON:\n{json.dumps(output, indent=4)}")
                del results
                return output, nt_in, nt_out, WFO_record, GEO_record

        self.logger.info(f"Failed to extract valid JSON after [{ind}] attempts")
        return None, nt_in, nt_out, None, None