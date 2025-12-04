import os, re, json, yaml, torch, transformers
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from langchain_classic.output_parsers.retry import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, count_tokens, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template

LORA = "phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05"
PARENT_MODEL = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"

class LocalFineTuneHandler: 
    RETRY_DELAY = 2  # Wait 2 seconds before retrying
    MAX_RETRIES = 5  # Maximum number of retries
    STARTING_TEMP = 0.001
    TOKENIZER_NAME = None
    VENDOR = 'mistral'
    MAX_GPU_MONITORING_INTERVAL = 2  # seconds

    def __init__(self, cfg, logger, model_name, JSON_dict_structure, config_vals_for_permutation=None):
        self.JSON_dict_structure_str = """{"catalogNumber": "", "scientificName": "", "genus": "", "specificEpithet": "", "scientificNameAuthorship": "", "collector": "", "recordNumber": "", "identifiedBy": "", "verbatimCollectionDate": "", "collectionDate": "", "occurrenceRemarks": "", "habitat": "", "locality": "", "country": "", "stateProvince": "", "county": "", "municipality": "", "verbatimCoordinates": "", "decimalLatitude": "", "decimalLongitude": "",  "minimumElevationInMeters": "", "maximumElevationInMeters": ""}"""

        self.cfg = cfg
        self.logger = logger
        self.print_output = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name  # This refers to the LoRA fine-tuned model
        self.starting_temp = float(self.STARTING_TEMP)
        self.adjust_temp = self.starting_temp
        self.load_in_4bit = True  # Loading in 4bit precision
        
        self.monitor = SystemLoadMonitor(logger)
        self._load_model()  # Load the model and tokenizer
        self._create_prompt()  # Set up the Alpaca-style prompt
        self._set_config()  # Set configuration options
        self._build_model_chain_parser()  # Build the pipeline for inference

    def _load_model(self):
        # Load the LoRA fine-tuned model
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,  # LoRA fine-tuned model
            load_in_4bit=self.load_in_4bit,
            low_cpu_mem_usage=True,
        ).to(self.device)
        
        # Load the tokenizer from the parent model
        self.tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL)
        self.eos_token_id = self.tokenizer.eos_token_id

    def _create_prompt(self):
        # Define the Alpaca prompt structure
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
        self.instructions_text = """Refactor the unstructured text into a valid JSON dictionary. The key names follow the Darwin Core Archive Standard. If a key lacks content, then insert an empty string. Fill in the following JSON structure as required: """
        self.instructions_json = self.JSON_dict_structure_str.replace("\n    ", " ").strip().replace("\n", " ")
        self.instructions = ''.join([self.instructions_text, self.instructions_json])

        # Create a prompt using LangChain's prompt template
        self.prompt = PromptTemplate(
            template=self.alpaca_prompt, 
            input_variables=["instructions", "OCR_text", "empty"]
        )
        self.parser = JsonOutputParser()

    def _set_config(self):
        self.config = {
            'max_new_tokens': 1024,
            'temperature': self.starting_temp,
            'seed': 2023,
            'top_p': 1,
            'do_sample': False,
            'n_ctx': 4096,
        }

    def _build_model_chain_parser(self):
        # Create text-generation pipeline for inference
        self.local_model_pipeline = transformers.pipeline(
            "text-generation", 
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.get('max_new_tokens'),
            top_k=self.config.get('top_k', None),
            top_p=self.config.get('top_p'),
            do_sample=self.config.get('do_sample'),
        )
        self.local_model = HuggingFacePipeline(pipeline=self.local_model_pipeline)

    def call_llm_local_custom_fine_tune(self, OCR_text, json_report, paths):
        json_file_path_wiki, txt_file_path_ind_prompt = paths[-2:]
        self.json_report = json_report
        if self.json_report:
            self.json_report.set_text(text_main=f'Sending request to {self.model_name}')
        
        self.monitor.start_monitoring_usage()

        self.inputs = self.tokenizer(
            [self.alpaca_prompt.format(self.instructions, OCR_text, "")], 
            return_tensors="pt"
        ).to(self.device)

        # Generate the output with the model
        outputs = self.model.generate(
            input_ids=self.inputs["input_ids"], 
            max_new_tokens=512,  # Adjust max length as needed
            eos_token_id=self.eos_token_id,
            use_cache=True
        )

        # Decode the generated output
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        ind = 0
        while ind < self.MAX_RETRIES:
            ind += 1
            try:
                # Perform inference
                results = self.local_model.invoke(OCR_text)
                output = self.retry_parser.parse_with_prompt(results, prompt_value=OCR_text)

                if output is None:
                    self.logger.error(f'Failed to extract JSON from:\n{results}')
                    self._adjust_config()
                else:
                    # Handle and validate the response
                    output = validate_and_align_JSON_keys_with_template(
                        output, 
                        json.loads(self.JSON_dict_structure_str)
                    )
                    self.logger.info(f"Formatted JSON:\n{json.dumps(output, indent=4)}")
                    return output
            except Exception as e:
                self.logger.error(f'Error during inference: {e}')

        self.logger.info(f"Failed to extract valid JSON after [{ind}] attempts")
        return None
