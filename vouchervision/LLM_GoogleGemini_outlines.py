import os
import time
import json
from pydantic import BaseModel
import outlines
from outlines.generate import json as generate_json
from outlines.models import transformers

from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, count_tokens, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template


class GoogleGeminiHandler:
    RETRY_DELAY = 10  # Wait 10 seconds before retrying
    MAX_RETRIES = 3  # Maximum number of retries
    TOKENIZER_NAME = 'gpt-4'
    VENDOR = 'google'
    STARTING_TEMP = 0.5

    def __init__(self, cfg, logger, model_name, JSON_dict_structure, config_vals_for_permutation):
        self.cfg = cfg
        self.logger = logger
        self.model_name = model_name
        self.JSON_dict_structure = JSON_dict_structure
        self.config_vals_for_permutation = config_vals_for_permutation
        self.monitor = SystemLoadMonitor(logger)

        self._set_config()
        self._initialize_model()

    def _set_config(self):
        if self.config_vals_for_permutation:
            self.starting_temp = float(self.config_vals_for_permutation.get('google').get('temperature', self.STARTING_TEMP))
            self.config = {
                "temperature": self.starting_temp,
                "max_output_tokens": self.config_vals_for_permutation.get('google').get('max_output_tokens', 1024),
                "top_p": self.config_vals_for_permutation.get('google').get('top_p', 1.0),
            }
        else:
            self.starting_temp = self.STARTING_TEMP
            self.config = {
                "temperature": self.starting_temp,
                "max_output_tokens": 1024,
                "top_p": 1.0,
            }

        self.temp_increment = 0.2
        self.adjust_temp = self.starting_temp

    def _initialize_model(self):
        # Initialize the model using Outlines
        self.model = transformers(self.model_name)

    def _adjust_temperature(self):
        self.adjust_temp += self.temp_increment
        self.logger.info(f"Adjusting temperature to {self.adjust_temp}")

    def _reset_temperature(self):
        self.adjust_temp = self.starting_temp
        self.logger.info(f"Resetting temperature to {self.starting_temp}")

    def call_llm_api_GoogleGemini(self, prompt_template, json_report, paths):
        _, _, _, _, _, json_file_path_wiki, txt_file_path_ind_prompt = paths
        self.json_report = json_report
        if self.json_report:
            self.json_report.set_text(text_main=f"Sending request to {self.model_name}")

        self.monitor.start_monitoring_usage()
        nt_in, nt_out = 0, 0

        generator = generate_json(self.model, self.JSON_dict_structure)

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Generate structured JSON
                response = generator(prompt_template)

                if response:
                    nt_in = count_tokens(prompt_template, self.VENDOR, self.TOKENIZER_NAME)
                    nt_out = count_tokens(json.dumps(response), self.VENDOR, self.TOKENIZER_NAME)

                    # Validate and align JSON keys
                    output = validate_and_align_JSON_keys_with_template(response, self.JSON_dict_structure)

                    if output:
                        self.monitor.stop_inference_timer()

                        if self.json_report:
                            self.json_report.set_text(text_main="Working on WFO, Geolocation, Links")

                        output_WFO, WFO_record, output_GEO, GEO_record = run_tools(
                            output, self.cfg['leafmachine']['project']['tool_WFO'],
                            self.cfg['leafmachine']['project']['tool_GEO'],
                            self.cfg['leafmachine']['project']['tool_wikipedia'], json_file_path_wiki
                        )

                        save_individual_prompt(sanitize_prompt(prompt_template), txt_file_path_ind_prompt)

                        self.logger.info(f"Formatted JSON:\n{json.dumps(output, indent=4)}")
                        usage_report = self.monitor.stop_monitoring_report_usage()

                        self._reset_temperature()

                        if self.json_report:
                            self.json_report.set_text(text_main="LLM call successful")

                        return output, nt_in, nt_out, WFO_record, GEO_record, usage_report

                self.logger.error(f"[Attempt {attempt}] Failed to extract JSON.")
                self._adjust_temperature()
                time.sleep(self.RETRY_DELAY)

            except Exception as e:
                self.logger.error(f"Exception during JSON generation: {e}")
                self._adjust_temperature()
                time.sleep(self.RETRY_DELAY)

        self.logger.info("Failed to extract valid JSON after maximum attempts")
        if self.json_report:
            self.json_report.set_text(text_main="Failed to extract valid JSON")
        
        self.monitor.stop_inference_timer()
        usage_report = self.monitor.stop_monitoring_report_usage()
        self._reset_temperature()

        return None, nt_in, nt_out, None, None, usage_report
