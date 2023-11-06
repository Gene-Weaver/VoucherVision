import openai
import os, sys, json, inspect, glob, tiktoken, shutil, yaml
import openpyxl
from openpyxl import Workbook, load_workbook
import google.generativeai as palm
from langchain.chat_models import AzureChatOpenAI

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from general_utils import get_cfg_from_full_path, num_tokens_from_string
from embeddings_db import VoucherVisionEmbedding
from OCR_google_cloud_vision import detect_text, overlay_boxes_on_image
from LLM_chatGPT_3_5 import OCR_to_dict, OCR_to_dict_16k
from LLM_PaLM import OCR_to_dict_PaLM
# from LLM_Falcon import OCR_to_dict_Falcon
from prompts import PROMPT_UMICH_skeleton_all_asia, PROMPT_OCR_Organized, PROMPT_UMICH_skeleton_all_asia_GPT4, PROMPT_OCR_Organized_GPT4, PROMPT_JSON
from prompt_catalog import PromptCatalog
'''
* For the prefix_removal, the image names have 'MICH-V-' prior to the barcode, so that is used for matching
  but removed for output.
* There is also code active to replace the LLM-predicted "Catalog Number" with the correct number since it is known.
  The LLMs to usually assign the barcode to the correct field, but it's not needed since it is already known.
        - Look for ####################### Catalog Number pre-defined
'''

'''
Prior to StructuredOutputParser:
    response = openai.ChatCompletion.create(
            model=MODEL,
            temperature = 0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant acting as a transcription expert and your job is to transcribe herbarium specimen labels based on OCR data and reformat it to meet Darwin Core Archive Standards into a Python dictionary based on certain rules."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
        )
        # print the model's response
        return response.choices[0].message['content']
'''

class VoucherVision():

    def __init__(self, cfg, logger, dir_home, path_custom_prompts, Project, Dirs):
        self.cfg = cfg
        self.logger = logger
        self.dir_home = dir_home
        self.path_custom_prompts = path_custom_prompts
        self.Project = Project
        self.Dirs = Dirs
        self.headers = None
        self.prompt_version = None

        self.set_API_keys()
        self.setup()


    def setup(self):
        self.logger.name = f'[Transcription]'
        self.logger.info(f'Setting up OCR and LLM')

        self.db_name = self.cfg['leafmachine']['project']['embeddings_database_name']
        self.path_domain_knowledge = self.cfg['leafmachine']['project']['path_to_domain_knowledge_xlsx']
        self.build_new_db = self.cfg['leafmachine']['project']['build_new_embeddings_database']

        self.continue_run_from_partial_xlsx = self.cfg['leafmachine']['project']['continue_run_from_partial_xlsx']

        self.prefix_removal = self.cfg['leafmachine']['project']['prefix_removal']
        self.suffix_removal = self.cfg['leafmachine']['project']['suffix_removal']
        self.catalog_numerical_only = self.cfg['leafmachine']['project']['catalog_numerical_only']

        self.prompt_version0 = self.cfg['leafmachine']['project']['prompt_version']
        self.use_domain_knowledge = self.cfg['leafmachine']['project']['use_domain_knowledge']

        self.catalog_name_options = ["Catalog Number", "catalog_number"]

        self.utility_headers = ["tokens_in", "tokens_out", "path_to_crop","path_to_original","path_to_content","path_to_helper",]

        self.map_prompt_versions()
        self.map_dir_labels()
        self.map_API_options()
        self.init_embeddings()
        self.init_transcription_xlsx()

        '''Logging'''
        self.logger.info(f'Transcribing dataset --- {self.dir_labels}')
        self.logger.info(f'Saving transcription batch to --- {self.path_transcription}')
        self.logger.info(f'Saving individual transcription files to --- {self.Dirs.transcription_ind}')
        self.logger.info(f'Starting transcription...')
        self.logger.info(f'     LLM MODEL --> {self.version_name}')
        self.logger.info(f'     Using Azure API --> {self.is_azure}')
        self.logger.info(f'     Model name passed to API --> {self.model_name}')
        self.logger.info(f'     API access token is found in PRIVATE_DATA.yaml --> {self.has_key}')

    def map_API_options(self):
        self.chat_version = self.cfg['leafmachine']['LLM_version']
        version_mapping = {
            'GPT 4': ('OpenAI GPT 4', False, 'GPT_4', self.has_key_openai),
            'GPT 3.5': ('OpenAI GPT 3.5', False, 'GPT_3_5', self.has_key_openai),
            'Azure GPT 3.5': ('(Azure) OpenAI GPT 3.5', True, 'Azure_GPT_3_5', self.has_key_azure_openai),
            'Azure GPT 4': ('(Azure) OpenAI GPT 4', True, 'Azure_GPT_4', self.has_key_azure_openai),
            'PaLM 2': ('Google PaLM 2', None, None, self.has_key_palm2)
        }
        if self.chat_version not in version_mapping:
            supported_LLMs = ", ".join(version_mapping.keys())
            raise Exception(f"Unsupported LLM: {self.chat_version}. Requires one of: {supported_LLMs}")

        self.version_name, self.is_azure, self.model_name, self.has_key = version_mapping[self.chat_version]

    def map_prompt_versions(self):
        self.prompt_version_map = {
            "Version 1": "prompt_v1_verbose",
            "Version 1 No Domain Knowledge": "prompt_v1_verbose_noDomainKnowledge",
            "Version 2": "prompt_v2_json_rules",
            "Version 1 PaLM 2": 'prompt_v1_palm2',
            "Version 1 PaLM 2 No Domain Knowledge": 'prompt_v1_palm2_noDomainKnowledge', 
            "Version 2 PaLM 2": 'prompt_v2_palm2',
        }
        self.prompt_version = self.prompt_version_map.get(self.prompt_version0, self.path_custom_prompts)
        self.is_predefined_prompt = self.is_in_prompt_version_map(self.prompt_version)

    def is_in_prompt_version_map(self, value):
        return value in self.prompt_version_map.values()

    def init_embeddings(self):
        if self.use_domain_knowledge:
            self.logger.info(f'*** USING DOMAIN KNOWLEDGE ***')
            self.logger.info(f'*** Initializing vector embeddings database ***')
            self.initialize_embeddings()
        else:
            self.Voucher_Vision_Embedding = None

    def map_dir_labels(self):
        if self.cfg['leafmachine']['use_RGB_label_images']:
            self.dir_labels = os.path.join(self.Dirs.save_per_annotation_class,'label')
        else:
            self.dir_labels = self.Dirs.save_original

        # Use glob to get all image paths in the directory
        self.img_paths = glob.glob(os.path.join(self.dir_labels, "*"))

    def load_rules_config(self):
        with open(self.path_custom_prompts, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return None
            
    def generate_xlsx_headers(self):
        # Extract headers from the 'Dictionary' keys in the JSON template rules
        xlsx_headers = list(self.rules_config_json['rules']["Dictionary"].keys())
        xlsx_headers = xlsx_headers + self.utility_headers
        return xlsx_headers

    def init_transcription_xlsx(self):
        self.HEADERS_v1_n22 = ["Catalog Number","Genus","Species","subspecies","variety","forma","Country","State","County","Locality Name","Min Elevation","Max Elevation","Elevation Units","Verbatim Coordinates","Datum","Cultivated","Habitat","Collectors","Collector Number","Verbatim Date","Date","End Date"] 
        self.HEADERS_v2_n26 = ["catalog_number","genus","species","subspecies","variety","forma","country","state","county","locality_name","min_elevation","max_elevation","elevation_units","verbatim_coordinates","decimal_coordinates","datum","cultivated","habitat","plant_description","collectors","collector_number","determined_by","multiple_names","verbatim_date","date","end_date"]
        self.HEADERS_v1_n22 = self.HEADERS_v1_n22 + self.utility_headers
        self.HEADERS_v2_n26 = self.HEADERS_v2_n26 + self.utility_headers
        # Initialize output file
        self.path_transcription = os.path.join(self.Dirs.transcription,"transcribed.xlsx")

        if self.prompt_version in ['prompt_v2_json_rules','prompt_v2_palm2']:
            self.headers = self.HEADERS_v2_n26
            self.headers_used = 'HEADERS_v2_n26'
        
        elif self.prompt_version in ['prompt_v1_verbose', 'prompt_v1_verbose_noDomainKnowledge','prompt_v1_palm2', 'prompt_v1_palm2_noDomainKnowledge']:
            self.headers = self.HEADERS_v1_n22
            self.headers_used = 'HEADERS_v1_n22'
        
        else:
            if not self.is_predefined_prompt:
                # Load the rules configuration
                self.rules_config_json = self.load_rules_config()
                # Generate the headers from the configuration
                self.headers = self.generate_xlsx_headers()
                # Set the headers used to the dynamically generated headers
                self.headers_used = 'CUSTOM'
            else:
                # If it's a predefined prompt, raise an exception as we don't have further instructions
                raise ValueError("Predefined prompt is not handled in this context.")

        self.create_or_load_excel_with_headers(os.path.join(self.Dirs.transcription,"transcribed.xlsx"), self.headers)


    def pick_model(self, vendor, nt):
        if vendor == 'GPT_3_5':
            if nt > 6000:
                return "gpt-3.5-turbo-16k-0613", True
            else:
                return "gpt-3.5-turbo", False
        if vendor == 'GPT_4':
            return "gpt-4", False
        if vendor == 'Azure_GPT_3_5':
            return "gpt-35-turbo", False
        if vendor == 'Azure_GPT_4':
            return "gpt-4", False
           
    def create_or_load_excel_with_headers(self, file_path, headers, show_head=False):
        output_dir_names = ['Archival_Components', 'Config_File', 'Cropped_Images', 'Logs', 'Original_Images', 'Transcription']
        self.completed_specimens = []

        # Check if the file exists and it's not None
        if self.continue_run_from_partial_xlsx is not None and os.path.isfile(self.continue_run_from_partial_xlsx):
            workbook = load_workbook(filename=self.continue_run_from_partial_xlsx)
            sheet = workbook.active
            show_head=True
            # Identify the 'path_to_crop' column
            try:
                path_to_crop_col = headers.index('path_to_crop') + 1
                path_to_original_col = headers.index('path_to_original') + 1
                path_to_content_col = headers.index('path_to_content') + 1
                path_to_helper_col = headers.index('path_to_helper') + 1
                # self.completed_specimens = list(sheet.iter_cols(min_col=path_to_crop_col, max_col=path_to_crop_col, values_only=True, min_row=2))
            except ValueError:
                print("'path_to_crop' not found in the header row.")

            
            path_to_crop = list(sheet.iter_cols(min_col=path_to_crop_col, max_col=path_to_crop_col, values_only=True, min_row=2))
            path_to_original = list(sheet.iter_cols(min_col=path_to_original_col, max_col=path_to_original_col, values_only=True, min_row=2))
            path_to_content = list(sheet.iter_cols(min_col=path_to_content_col, max_col=path_to_content_col, values_only=True, min_row=2))
            path_to_helper = list(sheet.iter_cols(min_col=path_to_helper_col, max_col=path_to_helper_col, values_only=True, min_row=2))
            others = [path_to_crop_col, path_to_original_col, path_to_content_col, path_to_helper_col]
            jsons = [path_to_content_col, path_to_helper_col]

            for cell in path_to_crop[0]:
                old_path = cell
                new_path = file_path
                for dir_name in output_dir_names:
                    if dir_name in old_path:
                        old_path_parts = old_path.split(dir_name)
                        new_path_parts = new_path.split('Transcription')
                        updated_path = new_path_parts[0] + dir_name + old_path_parts[1]
                        self.completed_specimens.append(os.path.basename(updated_path))
            print(f"{len(self.completed_specimens)} images are already completed")

            ### Copy the JSON files over
            for colu in jsons:
                cell = next(sheet.iter_rows(min_row=2, min_col=colu, max_col=colu))[0]
                old_path = cell.value
                new_path = file_path

                old_path_parts = old_path.split('Transcription')
                new_path_parts = new_path.split('Transcription')
                updated_path = new_path_parts[0] + 'Transcription' + old_path_parts[1]

                # Copy files
                old_dir = os.path.dirname(old_path)
                new_dir = os.path.dirname(updated_path)

                # Check if old_dir exists and it's a directory
                if os.path.exists(old_dir) and os.path.isdir(old_dir):
                    # Check if new_dir exists. If not, create it.
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)

                    # Iterate through all files in old_dir and copy each to new_dir
                    for filename in os.listdir(old_dir):
                        shutil.copy2(os.path.join(old_dir, filename), new_dir) # copy2 preserves metadata

            ### Update the file names
            for colu in others:
                for row in sheet.iter_rows(min_row=2, min_col=colu, max_col=colu):
                    for cell in row:
                        old_path = cell.value
                        new_path = file_path
                        for dir_name in output_dir_names:
                            if dir_name in old_path:
                                old_path_parts = old_path.split(dir_name)
                                new_path_parts = new_path.split('Transcription')
                                updated_path = new_path_parts[0] + dir_name + old_path_parts[1]
                                cell.value = updated_path
            show_head=True

                
        else:
            # Create a new workbook and select the active worksheet
            workbook = Workbook()
            sheet = workbook.active

            # Write headers in the first row
            for i, header in enumerate(headers, start=1):
                sheet.cell(row=1, column=i, value=header)
            self.completed_specimens = []
            
        # Save the workbook
        workbook.save(file_path)

        if show_head:
            print("continue_run_from_partial_xlsx:")
            for i, row in enumerate(sheet.iter_rows(values_only=True)):
                print(row)
                if i == 3:  # print the first 5 rows (0-indexed)
                    print("\n")
                    break



    def add_data_to_excel_from_response(self, path_transcription, response, filename_without_extension, path_to_crop, path_to_content, path_to_helper, nt_in, nt_out):
        wb = openpyxl.load_workbook(path_transcription)
        sheet = wb.active

        # find the next empty row
        next_row = sheet.max_row + 1

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                print(f"Failed to parse response: {response}")
                return

        # iterate over headers in the first row
        for i, header in enumerate(sheet[1], start=1):
            # check if header value is in response keys
            if (header.value in response) and (header.value not in self.catalog_name_options): ####################### Catalog Number pre-defined
                # check if the response value is a dictionary
                if isinstance(response[header.value], dict):
                    # if it is a dictionary, extract the 'value' field
                    cell_value = response[header.value].get('value', '')
                else:
                    # if it's not a dictionary, use it directly
                    cell_value = response[header.value]
                
                try:
                    # write the value to the cell
                    sheet.cell(row=next_row, column=i, value=cell_value)
                except:
                    sheet.cell(row=next_row, column=i, value=cell_value[0])

            elif header.value in self.catalog_name_options: 
                # if self.prefix_removal:
                #     filename_without_extension = filename_without_extension.replace(self.prefix_removal, "")
                # if self.suffix_removal:
                #     filename_without_extension = filename_without_extension.replace(self.suffix_removal, "")
                # if self.catalog_numerical_only:
                #     filename_without_extension = self.remove_non_numbers(filename_without_extension)
                sheet.cell(row=next_row, column=i, value=filename_without_extension)
            elif header.value == "path_to_crop":
                sheet.cell(row=next_row, column=i, value=path_to_crop)
            elif header.value == "path_to_original":
                if self.cfg['leafmachine']['use_RGB_label_images']:
                    fname = os.path.basename(path_to_crop)
                    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path_to_crop))))
                    path_to_original = os.path.join(base, 'Original_Images', fname)
                    sheet.cell(row=next_row, column=i, value=path_to_original)
                else:
                    fname = os.path.basename(path_to_crop)
                    base = os.path.dirname(os.path.dirname(path_to_crop))
                    path_to_original = os.path.join(base, 'Original_Images', fname)
                    sheet.cell(row=next_row, column=i, value=path_to_original)
            elif header.value == "path_to_content":
                sheet.cell(row=next_row, column=i, value=path_to_content)
            elif header.value == "path_to_helper":
                sheet.cell(row=next_row, column=i, value=path_to_helper)
            elif header.value == "tokens_in":
                sheet.cell(row=next_row, column=i, value=nt_in)
            elif header.value == "tokens_out":
                sheet.cell(row=next_row, column=i, value=nt_out)
        # save the workbook
        wb.save(path_transcription)


    

    def has_API_key(self, val):
        if val != '':
            return True
        else:
            return False

    def set_API_keys(self):
        self.dir_home = os.path.dirname(os.path.dirname(__file__))
        self.path_cfg_private = os.path.join(self.dir_home, 'PRIVATE_DATA.yaml')
        self.cfg_private = get_cfg_from_full_path(self.path_cfg_private)

        self.has_key_openai = self.has_API_key(self.cfg_private['openai']['OPENAI_API_KEY'])

        self.has_key_azure_openai = self.has_API_key(self.cfg_private['openai_azure']['api_version'])
        
        self.has_key_palm2 = self.has_API_key(self.cfg_private['google_palm']['google_palm_api'])

        self.has_key_google_OCR = self.has_API_key(self.cfg_private['google_cloud']['path_json_file'])

        if self.has_key_openai:
            openai.api_key = self.cfg_private['openai']['OPENAI_API_KEY']
            os.environ["OPENAI_API_KEY"] = self.cfg_private['openai']['OPENAI_API_KEY']
            

        if self.has_key_azure_openai:
            # os.environ["OPENAI_API_KEY"] = self.cfg_private['openai_azure']['openai_api_key']
            self.llm = AzureChatOpenAI(
                deployment_name='gpt-35-turbo',
                openai_api_version=self.cfg_private['openai_azure']['api_version'],
                openai_api_key=self.cfg_private['openai_azure']['openai_api_key'],
                openai_api_base=self.cfg_private['openai_azure']['openai_api_base'],
                openai_organization=self.cfg_private['openai_azure']['openai_organization'],
                openai_api_type=self.cfg_private['openai_azure']['openai_api_type']
            )

        if self.has_key_google_OCR:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.cfg_private['google_cloud']['path_json_file']

        if self.has_key_palm2:
            os.environ['PALM'] = self.cfg_private['google_palm']['google_palm_api']
            palm.configure(api_key=os.environ['PALM'])

        
    def initialize_embeddings(self):
        '''Loading embedding search       __init__(self, db_name, path_domain_knowledge, logger, build_new_db=False, model_name="hkunlp/instructor-xl", device="cuda")'''
        self.Voucher_Vision_Embedding = VoucherVisionEmbedding(self.db_name, self.path_domain_knowledge, logger=self.logger, build_new_db=self.build_new_db)

    def clean_catalog_number(self, data, filename_without_extension):
        #Cleans up the catalog number in data if it's a dict
        
        def modify_catalog_key(catalog_key, filename_without_extension, data):
            # Helper function to apply modifications on catalog number
            if catalog_key not in data:
                new_data = {catalog_key: None}
                data = {**new_data, **data}

            if self.prefix_removal:
                filename_without_extension = filename_without_extension.replace(self.prefix_removal, "")
            if self.suffix_removal:
                filename_without_extension = filename_without_extension.replace(self.suffix_removal, "")
            if self.catalog_numerical_only:
                filename_without_extension = self.remove_non_numbers(data[catalog_key])
            data[catalog_key] = filename_without_extension
            return data
        
        if isinstance(data, dict):
            if self.headers_used == 'HEADERS_v1_n22':
                return modify_catalog_key("Catalog Number", filename_without_extension, data)
            elif self.headers_used in ['HEADERS_v2_n26', 'CUSTOM']:
                return modify_catalog_key("catalog_number", filename_without_extension, data)
            else:
                raise ValueError("Invalid headers used.")
        else:
            raise TypeError("Data is not of type dict.")
        

    def write_json_to_file(self, filepath, data):
        '''Writes dictionary data to a JSON file.'''
        with open(filepath, 'w') as txt_file:
            if isinstance(data, dict):
                data = json.dumps(data, indent=4)
            txt_file.write(data)

    def create_null_json(self):
        return {}
    
    def remove_non_numbers(self, s):
        return ''.join([char for char in s if char.isdigit()])
    
    def create_null_row(self, filename_without_extension, path_to_crop, path_to_content, path_to_helper):
        json_dict = {header: '' for header in self.headers} 
        for header, value in json_dict.items():
            if header in self.catalog_name_options:
                if self.prefix_removal:
                    json_dict[header] = filename_without_extension.replace(self.prefix_removal, "")
                if self.suffix_removal:
                    json_dict[header] = filename_without_extension.replace(self.suffix_removal, "")
                if self.catalog_numerical_only:
                    json_dict[header] = self.remove_non_numbers(json_dict[header])
            elif header == "path_to_crop":
                json_dict[header] = path_to_crop
            elif header == "path_to_original":
                fname = os.path.basename(path_to_crop)
                base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path_to_crop))))
                path_to_original = os.path.join(base, 'Original_Images', fname)
                json_dict[header] = path_to_original
            elif header == "path_to_content":
                json_dict[header] = path_to_content
            elif header == "path_to_helper":
                json_dict[header] = path_to_helper
        return json_dict


    def setup_GPT(self, prompt_version, gpt):
        Catalog = PromptCatalog()
        self.logger.info(f'Length of OCR raw -- {len(self.OCR)}')

        # if prompt_version == 'prompt_v1_verbose':
        if self.is_predefined_prompt:
            if self.use_domain_knowledge:
                # Find a similar example from the domain knowledge
                domain_knowledge_example = self.Voucher_Vision_Embedding.query_db(self.OCR, 1)
                similarity= self.Voucher_Vision_Embedding.get_similarity()

                if prompt_version == 'prompt_v1_verbose':
                    prompt, n_fields, xlsx_headers = Catalog.prompt_v1_verbose(OCR=self.OCR,domain_knowledge_example=domain_knowledge_example,similarity=similarity)

            else:
                if prompt_version == 'prompt_v1_verbose_noDomainKnowledge':
                    prompt, n_fields, xlsx_headers = Catalog.prompt_v1_verbose_noDomainKnowledge(OCR=self.OCR)

                elif prompt_version ==  'prompt_v2_json_rules':
                    prompt, n_fields, xlsx_headers = Catalog.prompt_v2_json_rules(OCR=self.OCR)
        else:
            prompt, n_fields, xlsx_headers = Catalog.prompt_v2_custom(self.path_custom_prompts, OCR=self.OCR)
            


        nt = num_tokens_from_string(prompt, "cl100k_base")
        self.logger.info(f'Prompt token length --- {nt}')

        MODEL, use_long_form = self.pick_model(gpt, nt)
        self.logger.info(f'Waiting for {gpt} API call --- Using {MODEL}')

        return MODEL, prompt, use_long_form, n_fields, xlsx_headers, nt

        
    # def setup_GPT(self, opt, gpt):
    #     if opt == 'dict':
    #         # Find a similar example from the domain knowledge
    #         domain_knowledge_example = self.Voucher_Vision_Embedding.query_db(self.OCR, 1)
    #         similarity= self.Voucher_Vision_Embedding.get_similarity()

    #         self.logger.info(f'Length of OCR raw -- {len(self.OCR)}')

    #         # prompt = PROMPT_UMICH_skeleton_all_asia_GPT4(self.OCR, domain_knowledge_example, similarity)
    #         prompt, n_fields, xlsx_headers = 

    #         nt = num_tokens_from_string(prompt, "cl100k_base")
    #         self.logger.info(f'Prompt token length --- {nt}')

    #         MODEL, use_long_form = self.pick_model(gpt, nt)

    #         ### Direct GPT ###
    #         self.logger.info(f'Waiting for {MODEL} API call --- Using chatGPT --- Content')

    #         return MODEL, prompt, use_long_form
        
    #     elif opt == 'helper':
    #         prompt = PROMPT_OCR_Organized_GPT4(self.OCR)
    #         nt = num_tokens_from_string(prompt, "cl100k_base")

    #         MODEL, use_long_form = self.pick_model(gpt, nt)

    #         self.logger.info(f'Length of OCR raw -- {len(self.OCR)}')
    #         self.logger.info(f'Prompt token length --- {nt}')
    #         self.logger.info(f'Waiting for {MODEL} API call --- Using chatGPT --- Helper')

    #         return MODEL, prompt, use_long_form


    def use_chatGPT(self, is_azure, progress_report, gpt):
        total_tokens_in = 0
        total_tokens_out = 0
        final_JSON_response = None
        if progress_report is not None:
            progress_report.set_n_batches(len(self.img_paths))
        for i, path_to_crop in enumerate(self.img_paths):
            if progress_report is not None:
                progress_report.update_batch(f"Working on image {i+1} of {len(self.img_paths)}")

            if os.path.basename(path_to_crop) in self.completed_specimens:
                self.logger.info(f'[Skipping] specimen {os.path.basename(path_to_crop)} already processed')
            else:
                filename_without_extension, txt_file_path, txt_file_path_OCR, txt_file_path_OCR_bounds, jpg_file_path_OCR_helper = self.generate_paths(path_to_crop, i)

                # Use Google Vision API to get OCR
                # self.OCR = detect_text(path_to_crop) 
                self.logger.info(f'Working on {i+1}/{len(self.img_paths)} --- Starting OCR')
                self.OCR, self.bounds, self.text_to_box_mapping = detect_text(path_to_crop)
                self.logger.info(f'Working on {i+1}/{len(self.img_paths)} --- Finished OCR')
                if len(self.OCR) > 0:
                    self.logger.info(f'Working on {i+1}/{len(self.img_paths)} --- Creating OCR Overlay Image')
                    self.overlay_image = overlay_boxes_on_image(path_to_crop, self.bounds)
                    self.logger.info(f'Working on {i+1}/{len(self.img_paths)} --- Saved OCR Overlay Image')
                    
                    self.write_json_to_file(txt_file_path_OCR, {"OCR":self.OCR})
                    self.write_json_to_file(txt_file_path_OCR_bounds, {"OCR_Bounds":self.bounds})
                    self.overlay_image.save(jpg_file_path_OCR_helper)

                    # Setup Dict
                    MODEL, prompt, use_long_form, n_fields, xlsx_headers, nt_in = self.setup_GPT(self.prompt_version, gpt)

                    if is_azure:
                        self.llm.deployment_name = MODEL
                    else:
                        self.llm = None

                    # Send OCR to chatGPT and return formatted dictonary
                    if use_long_form:
                        response_candidate = OCR_to_dict_16k(is_azure, self.logger, MODEL, prompt, self.llm, self.prompt_version) 
                        nt_out = num_tokens_from_string(response_candidate, "cl100k_base")
                    else:
                        response_candidate = OCR_to_dict(is_azure, self.logger, MODEL, prompt, self.llm, self.prompt_version)
                        nt_out = num_tokens_from_string(response_candidate, "cl100k_base")
                else: 
                    response_candidate = None
                    nt_out = 0

                total_tokens_in += nt_in
                total_tokens_out += nt_out

                final_JSON_response0 = self.save_json_and_xlsx(response_candidate, filename_without_extension, path_to_crop, txt_file_path, jpg_file_path_OCR_helper, nt_in, nt_out)
                if response_candidate is not None:
                    final_JSON_response = final_JSON_response0

                self.logger.info(f'Formatted JSON\n{final_JSON_response}')
                self.logger.info(f'Finished {MODEL} API calls\n')
        
        if progress_report is not None:
            progress_report.reset_batch(f"Batch Complete")
        try:
            final_JSON_response = json.loads(final_JSON_response.strip('```').replace('json\n', '', 1).replace('json', '', 1))
        except:
            pass
        return final_JSON_response, total_tokens_in, total_tokens_out

                    

    def use_PaLM(self, progress_report):
        total_tokens_in = 0
        total_tokens_out = 0
        final_JSON_response = None
        if progress_report is not None:
            progress_report.set_n_batches(len(self.img_paths))
        for i, path_to_crop in enumerate(self.img_paths):
            if progress_report is not None:
                progress_report.update_batch(f"Working on image {i+1} of {len(self.img_paths)}")
            if os.path.basename(path_to_crop) in self.completed_specimens:
                self.logger.info(f'[Skipping] specimen {os.path.basename(path_to_crop)} already processed')
            else:
                filename_without_extension, txt_file_path, txt_file_path_OCR, txt_file_path_OCR_bounds, jpg_file_path_OCR_helper = self.generate_paths(path_to_crop, i)
                
                # Use Google Vision API to get OCR
                self.OCR, self.bounds, self.text_to_box_mapping = detect_text(path_to_crop)
                if len(self.OCR) > 0:
                    self.logger.info(f'Working on {i+1}/{len(self.img_paths)} --- Starting OCR')
                    self.OCR = self.OCR.replace("\'", "Minutes").replace('\"', "Seconds")
                    self.logger.info(f'Working on {i+1}/{len(self.img_paths)} --- Finished OCR')

                    self.logger.info(f'Working on {i+1}/{len(self.img_paths)} --- Creating OCR Overlay Image')
                    self.overlay_image = overlay_boxes_on_image(path_to_crop, self.bounds)
                    self.logger.info(f'Working on {i+1}/{len(self.img_paths)} --- Saved OCR Overlay Image')
                    
                    self.write_json_to_file(txt_file_path_OCR, {"OCR":self.OCR})
                    self.write_json_to_file(txt_file_path_OCR_bounds, {"OCR_Bounds":self.bounds})
                    self.overlay_image.save(jpg_file_path_OCR_helper)

                    # Send OCR to chatGPT and return formatted dictonary
                    response_candidate, nt_in = OCR_to_dict_PaLM(self.logger, self.OCR, self.prompt_version, self.Voucher_Vision_Embedding)
                    nt_out = num_tokens_from_string(response_candidate, "cl100k_base")
                    
                else:
                    response_candidate = None
                    nt_out = 0

                total_tokens_in += nt_in
                total_tokens_out += nt_out

                final_JSON_response0 = self.save_json_and_xlsx(response_candidate, filename_without_extension, path_to_crop, txt_file_path, jpg_file_path_OCR_helper, nt_in, nt_out)
                if response_candidate is not None:
                    final_JSON_response = final_JSON_response0
                self.logger.info(f'Formatted JSON\n{final_JSON_response}')
                self.logger.info(f'Finished PaLM 2 API calls\n')

        if progress_report is not None:
            progress_report.reset_batch(f"Batch Complete")
        return final_JSON_response, total_tokens_in, total_tokens_out


    '''
    def use_falcon(self, progress_report):
        for i, path_to_crop in enumerate(self.img_paths):
            progress_report.update_batch(f"Working on image {i+1} of {len(self.img_paths)}")
            if os.path.basename(path_to_crop) in self.completed_specimens:
                self.logger.info(f'[Skipping] specimen {os.path.basename(path_to_crop)} already processed')
            else:
                filename_without_extension = os.path.splitext(os.path.basename(path_to_crop))[0]
                txt_file_path = os.path.join(self.Dirs.transcription_ind, filename_without_extension + '.json')
                txt_file_path_helper = os.path.join(self.Dirs.transcription_ind_helper, filename_without_extension + '.json')
                self.logger.info(f'Working on {i+1}/{len(self.img_paths)} --- {filename_without_extension}')

                # Use Google Vision API to get OCR
                self.OCR, self.bounds, self.text_to_box_mapping = detect_text(path_to_crop)
                if len(self.OCR) > 0:
                    self.OCR = self.OCR.replace("\'", "Minutes").replace('\"', "Seconds")

                    # Send OCR to Falcon and return formatted dictionary
                    response = OCR_to_dict_Falcon(self.logger, self.OCR, self.Voucher_Vision_Embedding)
                    # response_helper = OCR_to_helper_Falcon(self.logger, OCR) # Assuming you have a similar helper function for Falcon
                    response_helper = None
                    
                    self.logger.info(f'Finished Falcon API calls\n')
                else:
                    response = None

                if (response is not None) and (response_helper is not None):
                    # Save transcriptions to json files
                    self.write_json_to_file(txt_file_path, response)
                    # self.write_json_to_file(txt_file_path_helper, response_helper)

                    # add to the xlsx file
                    self.add_data_to_excel_from_response(self.path_transcription, response, filename_without_extension, path_to_crop, txt_file_path, txt_file_path_helper)
        progress_report.reset_batch()
    '''

    def generate_paths(self, path_to_crop, i):
        filename_without_extension = os.path.splitext(os.path.basename(path_to_crop))[0]
        txt_file_path = os.path.join(self.Dirs.transcription_ind, filename_without_extension + '.json')
        txt_file_path_OCR = os.path.join(self.Dirs.transcription_ind_OCR, filename_without_extension + '.json')
        txt_file_path_OCR_bounds = os.path.join(self.Dirs.transcription_ind_OCR_bounds, filename_without_extension + '.json')
        jpg_file_path_OCR_helper = os.path.join(self.Dirs.transcription_ind_OCR_helper, filename_without_extension + '.jpg')

        self.logger.info(f'Working on {i+1}/{len(self.img_paths)} --- {filename_without_extension}')

        return filename_without_extension, txt_file_path, txt_file_path_OCR, txt_file_path_OCR_bounds, jpg_file_path_OCR_helper

    def save_json_and_xlsx(self, response, filename_without_extension, path_to_crop, txt_file_path, jpg_file_path_OCR_helper, nt_in, nt_out):
        if response is None:
            response = self.create_null_json()
            self.write_json_to_file(txt_file_path, response)

            # Then add the null info to the spreadsheet
            response_null = self.create_null_row(filename_without_extension, path_to_crop, txt_file_path, jpg_file_path_OCR_helper)
            self.add_data_to_excel_from_response(self.path_transcription, response_null, filename_without_extension, path_to_crop, txt_file_path, jpg_file_path_OCR_helper, nt_in=0, nt_out=0)
        
        ### Set completed JSON
        else:
            response = self.clean_catalog_number(response, filename_without_extension)
            self.write_json_to_file(txt_file_path, response)
            # add to the xlsx file
            self.add_data_to_excel_from_response(self.path_transcription, response, filename_without_extension, path_to_crop, txt_file_path, jpg_file_path_OCR_helper, nt_in, nt_out)
        return response
    
    def process_specimen_batch(self, progress_report):
        try:
            if self.has_key:
                if self.model_name:
                    final_json_response, total_tokens_in, total_tokens_out = self.use_chatGPT(self.is_azure, progress_report, self.model_name)
                else:
                    final_json_response, total_tokens_in, total_tokens_out = self.use_PaLM(progress_report)
                return final_json_response, total_tokens_in, total_tokens_out
            else:
                self.logger.info(f'No API key found for {self.version_name}')
                raise Exception(f"No API key found for {self.version_name}")
        except:
            if progress_report is not None:
                progress_report.reset_batch(f"Batch Failed")
            self.logger.error("LLM call failed. Ending batch. process_specimen_batch()")
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
            raise

    def process_specimen_batch_OCR_test(self, path_to_crop):
        for img_filename in os.listdir(path_to_crop):
            img_path = os.path.join(path_to_crop, img_filename)
        self.OCR, self.bounds, self.text_to_box_mapping = detect_text(img_path)



def space_saver(cfg, Dirs, logger):
    dir_out = cfg['leafmachine']['project']['dir_output']
    run_name = Dirs.run_name

    path_project = os.path.join(dir_out, run_name)

    if cfg['leafmachine']['project']['delete_temps_keep_VVE']:
        logger.name = '[DELETE TEMP FILES]'
        logger.info("Deleting temporary files. Keeping files required for VoucherVisionEditor.")
        delete_dirs = ['Archival_Components', 'Config_File']
        for d in delete_dirs:
            path_delete = os.path.join(path_project, d)
            if os.path.exists(path_delete):
                shutil.rmtree(path_delete)

    elif cfg['leafmachine']['project']['delete_all_temps']:
        logger.name = '[DELETE TEMP FILES]'
        logger.info("Deleting ALL temporary files!")
        delete_dirs = ['Archival_Components', 'Config_File', 'Original_Images', 'Cropped_Images']
        for d in delete_dirs:
            path_delete = os.path.join(path_project, d)
            if os.path.exists(path_delete):
                shutil.rmtree(path_delete)

        # Delete the transctiption folder, but keep the xlsx
        transcription_path = os.path.join(path_project, 'Transcription')
        if os.path.exists(transcription_path):
            for item in os.listdir(transcription_path):
                item_path = os.path.join(transcription_path, item)
                if os.path.isdir(item_path):  # if the item is a directory
                    if os.path.exists(item_path):
                        shutil.rmtree(item_path)  # delete the directory
