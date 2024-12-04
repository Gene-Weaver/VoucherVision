import os, io, openai, vertexai, json, tempfile, requests
import webbrowser
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import vision
from google.cloud import vision_v1p3beta1 as vision_beta
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAI
from huggingface_hub import HfApi, HfFolder

from datetime import datetime
# import google.generativeai as genai
from google.oauth2 import service_account
# from googleapiclient.discovery import build




class APIvalidation:

    def __init__(self, cfg_private, dir_home, is_hf) -> None:
        self.cfg_private = cfg_private
        self.dir_home = dir_home
        self.is_hf = is_hf
        self.formatted_date = self.get_formatted_date()

        self.HF_MODEL_LIST = ['mistralai/Mistral-Small-Instruct-2409','mistralai/Mistral-7B-Instruct-v0.3','mistralai/Mixtral-8x22B-v0.1','mistralai/Mixtral-8x22B-Instruct-v0.1',
                              'meta-llama/Llama-3.2-1B','meta-llama/Llama-3.2-3B','meta-llama/Llama-3.2-11B-Vision-Instruct',
                            'microsoft/Florence-2-large','microsoft/Florence-2-base',
                            'microsoft/trocr-base-handwritten','microsoft/trocr-large-handwritten',
                            'google/gemma-2-9b','google/gemma-2-9b-it','google/gemma-2-27b','google/gemma-2-27b-it',
                            'unsloth/mistral-7b-instruct-v0.3-bnb-4bit'
            ]

    def get_formatted_date(self):
        # Get the current date
        current_date = datetime.now()

        # Format the date as "Month day, year" (e.g., "January 23, 2024")
        formatted_date = current_date.strftime("%B %d, %Y")

        return formatted_date


    def has_API_key(self, val):
        return isinstance(val, str) and bool(val.strip())
        # if val:
        #     return True
        # else:
        #     return False
            
    def check_openai_api_key(self):
        if self.is_hf:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            openai.api_key = self.cfg_private['openai']['OPENAI_API_KEY']

        try:
            openai.models.list()
            return True
        except:
            return False
        
    def check_azure_openai_api_key(self):
        if not self.is_hf:
            try:
                # Initialize the Azure OpenAI client
                model = AzureChatOpenAI(
                    deployment_name = 'gpt-4',#'gpt-35-turbo',
                    openai_api_version = self.cfg_private['openai_azure']['OPENAI_API_VERSION'],
                    openai_api_key = self.cfg_private['openai_azure']['OPENAI_API_KEY_AZURE'],
                    azure_endpoint = self.cfg_private['openai_azure']['OPENAI_API_BASE'],
                    openai_organization = self.cfg_private['openai_azure']['OPENAI_ORGANIZATION'],
                )
                msg = HumanMessage(content="hello")
                # self.llm_object.temperature = self.config.get('temperature')
                response = model.invoke([msg])

                # Check the response content (you might need to adjust this depending on how your AzureChatOpenAI class handles responses)
                if response:
                    return True
                else:
                    return False

            except Exception as e:  # Use a more specific exception if possible
                return False
        else:
            try:
                azure_api_version = os.getenv('AZURE_API_VERSION')
                azure_api_key = os.getenv('AZURE_API_KEY')
                azure_api_base = os.getenv('AZURE_API_BASE')
                azure_organization = os.getenv('AZURE_ORGANIZATION')
                # Initialize the Azure OpenAI client
                model = AzureChatOpenAI(
                    deployment_name = 'gpt-4',#'gpt-35-turbo',
                    openai_api_version = azure_api_version,
                    openai_api_key = azure_api_key,
                    azure_endpoint = azure_api_base,
                    openai_organization = azure_organization,
                )
                msg = HumanMessage(content="hello")
                # self.llm_object.temperature = self.config.get('temperature')
                response = model.invoke([msg])

                # Check the response content (you might need to adjust this depending on how your AzureChatOpenAI class handles responses)
                if response:
                    return True
                else:
                    return False

            except Exception as e:  # Use a more specific exception if possible
                return False
        
    def check_hyperbolic_api_key(self):
        try:
            import requests

            if not self.is_hf:
                api_key=self.cfg_private['hyperbolic']['HYPERBOLIC_API_KEY']
            else:
                api_key=os.getenv('HYPERBOLIC_API_KEY')

            
            url = "https://api.hyperbolic.xyz/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": "hello"
                    }
                ],
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "max_tokens": 1,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            response = requests.post(url, headers=headers, json=data)
            response_json = response.json()

            if "choices" in response_json:
                print(f"hyperbolic meta-llama/Meta-Llama-3.1-8B-Instruct: {response_json['choices'][0]['message']['content']}")
                return True
            else:
                print("hyperbolic failed to return expected response")
                return False

        except Exception as e:  # Replace with a more specific exception if possible
            return False
        
    def check_mistral_api_key(self):
        try:
            if not self.is_hf:
                client = MistralClient(api_key=self.cfg_private['mistral']['MISTRAL_API_KEY'])
            else:
                client = MistralClient(api_key=os.getenv('MISTRAL_API_KEY'))

            
            # Initialize the Mistral Client with the API key

            # Create a simple message
            messages = [ChatMessage(role="user", content="hello")]

            # Send the message and get the response
            chat_response = client.chat(
                model="mistral-tiny",  
                messages=messages,
            )

            # Check if the response is valid (adjust this according to the actual response structure)
            if chat_response and chat_response.choices:
                return True
            else:
                return False
        except Exception as e:  # Replace with a more specific exception if possible
            return False
        
    def check_google_vision_client(self):
        results = {"ocr_print": False, "ocr_hand": False}

        if self.is_hf:
            client_beta = vision_beta.ImageAnnotatorClient(credentials=self.get_google_credentials())
            client = vision.ImageAnnotatorClient(credentials=self.get_google_credentials())
        else:
            client_beta = vision_beta.ImageAnnotatorClient(credentials=self.get_google_credentials()) 
            client = vision.ImageAnnotatorClient(credentials=self.get_google_credentials())
        
        try:
            with open(os.path.join(self.dir_home,'demo', 'ocr_test', 'ocr_test.jpg'), "rb") as image_file:
                content = image_file.read()
        except:
            with open("./demo/ocr_test/ocr_test.jpg", "rb") as image_file:
                content = image_file.read()

        try:
            image = vision_beta.Image(content=content)
            image_context = vision_beta.ImageContext(language_hints=["en-t-i0-handwrit"])
            response = client_beta.document_text_detection(image=image, image_context=image_context)
            texts = response.text_annotations
            
            print(f"OCR Hand:\n{texts[0].description}")
            if len(texts[0].description) > 0:
                results['ocr_hand'] = True
        except:
            pass

        try:
            image = vision.Image(content=content)
            response = client.document_text_detection(image=image)
            texts = response.text_annotations
        
            print(f"OCR Print:\n{texts[0].description}")
            if len(texts[0].description) > 0:
                results['ocr_print'] = True
        except:
            pass

        return results


    def check_google_vertex_genai_api_key(self):
        results = {"palm2": False, "gemini": False, "palm2_langchain": False}


        try:
            model = TextGenerationModel.from_pretrained("text-bison@001")
            response = model.predict("Hello")
            test_response_palm = response.text
            if test_response_palm:
                results["palm2"] = True
                print(f"palm2 pass [{test_response_palm}]")
            else:
                print(f"palm2 fail [{test_response_palm}]")

        except Exception as e:
            # print(f"palm2 fail2 [{e}]")
            print(f"palm2 fail2")

        try:
            model = VertexAI(model="text-bison@001", max_output_tokens=10)
            response = model.predict("Hello")
            test_response_palm2 = response
            if test_response_palm2:
                results["palm2_langchain"] = True
                print(f"palm2_langchain pass [{test_response_palm2}]")
            else:
                print(f"palm2_langchain fail [{test_response_palm2}]")

        except Exception as e:
            print(f"palm2 fail2 [{e}]")
            print(f"palm2_langchain fail2")
            

        try:
            model = GenerativeModel("gemini-pro")
            response = model.generate_content("Hello")
            test_response_gemini = response.text
            if test_response_gemini:
                results["gemini"] = True
                print(f"gemini pass [{test_response_gemini}]")
            else:
                print(f"gemini fail [{test_response_gemini}]")

        except Exception as e:
            # print(f"palm2 fail2 [{e}]")
            print(f"palm2 fail2")

        return results

    def test_hf_token(self, k_huggingface):
        if not k_huggingface:
            print("Hugging Face API token not found in environment variables.")
            return False

        # Create an instance of the API
        api = HfApi()

        try:
            # Try to get details of a known public model
            model_info = api.model_info("bert-base-uncased", use_auth_token=k_huggingface)
            if model_info:
                print("Token is valid. Accessed model details successfully.")
                return True
            else:
                print("Token is valid but failed to access model details.")
                return True
        except Exception as e:
            print(f"Failed to validate token: {e}")
            return False

    def check_gated_model_access(self, model_id, k_huggingface):
        api = HfApi()
        attempts = 0
        max_attempts = 2

        while attempts < max_attempts:
            try:
                files = api.list_repo_files(model_id, token=k_huggingface)
                # Check if a common small file like 'config.json' is present
                if 'config.json' in files:
                    file_url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
                    # Attempt to fetch the file using the token for authentication
                    headers = {'Authorization': f'Bearer {k_huggingface}'} if k_huggingface else {}
                    response = requests.get(file_url, headers=headers)
                    
                    # Check the response status
                    if response.status_code == 200:
                        print(f"Access to model '{model_id}' is granted.")
                        return "Valid"
                    else:
                        print(f"Access to model '{model_id}' denied or requires additional steps. Status code: {response.status_code}")
                        print(f"        Visit https://huggingface.co/{model_id} to accept terms.")
                        return "Invalid"
                else:
                    print(f"'config.json' not found in {model_id}.")
                    return "Invalid"
            except Exception as e:
                print(f"Error accessing model '{model_id}': {e}")
                return "Invalid"

            except Exception as e:
                error_message = str(e)
                if 'awaiting a review' in error_message:
                    print(f"Access to model '{model_id}' is awaiting review. (Under Review)")
                    return "under_review"
                # print(f"Access to model '{model_id}' is denied. Please accept the terms and conditions.")
                # print(f"Error: {e}")
                # webbrowser.open(f"https://huggingface.co/{model_id}")
                # input("Press Enter after you have accepted the terms and conditions...")

            attempts += 1

        print(f"Failed to access model '{model_id}' after {max_attempts} attempts.")
        return "Invalid"


    

    def get_google_credentials(self):
        if self.is_hf:
            creds_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json_str))
            return credentials
        else:
            with open(self.cfg_private['google']['GOOGLE_APPLICATION_CREDENTIALS'], 'r') as file:
                data = json.load(file)
            creds_json_str = json.dumps(data)
            credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json_str))
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_json_str
            return credentials

    def safe_get_key(self, config, section, key):
        try:
            return config[section][key]
        except KeyError:
            error_message = (f"Your PRIVATE_DATA.yaml is missing the '{key}' configuration under '{section}'. "
                            "If you do not have or need this key, please add it to your PRIVATE_DATA.yaml file "
                            f"with the value set to an empty string or your actual key. Alternatively, you can adjust "
                            "these settings via the sidebar in the Streamlit interface by clicking the 'Set API Keys' button."
                            f" You can manually fix this by adding this to your PRIVATE_DATA.yaml file, matching the format of existing keys >>>\n{section}:\n    {key}: YOUR_KEY_OR_AN_EMPTY_STRING \n<<<")
            raise ValueError(error_message)
        except Exception as e:
            raise ValueError(f"An unexpected error occurred while retrieving the '{key}' under '{section}': {e}")
        
    def report_api_key_status(self):
        missing_keys = []
        present_keys = []

        if self.is_hf:
            k_OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            k_openai_azure = os.getenv('AZURE_API_VERSION')

            k_google_application_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            k_project_id = os.getenv('GOOGLE_PROJECT_ID')
            k_location = os.getenv('GOOGLE_LOCATION')

            k_huggingface = None
            
            k_hyperbolic = os.getenv('HYPERBOLIC_API_KEY')
            k_mistral = os.getenv('MISTRAL_API_KEY')
            k_here = os.getenv('HERE_API_KEY')
            k_opencage = os.getenv('OPENCAGE_API_KEY')
        else:
            # OpenAI Keys
            k_OPENAI_API_KEY = self.safe_get_key(self.cfg_private, 'openai', 'OPENAI_API_KEY')
            k_openai_azure = self.safe_get_key(self.cfg_private, 'openai_azure', 'OPENAI_API_KEY_AZURE')

            # Google Cloud Keys
            k_project_id = self.safe_get_key(self.cfg_private, 'google', 'GOOGLE_PROJECT_ID')
            k_location = self.safe_get_key(self.cfg_private, 'google', 'GOOGLE_LOCATION')
            k_google_application_credentials = self.safe_get_key(self.cfg_private, 'google', 'GOOGLE_APPLICATION_CREDENTIALS')

            # Other API Keys
            k_hyperbolic = self.safe_get_key(self.cfg_private, 'hyperbolic', 'HYPERBOLIC_API_KEY')
            k_mistral = self.safe_get_key(self.cfg_private, 'mistral', 'MISTRAL_API_KEY')
            k_here = self.safe_get_key(self.cfg_private, 'here', 'API_KEY')
            k_opencage = self.safe_get_key(self.cfg_private, 'open_cage_geocode', 'API_KEY')

            # Huggingface Token
            k_huggingface = self.safe_get_key(self.cfg_private, 'huggingface', 'hf_token')
            if k_huggingface:
                os.environ["HUGGING_FACE_KEY"] = k_huggingface



        # Check each key and add to the respective list
        # Google OCR key check
        if self.has_API_key(k_google_application_credentials) and self.has_API_key(k_project_id) and self.has_API_key(k_location):
            google_ocr_results = self.check_google_vision_client() 
            if google_ocr_results['ocr_print']:
                present_keys.append('Google OCR Print (Valid)')
            else:
                present_keys.append('Google OCR Print (Invalid)')
            if google_ocr_results['ocr_hand']:
                present_keys.append('Google OCR Handwriting (Valid)')
            else:
                present_keys.append('Google OCR Handwriting (Invalid)')
        else:
            missing_keys.append('Google OCR')

        # present_keys.append('[MODEL] TEST (Under Review)')

        # HF key check
        if self.has_API_key(k_huggingface):
            is_valid = self.test_hf_token(k_huggingface)
            if is_valid:
                present_keys.append('Hugging Face Local LLMs (Valid)')
            else:
                present_keys.append('Hugging Face Local LLMs (Invalid)')
        else:
            missing_keys.append('Hugging Face Local LLMs')

        # List of gated models to check access for
        for model_id in self.HF_MODEL_LIST:
            access_status = self.check_gated_model_access(model_id, k_huggingface)
            if access_status == "Valid":
                present_keys.append(f'[MODEL] {model_id} (Valid)')
            elif access_status == "under_review":
                present_keys.append(f'[MODEL] {model_id} (Under Review)')
            else:
                present_keys.append(f'[MODEL] {model_id} (Invalid)')
        
        
        
        # OpenAI key check
        if self.has_API_key(k_OPENAI_API_KEY):
            is_valid = self.check_openai_api_key()
            if is_valid:
                present_keys.append('OpenAI (Valid)')
            else:
                present_keys.append('OpenAI (Invalid)')
        else:
            missing_keys.append('OpenAI')

        # Azure OpenAI key check
        # if self.has_API_key(k_openai_azure):
        #     is_valid = self.check_azure_openai_api_key()
        #     if is_valid:
        #         present_keys.append('Azure OpenAI (Valid)')
        #     else:
        #         present_keys.append('Azure OpenAI (Invalid)')
        # else:
        #     missing_keys.append('Azure OpenAI')

        # Google PALM2/Gemini key check
        if self.has_API_key(k_google_application_credentials) and self.has_API_key(k_project_id) and self.has_API_key(k_location): ##################
            vertexai.init(project=k_project_id, location=k_location, credentials=self.get_google_credentials())
            google_results = self.check_google_vertex_genai_api_key()
            if google_results['palm2']:
                present_keys.append('Palm2 (Valid)')
            else:
                present_keys.append('Palm2 (Invalid)')
            if google_results['palm2_langchain']:
                present_keys.append('Palm2 LangChain (Valid)')
            else:
                present_keys.append('Palm2 LangChain (Invalid)')
            if google_results['gemini']:
                present_keys.append('Gemini (Valid)')
            else:
                present_keys.append('Gemini (Invalid)')
        else:
            missing_keys.append('Google VertexAI/GenAI')

        
        # hyperbolic key check
        if self.has_API_key(k_hyperbolic):
            is_valid = self.check_hyperbolic_api_key()
            if is_valid:
                present_keys.append('Hyperbolic (Valid)')
            else:
                present_keys.append('Hyperbolic (Invalid)')
        else:
            missing_keys.append('Hyperbolic')


        # Mistral key check
        if self.has_API_key(k_mistral):
            is_valid = self.check_mistral_api_key()
            if is_valid:
                present_keys.append('Mistral (Valid)')
            else:
                present_keys.append('Mistral (Invalid)')
        else:
            missing_keys.append('Mistral')


        if self.has_API_key(k_here):
            present_keys.append('HERE Geocode (Valid)')
        else:
            missing_keys.append('HERE Geocode (Invalid)')

        if self.has_API_key(k_opencage):
            present_keys.append('OpenCage Geocode (Valid)')
        else:
            missing_keys.append('OpenCage Geocode (Invalid)')

        # Create a report string
        report = "API Key Status Report:\n"
        report += "Present Keys: " + ", ".join(present_keys) + "\n"
        report += "Missing Keys: " + ", ".join(missing_keys) + "\n"

        print(report)
        return present_keys, missing_keys, self.formatted_date