import streamlit as st
import yaml, os, json, random, time, re
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
from itertools import chain
from PIL import Image
import pandas as pd
from typing import Union
from streamlit_extras.let_it_rain import rain
from vouchervision.LeafMachine2_Config_Builder import write_config_file
from vouchervision.VoucherVision_Config_Builder import build_VV_config, run_demo_tests_GPT, run_demo_tests_Palm , TestOptionsGPT, TestOptionsPalm, check_if_usable, run_api_tests
from vouchervision.vouchervision_main import voucher_vision, voucher_vision_OCR_test
from vouchervision.general_utils import test_GPU, get_cfg_from_full_path, summarize_expense_report, create_google_ocr_yaml_config

PROMPTS_THAT_NEED_DOMAIN_KNOWLEDGE = ["Version 1","Version 1 PaLM 2"]
COLORS_EXPENSE_REPORT = {
        'GPT_4': '#8fff66',    # Bright Green
        'GPT_3_5': '#006400',  # Dark Green
        'PALM2': '#66a8ff'     # blue
    }

class ProgressReport:
    def __init__(self, overall_bar, batch_bar, text_overall, text_batch):
        self.overall_bar = overall_bar
        self.batch_bar = batch_bar
        self.text_overall = text_overall
        self.text_batch = text_batch
        self.current_overall_step = 0
        self.total_overall_steps = 20  # number of major steps in machine function
        self.current_batch = 0
        self.total_batches = 20

    def update_overall(self, step_name=""):
        self.current_overall_step += 1
        self.overall_bar.progress(self.current_overall_step / self.total_overall_steps)
        self.text_overall.text(step_name)

    def update_batch(self, step_name=""):
        self.current_batch += 1
        self.batch_bar.progress(self.current_batch / self.total_batches)
        self.text_batch.text(step_name)

    def set_n_batches(self, n_batches):
        self.total_batches = n_batches

    def set_n_overall(self, total_overall_steps):
        self.total_overall_steps = total_overall_steps

    def reset_batch(self, step_name):
        self.current_batch = 0
        self.batch_bar.progress(0)
        self.text_batch.text(step_name)
    def reset_overall(self, step_name):
        self.current_overall_step = 0
        self.overall_bar.progress(0)
        self.text_overall.text(step_name)
    
    def get_n_images(self):
        return self.n_images
    def get_n_overall(self):
        return self.total_overall_steps

def does_private_file_exist():
    dir_home = os.path.dirname(os.path.dirname(__file__))
    path_cfg_private = os.path.join(dir_home, 'PRIVATE_DATA.yaml')
    return os.path.exists(path_cfg_private)

def setup_streamlit_config(dir_home):
    # Define the directory path and filename
    dir_path = os.path.join(dir_home, ".streamlit")
    file_path = os.path.join(dir_path, "config.toml")

    # Check if directory exists, if not create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Create or modify the file with the provided content
    config_content = f"""
    [theme]
    base = "dark"
    primaryColor = "#00ff00"

    [server]
    enableStaticServing = false
    runOnSave = true
    port = 8524
    """

    with open(file_path, "w") as f:
        f.write(config_content.strip())

def display_scrollable_results(JSON_results, test_results, OPT2, OPT3):
    """
    Display the results from JSON_results in a scrollable container.
    """
    # Initialize the container
    con_results = st.empty()
    with con_results.container():
        
        # Start the custom container for all the results
        results_html = """<div class='scrollable-results-container'>"""
        
        for idx, (test_name, _) in enumerate(sorted(test_results.items())):
            _, ind_opt1, ind_opt2, ind_opt3 = test_name.split('__')
            opt2_readable = "Use LeafMachine2" if OPT2[int(ind_opt2.split('-')[1])] else "Don't use LeafMachine2"
            opt3_readable = f"{OPT3[int(ind_opt3.split('-')[1])]}"

            if JSON_results[idx] is None:
                results_html += f"<p>None</p>"
            else:
                formatted_json = json.dumps(JSON_results[idx], indent=4)
                results_html += f"<pre>[{opt2_readable}] + [{opt3_readable}]<br/>{formatted_json}</pre>"
        
        # End the custom container
        results_html += """</div>"""

        # The CSS to make this container scrollable
        css = """
        <style>
            .scrollable-results-container {
                overflow-y: auto;
                height: 600px;
                width: 100%;
                white-space: pre-wrap;  # To wrap the content
                font-family: monospace;  # To give the JSON a code-like appearance
            }
        </style>
        """

        # Apply the CSS and then the results
        st.markdown(css, unsafe_allow_html=True)
        st.markdown(results_html, unsafe_allow_html=True)

def display_test_results(test_results, JSON_results, llm_version):
    if llm_version == 'gpt':
        OPT1, OPT2, OPT3 = TestOptionsGPT.get_options()
    elif llm_version == 'palm':
        OPT1, OPT2, OPT3 = TestOptionsPalm.get_options()
    else:
        raise

    widths = [1] * (len(OPT1) + 2) + [2]
    columns = st.columns(widths)

    with columns[0]:
        st.write("LeafMachine2")
    with columns[1]:
        st.write("Prompt")
    with columns[len(OPT1) + 2]:
        st.write("Scroll to See Last Transcription in Each Test")

    already_written = set()

    for test_name, result in sorted(test_results.items()):
        _, ind_opt1, _, _ = test_name.split('__')
        option_value = OPT1[int(ind_opt1.split('-')[1])]

        if option_value not in already_written:
            with columns[int(ind_opt1.split('-')[1]) + 2]:
                st.write(option_value)
            already_written.add(option_value)

    printed_options = set()

    with columns[-1]:
        display_scrollable_results(JSON_results, test_results, OPT2, OPT3)

    # Close the custom container
    st.write('</div>', unsafe_allow_html=True)


    for idx, (test_name, result) in enumerate(sorted(test_results.items())):
        _, ind_opt1, ind_opt2, ind_opt3 = test_name.split('__')
        opt2_readable = "Use LeafMachine2" if OPT2[int(ind_opt2.split('-')[1])] else "Don't use LeafMachine2"
        opt3_readable = f"{OPT3[int(ind_opt3.split('-')[1])]}"

        if (opt2_readable, opt3_readable) not in printed_options:
            with columns[0]:
                st.info(f"{opt2_readable}")
                st.write('---')
            with columns[1]:
                st.info(f"{opt3_readable}")
                st.write('---')
            printed_options.add((opt2_readable, opt3_readable))

        with columns[int(ind_opt1.split('-')[1]) + 2]:
            if result:
                st.success(f"Test Passed")
            else:
                st.error(f"Test Failed")
            st.write('---')
    
    # success_count = sum(1 for result in test_results.values() if result)
    # failure_count = len(test_results) - success_count
    # proportional_rain("🥇", success_count, "💔", failure_count, font_size=72, falling_speed=5, animation_length="infinite")
    rain_emojis(test_results)

def add_emoji_delay():
    time.sleep(0.3)

def rain_emojis(test_results):
    # test_results = {
    #     'test1': True,   # Test passed
    #     'test2': True,   # Test passed
    #     'test3': True,   # Test passed
    #     'test4': False,  # Test failed
    #     'test5': False,  # Test failed
    #     'test6': False,  # Test failed
    #     'test7': False,  # Test failed
    #     'test8': False,  # Test failed
    #     'test9': False,  # Test failed
    #     'test10': False,  # Test failed
    # }
    success_emojis = ["🥇", "🏆", "🍾", "🙌"]
    failure_emojis = ["💔", "😭"]

    success_count = sum(1 for result in test_results.values() if result)
    failure_count = len(test_results) - success_count

    chosen_emoji = random.choice(success_emojis)
    for _ in range(success_count):
        rain(
            emoji=chosen_emoji,
            font_size=72,
            falling_speed=4,
            animation_length=2,
        )
        add_emoji_delay()

    chosen_emoji = random.choice(failure_emojis)
    for _ in range(failure_count):
        rain(
            emoji=chosen_emoji,
            font_size=72,
            falling_speed=5,
            animation_length=1,
        )
        add_emoji_delay()

def get_prompt_versions(LLM_version):
    yaml_files = [f for f in os.listdir(os.path.join(st.session_state.dir_home, 'custom_prompts')) if f.endswith('.yaml')]

    if LLM_version in ["GPT 4", "GPT 3.5", "Azure GPT 4", "Azure GPT 3.5"]:
        versions = ["Version 1", "Version 1 No Domain Knowledge", "Version 2"]
        return (versions + yaml_files, "Version 2")
    elif LLM_version in ["PaLM 2",]:
        versions = ["Version 1 PaLM 2", "Version 1 PaLM 2 No Domain Knowledge", "Version 2 PaLM 2"]
        return (versions + yaml_files, "Version 2 PaLM 2")
    else:
        # Handle other cases or raise an error
        return (yaml_files, None)

def get_private_file():
    dir_home = os.path.dirname(os.path.dirname(__file__))
    path_cfg_private = os.path.join(dir_home, 'PRIVATE_DATA.yaml')
    return get_cfg_from_full_path(path_cfg_private)

def create_space_saver():
    st.subheader("Space Saving Options")
    col_ss_1, col_ss_2 = st.columns([2,2])
    with col_ss_1:
        st.write("Several folders are created and populated with data during the VoucherVision transcription process.")
        st.write("Below are several options that will allow you to automatically delete temporary files that you may not need for everyday operations.")
        st.write("VoucherVision creates the following folders. Folders marked with a :star: are required if you want to use VoucherVisionEditor for quality control.")
        st.write("`../[Run Name]/Archival_Components`")
        st.write("`../[Run Name]/Config_File`")
        st.write("`../[Run Name]/Cropped_Images` :star:")
        st.write("`../[Run Name]/Logs`")
        st.write("`../[Run Name]/Original_Images` :star:")
        st.write("`../[Run Name]/Transcription` :star:")
    with col_ss_2:
        st.session_state.config['leafmachine']['project']['delete_temps_keep_VVE'] = st.checkbox("Delete Temporary Files (KEEP files required for VoucherVisionEditor)", st.session_state.config['leafmachine']['project'].get('delete_temps_keep_VVE', False))
        st.session_state.config['leafmachine']['project']['delete_all_temps'] = st.checkbox("Keep only the final transcription file", st.session_state.config['leafmachine']['project'].get('delete_all_temps', False),help="*WARNING:* This limits your ability to do quality assurance. This will delete all folders created by VoucherVision, leaving only the `transcription.xlsx` file.")


# def create_private_file():
#     st.session_state.proceed_to_main = False

#     if st.session_state.private_file:
#         cfg_private = get_private_file()
#         create_private_file_0(cfg_private)
#     else:
#         st.title("VoucherVision")
#         create_private_file_0()

def create_private_file():
    st.session_state.proceed_to_main = False
    st.title("VoucherVision")
    col_private,_= st.columns([12,2])

    if st.session_state.private_file:
        cfg_private = get_private_file()
    else:
        cfg_private = {}
        cfg_private['openai'] = {}
        cfg_private['openai']['OPENAI_API_KEY'] =''
        
        cfg_private['openai_azure'] = {}
        cfg_private['openai_azure']['openai_api_key'] = ''
        cfg_private['openai_azure']['api_version'] = ''
        cfg_private['openai_azure']['openai_api_base'] =''
        cfg_private['openai_azure']['openai_organization'] =''
        cfg_private['openai_azure']['openai_api_type'] =''

        cfg_private['google_cloud'] = {}
        cfg_private['google_cloud']['path_json_file'] =''

        cfg_private['google_palm'] = {}
        cfg_private['google_palm']['google_palm_api'] =''
    

    with col_private:
        st.header("Set API keys")
        st.info("***Note:*** There is a known bug with tabs in Streamlit. If you update an input field it may take you back to the 'Project Settings' tab. Changes that you made are saved, it's just an annoying glitch. We are aware of this issue and will fix it as soon as we can.")
        st.warning("To commit changes to API keys you must press the 'Set API Keys' button at the bottom of the page.")
        st.write("Before using VoucherVision you must set your API keys. All keys are stored locally on your computer and are never made public.")
        st.write("API keys are stored in `../VoucherVision/PRIVATE_DATA.yaml`.")
        st.write("Deleting this file will allow you to reset API keys. Alternatively, you can edit the keys in the user interface.")
        st.write("Leave keys blank if you do not intend to use that service.")
        
        st.write("---")
        st.subheader("Google Vision  (*Required*)")
        st.markdown("VoucherVision currently uses [Google Vision API](https://cloud.google.com/vision/docs/ocr) for OCR. Generating an API key for this is more involved than the others. [Please carefully follow the instructions outlined here to create and setup your account.](https://cloud.google.com/vision/docs/setup) ")
        st.markdown("""
        Once your account is created, [visit this page](https://console.cloud.google.com) and create a project. Then follow these instructions:

        - **Select your Project**: If you have multiple projects, ensure you select the one where you've enabled the Vision API.
        - **Open the Navigation Menu**: Click on the hamburger menu (three horizontal lines) in the top left corner.
        - **Go to IAM & Admin**: In the navigation pane, hover over "IAM & Admin" and then click on "Service accounts."
        - **Locate Your Service Account**: Find the service account for which you wish to download the JSON key. If you haven't created a service account yet, you'll need to do so by clicking the "CREATE SERVICE ACCOUNT" button at the top.
        - **Download the JSON Key**:
            - Click on the three dots (actions menu) on the right side of your service account name.
            - Select "Manage keys."
            - In the pop-up window, click on the "ADD KEY" button and select "JSON."
            - The JSON key file will automatically be downloaded to your computer.
        - **Store Safely**: This file contains sensitive data that can be used to authenticate and bill your Google Cloud account. Never commit it to public repositories or expose it in any way. Always keep it safe and secure.
        """)
        with st.container():
            c_in_ocr, c_button_ocr = st.columns([10,2])
            with c_in_ocr:
                google_vision = st.text_input(label = 'Full path to Google Cloud JSON API key file', value = cfg_private['google_cloud'].get('path_json_file', ''),
                                                 placeholder = 'e.g. C:/Documents/Secret_Files/google_API/application_default_credentials.json',
                                                 help ="This API Key is in the form of a JSON file. Please save the JSON file in a safe directory. DO NOT store the JSON key inside of the VoucherVision directory.",
                                                 type='password',key='924857298734590283750932809238')
            with c_button_ocr:
                st.empty()

        
        st.write("---")
        st.subheader("OpenAI")
        st.markdown("API key for first-party OpenAI API. Create an account with OpenAI [here](https://platform.openai.com/signup), then create an API key [here](https://platform.openai.com/account/api-keys).")
        with st.container():
            c_in_openai, c_button_openai = st.columns([10,2])
            with c_in_openai:
                openai_api_key = st.text_input("openai_api_key", cfg_private['openai'].get('OPENAI_API_KEY', ''),
                                                 help='The actual API key. Likely to be a string of 2 character, a dash, and then a 48-character string: sk-XXXXXXXX...',
                                                 placeholder = 'e.g. sk-XXXXXXXX...',
                                                 type='password')
            with c_button_openai:
                st.empty()

        st.write("---")
        st.subheader("OpenAI - Azure")
        st.markdown("This version OpenAI relies on Azure servers directly as is intended for private enterprise instances of OpenAI's services, such as [UM-GPT](https://its.umich.edu/computing/ai). Administrators will provide you with the following information.")
        azure_openai_api_version = st.text_input("azure_openai_api_version", cfg_private['openai_azure'].get('api_version', ''),
                                                 help='API Version e.g. "2023-05-15"',
                                                 placeholder = 'e.g. 2023-05-15',
                                                 type='password')
        azure_openai_api_key = st.text_input("azure_openai_api_key", cfg_private['openai_azure'].get('openai_api_key', ''),
                                                 help='The actual API key. Likely to be a 32-character string',
                                                 placeholder = 'e.g. 12333333333333333333333333333332',
                                                 type='password')
        azure_openai_api_base = st.text_input("azure_openai_api_base", cfg_private['openai_azure'].get('openai_api_base', ''),
                                                 help='The base url for the API e.g. "https://api.umgpt.umich.edu/azure-openai-api"',
                                                 placeholder = 'e.g. https://api.umgpt.umich.edu/azure-openai-api',
                                                 type='password')
        azure_openai_organization = st.text_input("azure_openai_organization", cfg_private['openai_azure'].get('openai_organization', ''),
                                                 help='Your organization code. Likely a short string',
                                                 placeholder = 'e.g. 123456',
                                                 type='password')
        azure_openai_api_type = st.text_input("azure_openai_api_type", cfg_private['openai_azure'].get('openai_api_type', ''),
                                                 help='The API type. Typically "azure"',
                                                 placeholder = 'e.g. azure',
                                                 type='password')
        with st.container():
            c_in_azure, c_button_azure = st.columns([10,2])
            with c_button_azure:
                st.empty()
        
        st.write("---")
        st.subheader("Google PaLM 2")
        st.markdown('Follow these [instructions](https://developers.generativeai.google/tutorials/setup) to generate an API key for PaLM 2. You may need to also activate an account with [MakerSuite](https://makersuite.google.com/app/apikey) and enable "early access."')
        with st.container():
            c_in_palm, c_button_palm = st.columns([10,2])
            with c_in_palm:
                google_palm = st.text_input("Google PaLM 2 API Key", cfg_private['google_palm'].get('google_palm_api', ''),
                                                 help='The MakerSuite API key e.g. a 32-character string',
                                                 placeholder='e.g. SATgthsykuE64FgrrrrEervr3S4455t_geyDeGq',
                                                 type='password')

        with st.container():
            with c_button_ocr:
                st.write("##")
                st.button("Test OCR", on_click=test_API, args=['google_vision',c_in_ocr, cfg_private,openai_api_key,azure_openai_api_version,azure_openai_api_key,
                                                                    azure_openai_api_base,azure_openai_organization,azure_openai_api_type,google_vision,google_palm])

        with st.container():
            with c_button_openai:
                st.write("##")
                st.button("Test OpenAI", on_click=test_API, args=['openai',c_in_openai, cfg_private,openai_api_key,azure_openai_api_version,azure_openai_api_key,
                                                                    azure_openai_api_base,azure_openai_organization,azure_openai_api_type,google_vision,google_palm])
                
        with st.container():
            with c_button_azure:
                st.write("##")
                st.button("Test Azure OpenAI", on_click=test_API, args=['azure_openai',c_in_azure, cfg_private,openai_api_key,azure_openai_api_version,azure_openai_api_key,
                                                                    azure_openai_api_base,azure_openai_organization,azure_openai_api_type,google_vision,google_palm])
                
        with st.container():
            with c_button_palm:
                st.write("##")
                st.button("Test PaLM 2", on_click=test_API, args=['palm',c_in_palm, cfg_private,openai_api_key,azure_openai_api_version,azure_openai_api_key,
                                                                    azure_openai_api_base,azure_openai_organization,azure_openai_api_type,google_vision,google_palm])


        st.button("Set API Keys",type='primary', on_click=save_changes_to_API_keys, args=[cfg_private,openai_api_key,azure_openai_api_version,azure_openai_api_key,
                                                                    azure_openai_api_base,azure_openai_organization,azure_openai_api_type,google_vision,google_palm])
        if st.button('Proceed to VoucherVision'):
            st.session_state.proceed_to_private = False
            st.session_state.proceed_to_main = True

def test_API(api, message_loc, cfg_private,openai_api_key,azure_openai_api_version,azure_openai_api_key, azure_openai_api_base,azure_openai_organization,azure_openai_api_type,google_vision,google_palm):
    # Save the API keys
    save_changes_to_API_keys(cfg_private,openai_api_key,azure_openai_api_version,azure_openai_api_key,azure_openai_api_base,azure_openai_organization,azure_openai_api_type,google_vision,google_palm)
    
    with st.spinner('Performing validation checks...'):
        if api == 'google_vision':
            print("*** Google Vision OCR API Key ***")
            try:
                demo_config_path = os.path.join(st.session_state.dir_home,'demo','validation_configs','google_vision_ocr_test.yaml')
                demo_images_path = os.path.join(st.session_state.dir_home, 'demo', 'demo_images')
                demo_out_path = os.path.join(st.session_state.dir_home, 'demo', 'demo_output','run_name')
                create_google_ocr_yaml_config(demo_config_path, demo_images_path, demo_out_path)
                voucher_vision_OCR_test(demo_config_path, st.session_state.dir_home, None, demo_images_path)
                with message_loc:
                    st.success("Google Vision OCR API Key Valid :white_check_mark:")
                return True
            except Exception as e:
                with message_loc:
                    st.error(f"Google Vision OCR API Key Failed! {e}")
                return False
            
        elif api == 'openai':
            print("*** OpenAI API Key ***")
            try:
                if run_api_tests('openai'):
                    with message_loc:
                        st.success("OpenAI API Key Valid :white_check_mark:")
                else:
                    with message_loc:
                        st.error("OpenAI API Key Failed:exclamation:")
                    return False
            except Exception as e:
                with message_loc:
                    st.error(f"OpenAI API Key Failed:exclamation: {e}")

        elif api == 'azure_openai':
            print("*** Azure OpenAI API Key ***")
            try:
                if run_api_tests('azure_openai'):
                    with message_loc:
                        st.success("Azure OpenAI API Key Valid :white_check_mark:")
                else:
                    with message_loc:
                        st.error(f"Azure OpenAI API Key Failed:exclamation:")
                    return False
            except Exception as e:
                with message_loc:
                    st.error(f"Azure OpenAI API Key Failed:exclamation: {e}")
        elif api == 'palm':
            print("*** Google PaLM 2 API Key ***")
            try:
                if run_api_tests('palm'):
                    with message_loc:
                        st.success("Google PaLM 2 API Key Valid :white_check_mark:")
                else:
                    with message_loc:
                        st.error("Google PaLM 2 API Key Failed:exclamation:")
                    return False
            except Exception as e:
                with message_loc:
                    st.error(f"Google PaLM 2 API Key Failed:exclamation: {e}")
       

def save_changes_to_API_keys(cfg_private,openai_api_key,azure_openai_api_version,azure_openai_api_key,
                             azure_openai_api_base,azure_openai_organization,azure_openai_api_type,google_vision,google_palm):
    # Update the configuration dictionary with the new values
    cfg_private['openai']['OPENAI_API_KEY'] = openai_api_key 

    cfg_private['openai_azure']['api_version'] = azure_openai_api_version
    cfg_private['openai_azure']['openai_api_key'] = azure_openai_api_key
    cfg_private['openai_azure']['openai_api_base'] = azure_openai_api_base
    cfg_private['openai_azure']['openai_organization'] = azure_openai_organization
    cfg_private['openai_azure']['openai_api_type'] = azure_openai_api_type

    cfg_private['google_cloud']['path_json_file'] = google_vision

    cfg_private['google_palm']['google_palm_api'] = google_palm
    # Call the function to write the updated configuration to the YAML file
    write_config_file(cfg_private, st.session_state.dir_home, filename="PRIVATE_DATA.yaml")
    st.session_state.private_file = does_private_file_exist()

# Function to load a YAML file and update session_state
def load_prompt_yaml(filename):
    with open(filename, 'r') as file:
        st.session_state['prompt_info'] = yaml.safe_load(file)
        st.session_state['instructions'] = st.session_state['prompt_info'].get('instructions', st.session_state['default_instructions']) 
        st.session_state['json_formatting_instructions'] = st.session_state['prompt_info'].get('json_formatting_instructions', st.session_state['default_json_formatting_instructions'] )
        st.session_state['rules'] = st.session_state['prompt_info'].get('rules', {})
        st.session_state['mapping'] = st.session_state['prompt_info'].get('mapping', {})
        st.session_state['LLM'] = st.session_state['prompt_info'].get('LLM', 'gpt')

        # Placeholder:
        st.session_state['assigned_columns'] = list(chain.from_iterable(st.session_state['mapping'].values())) 

def save_prompt_yaml(filename):
    yaml_content = {
        'instructions': st.session_state['instructions'],
        'json_formatting_instructions': st.session_state['json_formatting_instructions'],
        'rules': st.session_state['rules'],
        'mapping': st.session_state['mapping'],
        'LLM': st.session_state['LLM']
    }
    
    dir_prompt = os.path.join(st.session_state.dir_home, 'custom_prompts')
    filepath = os.path.join(dir_prompt, f"{filename}.yaml")

    with open(filepath, 'w') as file:
        yaml.safe_dump(yaml_content, file)

    st.success(f"Prompt saved as '{filename}.yaml'.")

def check_unique_mapping_assignments():
    if len(st.session_state['assigned_columns']) != len(set(st.session_state['assigned_columns'])):
        st.error("Each column name must be assigned to only one category.")
        return False
    else:
        st.success("Mapping confirmed.")
        return True

def check_prompt_yaml_filename(fname):
    # Check if the filename only contains letters, numbers, underscores, and dashes
    pattern = r'^[\w-]+$'
    
    # The \w matches any alphanumeric character and is equivalent to the character class [a-zA-Z0-9_].
    # The hyphen - is literally matched.

    if re.match(pattern, fname):
        return True
    else:
        return False


def btn_load_prompt(selected_yaml_file, dir_prompt):
    if selected_yaml_file:
        yaml_file_path = os.path.join(dir_prompt, selected_yaml_file)
        load_prompt_yaml(yaml_file_path)
    elif not selected_yaml_file:
        # Directly assigning default values since no file is selected
        st.session_state['prompt_info'] = {}
        st.session_state['instructions'] = st.session_state['default_instructions']
        st.session_state['json_formatting_instructions'] = st.session_state['default_json_formatting_instructions'] 
        st.session_state['rules'] = {}
        st.session_state['LLM'] = 'gpt'
        
        st.session_state['assigned_columns'] = []

        st.session_state['prompt_info'] = {
            'instructions': st.session_state['instructions'],
            'json_formatting_instructions': st.session_state['json_formatting_instructions'],
            'rules': st.session_state['rules'],
            'mapping': st.session_state['mapping'],
            'LLM': st.session_state['LLM']
        }

def build_LLM_prompt_config():
    st.session_state['assigned_columns'] = []
    st.session_state['default_instructions'] = """1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below.
2. You should map the unstructured OCR text to the appropriate JSON key and then populate the field based on its rules.
3. Some JSON key fields are permitted to remain empty if the corresponding information is not found in the unstructured OCR text.
4. Ignore any information in the OCR text that doesn't fit into the defined JSON structure.
5. Duplicate dictionary fields are not allowed.
6. Ensure that all JSON keys are in lowercase.
7. Ensure that new JSON field values follow sentence case capitalization.
8. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
9. Ensure the output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
10. Only return a JSON dictionary represented as a string. You should not explain your answer."""
    st.session_state['default_json_formatting_instructions'] = """The next section of instructions outlines how to format the JSON dictionary. The keys are the same as those of the final formatted JSON object.
For each key there is a format requirement that specifies how to transcribe the information for that key. 
The possible formatting options are:
1. "verbatim transcription" - field is populated with verbatim text from the unformatted OCR.
2. "spell check transcription" - field is populated with spelling corrected text from the unformatted OCR.
3. "boolean yes no" - field is populated with only yes or no.
4. "boolean 1 0" - field is populated with only 1 or 0.
5. "integer" - field is populated with only an integer.
6. "[list]" - field is populated from one of the values in the list.
7. "yyyy-mm-dd" - field is populated with a date in the format year-month-day.
The desired null value is also given. Populate the field with the null value of the information for that key is not present in the unformatted OCR text."""
    
    # Start building the Streamlit app
    col_prompt_main_left, ___, col_prompt_main_right = st.columns([6,1,3])

    
    with col_prompt_main_left:
        
        st.title("Custom LLM Prompt Builder")
        st.subheader('About')
        st.write("This form allows you to craft a prompt for your specific task.")
        st.subheader('How it works')
        st.write("1. Edit this page until you are happy with your instructions. We recommend looking at the basic structure, writing down your prompt inforamtion in a Word document so that it does not randomly disappear, and then copying and pasting that info into this form once your whole prompt structure is defined.")
        st.write("2. After you enter all of your prompt instructions, click 'Save' and give your file a name.")
        st.write("3. This file will be saved as a yaml configuration file in the `..VoucherVision/custom_prompts` folder.")
        st.write("4. When you go back the main VoucherVision page you will now see your custom prompt available in the 'Prompt Version' dropdown menu.")
        st.write("5. Select your custom prompt. Note, your prompt will only be available for the LLM that you set when filling out the form below.")


        dir_prompt = os.path.join(st.session_state.dir_home, 'custom_prompts')
        yaml_files = [f for f in os.listdir(dir_prompt) if f.endswith('.yaml')]
        col_load_text, col_load_btn = st.columns([8,2])
        with col_load_text:
        # Dropdown for selecting a YAML file
            selected_yaml_file = st.selectbox('Select a prompt YAML file to load:', [''] + yaml_files)
        with col_load_btn:
            st.write('##')
            # Button to load the selected prompt
            st.button('Load Prompt', on_click=btn_load_prompt, args=[selected_yaml_file, dir_prompt])
                


        # Define the options for the dropdown
        llm_options = ['gpt', 'palm']
        # Create the dropdown and set the value to session_state['LLM']
        st.session_state['LLM'] = st.selectbox('Set LLM:', llm_options, index=llm_options.index(st.session_state.get('LLM', 'gpt')))

        

        # Instructions Section
        st.header("Instructions")
        st.write("These are the general instructions that guide the LLM through the transcription task. We recommend using the default instructions unless you have a specific reason to change them.")
        
        st.session_state['instructions'] = st.text_area("Enter instructions:", value=st.session_state['default_instructions'].strip(), height=350, disabled=True)

        st.write('---')

        # Column Instructions Section
        st.header("JSON Formatting Instructions")
        st.write("The following section tells the LLM how we want to structure the JSON dictionary. We do not recommend changing this section because it would likely result in unstable and inconsistent behavior.")
        st.session_state['json_formatting_instructions'] = st.text_area("Enter column instructions:", value=st.session_state['default_json_formatting_instructions'], height=350, disabled=True)





        st.write('---')
        col_left, col_right = st.columns([6,4])
        with col_left:
            st.subheader('Add/Edit Columns')
            
            # Initialize rules in session state if not already present
            if 'rules' not in st.session_state or not st.session_state['rules']:
                st.session_state['rules']['Dictionary'] = {
                    "catalog_number": {
                        "format": "verbatim transcription",
                        "null_value": "",
                        "description": "The barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits."
                    }
                }
                st.session_state['rules']['SpeciesName'] = {
                    "taxonomy": ["Genus_species"]
                }

            # Layout for adding a new column name
            # col_text, col_textbtn = st.columns([8, 2])
            # with col_text:
            new_column_name = st.text_input("Enter a new column name:")
            # with col_textbtn:
            # st.write('##')
            if st.button("Add New Column") and new_column_name:
                if new_column_name not in st.session_state['rules']['Dictionary']:
                    st.session_state['rules']['Dictionary'][new_column_name] = {"format": "", "null_value": "", "description": ""}
                    st.success(f"New column '{new_column_name}' added. Now you can edit its properties.")
                else:
                    st.error("Column name already exists. Please enter a unique column name.")

            # Get columns excluding the protected "catalog_number"
            st.write('#')
            editable_columns = [col for col in st.session_state['rules']['Dictionary'] if col != "catalog_number"]
            column_name = st.selectbox("Select a column to edit:", [""] + editable_columns)

            # Handle rules editing
            current_rule = st.session_state['rules']['Dictionary'].get(column_name, {
                "format": "",
                "null_value": "",
                "description": ""
            })

            if 'selected_column' not in st.session_state:
                st.session_state['selected_column'] = column_name

            


            # Form for input fields
            with st.form(key='rule_form'):
                format_options = ["verbatim transcription", "spell check transcription", "boolean yes no", "boolean 1 0", "integer", "[list]", "yyyy-mm-dd"]
                current_rule["format"] = st.selectbox("Format:", format_options, index=format_options.index(current_rule["format"]) if current_rule["format"] else 0)
                current_rule["null_value"] = st.text_input("Null value:", value=current_rule["null_value"])
                current_rule["description"] = st.text_area("Description:", value=current_rule["description"])
                commit_button = st.form_submit_button("Commit Column")

            default_rule = {
                "format": format_options[0],  # default format
                "null_value": "",  # default null value
                "description": "",  # default description
            }
            if st.session_state['selected_column'] != column_name:
                # Column has changed. Update the session_state selected column.
                st.session_state['selected_column'] = column_name
                # Reset the current rule to the default for this new column, or a blank rule if not set.
                current_rule = st.session_state['rules']['Dictionary'].get(column_name, default_rule.copy())

            # Handle commit action
            if commit_button and column_name:
                # Commit the rules to the session state.
                st.session_state['rules']['Dictionary'][column_name] = current_rule.copy()
                st.success(f"Column '{column_name}' added/updated in rules.")

                # Force the form to reset by clearing the fields from the session state
                st.session_state.pop('selected_column', None)  # Clear the selected column to force reset

                # st.session_state['rules'][column_name] = current_rule
                # st.success(f"Column '{column_name}' added/updated in rules.")

                # # Reset current_rule to default values for the next input
                # current_rule["format"] = default_rule["format"]
                # current_rule["null_value"] = default_rule["null_value"]
                # current_rule["description"] = default_rule["description"]

                # # To ensure that the form fields are reset, we can clear them from the session state
                # for key in current_rule.keys():
                #     st.session_state[key] = default_rule[key]

            # Layout for removing an existing column
            # del_col, del_colbtn = st.columns([8, 2])
            # with del_col:
            delete_column_name = st.selectbox("Select a column to delete:", [""] + editable_columns, key='delete_column')
            # with del_colbtn:
            # st.write('##')
            if st.button("Delete Column") and delete_column_name:
                del st.session_state['rules'][delete_column_name]
                st.success(f"Column '{delete_column_name}' removed from rules.")


            

        with col_right:
            # Display the current state of the JSON rules
            st.subheader('Formatted Columns')
            st.json(st.session_state['rules']['Dictionary'])

            # st.subheader('All Prompt Info')
            # st.json(st.session_state['prompt_info'])


        st.write('---')


        col_left_mapping, col_right_mapping = st.columns([6,4])
        with col_left_mapping:
            st.header("Mapping")
            st.write("Assign each column name to a single category.")
            st.session_state['refresh_mapping'] = False

            # Dynamically create a list of all column names that can be assigned
            # This assumes that the column names are the keys in the dictionary under 'rules'
            all_column_names = list(st.session_state['rules']['Dictionary'].keys())

            categories = ['TAXONOMY', 'GEOGRAPHY', 'LOCALITY', 'COLLECTING', 'MISCELLANEOUS']
            if ('mapping' not in st.session_state) or (st.session_state['mapping'] == {}):
                st.session_state['mapping'] = {category: [] for category in categories}
            for category in categories:
                # Filter out the already assigned columns
                available_columns = [col for col in all_column_names if col not in st.session_state['assigned_columns'] or col in st.session_state['mapping'].get(category, [])]

                # Ensure the current mapping is a subset of the available options
                current_mapping = [col for col in st.session_state['mapping'].get(category, []) if col in available_columns]

                # Provide a safe default if the current mapping is empty or contains invalid options
                safe_default = current_mapping if all(col in available_columns for col in current_mapping) else []

                # Create a multi-select widget for the category with a safe default
                selected_columns = st.multiselect(
                    f"Select columns for {category}:",
                    available_columns,
                    default=safe_default,
                    key=f"mapping_{category}"
                )
                # Update the assigned_columns based on the selections
                for col in current_mapping:
                    if col not in selected_columns and col in st.session_state['assigned_columns']:
                        st.session_state['assigned_columns'].remove(col)
                        st.session_state['refresh_mapping'] = True

                for col in selected_columns:
                    if col not in st.session_state['assigned_columns']:
                        st.session_state['assigned_columns'].append(col)
                        st.session_state['refresh_mapping'] = True

                # Update the mapping in session state when there's a change
                st.session_state['mapping'][category] = selected_columns
            if st.session_state['refresh_mapping']:
                st.session_state['refresh_mapping'] = False

        # Button to confirm and save the mapping configuration
        if st.button('Confirm Mapping'):
            if check_unique_mapping_assignments():
                # Proceed with further actions since the mapping is confirmed and unique
                pass

        with col_right_mapping:
            # Display the current state of the JSON rules
            st.subheader('Formatted Column Maps')
            st.json(st.session_state['mapping'])


        col_left_save, col_right_save = st.columns([6,4])
        with col_left_save:
            # Input for new file name
            new_filename = st.text_input("Enter filename to save your prompt as a configuration YAML:",placeholder='my_prompt_name')
            # Button to save the new YAML file
            if st.button('Save YAML', type='primary'):
                if new_filename:
                    if check_unique_mapping_assignments():
                        if check_prompt_yaml_filename(new_filename):
                            save_prompt_yaml(new_filename)
                        else:
                            st.error("File name can only contain letters, numbers, underscores, and dashes. Cannot contain spaces.")
                    else:
                        st.error("Mapping contains an error. Make sure that each column is assigned to only ***one*** category.")
                else:
                    st.error("Please enter a filename.")
        
            if st.button('Exit'):
                st.session_state.proceed_to_build_llm_prompt = False
                st.session_state.proceed_to_main = True
                st.rerun()
    with col_prompt_main_right:
        st.subheader('All Prompt Components')
        st.session_state['prompt_info'] = {
            'instructions': st.session_state['instructions'],
            'json_formatting_instructions': st.session_state['json_formatting_instructions'],
            'rules': st.session_state['rules'],
            'mapping': st.session_state['mapping'],
            'LLM': st.session_state['LLM']
        }
        st.json(st.session_state['prompt_info'])
    # # Mapping Section
    # st.header("Mapping")
    # mapping = {}
    # category_options = ["TAXONOMY", "GEOGRAPHY", "LOCALITY", "COLLECTING", "MISCELLANEOUS"]
    # if columns:
    #     for column in columns.split(','):
    #         column = column.strip()
    #         category = st.selectbox(f"Category for {column}:", category_options, key=f"category_{column}")
    #         mapping[column] = category
    

    # # Button to save YAML file
    # if st.button("Save YAML"):
    #     yaml_content = {
    #         "instructions": instructions_list,
    #         "column_instructions": column_instructions,
    #         "rules": rules,
    #         "mapping": mapping
    #     }
    #     save_yaml(yaml_content, filename="rules_config.yaml")
    #     st.success("YAML configuration saved!")

    # Optional: Display the YAML content on the page
    # if st.checkbox("Show YAML"):
    # st.write(yaml_content)

def save_yaml(content, filename="rules_config.yaml"):
    with open(filename, 'w') as file:
        yaml.dump(content, file)

# def process_batch(progress_report):
#     # First, write the config file.
#     write_config_file(st.session_state.config, st.session_state.dir_home, filename="VoucherVision.yaml")

#     # If using a custom prompt, pass the full path to the prompt config. else it's just a meaningless path that won't do anything
#     path_custom_prompts = os.path.join(st.session_state.dir_home,'custom_prompts',st.session_state.config['leafmachine']['LLM_version'])

#     # Call the machine function.
#     last_JSON_response = voucher_vision(None, st.session_state.dir_home, path_custom_prompts, None, progress_report)
#     # Format the JSON string for display.
#     if last_JSON_response is None:
#         st.markdown(f"Last JSON object in the batch: NONE")
#     else:
#         try:
#             formatted_json = json.dumps(json.loads(last_JSON_response), indent=4)
#         except:
#             formatted_json = json.dumps(last_JSON_response, indent=4)
#         st.markdown(f"Last JSON object in the batch:\n```\n{formatted_json}\n```")
#         st.balloons()

def show_header_welcome():
    st.session_state.logo_path = os.path.join(st.session_state.dir_home, 'img','logo.png')
    st.session_state.logo = Image.open(st.session_state.logo_path)
    st.image(st.session_state.logo, width=250)
    # # st.image("img/logo.png", use_column_width=True)
    # st.markdown(f'<a href="https://github.com/Gene-Weaver/VoucherVision"><img src="http://localhost:8000/{st.session_state.logo_path}" width="200"></a>', unsafe_allow_html=True)
    # hide_img_fs = '''
    # <style>
    # button[title="View fullscreen"]{
    #     visibility: hidden;}
    # </style>
    # '''
    # st.markdown(hide_img_fs, unsafe_allow_html=True)

def content_header():
    # st.title("VoucherVision")

    col_run_1, col_run_2, col_run_3 = st.columns([4,2,2])
    col_test = st.container()
    # _, col_test, __ = st.columns([1,10,1])
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.subheader("Overall Progress")
    col_run_info_1 = st.columns([1])[0]
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.header("Configuration Settings")

    with col_run_info_1:
        # Progress
        # Progress
        # st.subheader('Project')
        # bar = st.progress(0)
        # new_text = st.empty()  # Placeholder for current step name
        # progress_report = ProgressReportVV(bar, new_text, n_images=10)

        # Progress
        overall_progress_bar = st.progress(0)
        text_overall = st.empty()  # Placeholder for current step name
        st.subheader('Transcription Progress')
        batch_progress_bar = st.progress(0)
        text_batch = st.empty()  # Placeholder for current step name
        progress_report = ProgressReport(overall_progress_bar, batch_progress_bar, text_overall, text_batch)
        st.info("***Note:*** There is a known bug with tabs in Streamlit. If you update an input field it may take you back to the 'Project Settings' tab. Changes that you made are saved, it's just an annoying glitch. We are aware of this issue and will fix it as soon as we can.")
        st.write("If you use VoucherVision frequently, you can change the default values that are auto-populated in the form below. In a text editor or IDE, edit the first few rows in the file `../VoucherVision/vouchervision/VoucherVision_Config_Builder.py`")
        

    with col_run_1:
        show_header_welcome()
        st.subheader('Run VoucherVision')
        if check_if_usable():
            if st.button("Start Processing", type='primary'):
            
                # First, write the config file.
                write_config_file(st.session_state.config, st.session_state.dir_home, filename="VoucherVision.yaml")

                path_custom_prompts = os.path.join(st.session_state.dir_home,'custom_prompts',st.session_state.config['leafmachine']['project']['prompt_version'])
                # Call the machine function.
                last_JSON_response, total_cost = voucher_vision(None, st.session_state.dir_home, path_custom_prompts, None, progress_report,path_api_cost=os.path.join(st.session_state.dir_home,'api_cost','api_cost.yaml'))
                
                if total_cost:
                    st.success(f":money_with_wings: This run cost :heavy_dollar_sign:{total_cost:.4f}")
                
                # Format the JSON string for display.
                if last_JSON_response is None:
                    st.markdown(f"Last JSON object in the batch: NONE")
                else:
                    try:
                        formatted_json = json.dumps(json.loads(last_JSON_response), indent=4)
                    except:
                        formatted_json = json.dumps(last_JSON_response, indent=4)
                    st.markdown(f"Last JSON object in the batch:\n```\n{formatted_json}\n```")
                    st.balloons()

        else:
            st.button("Start Processing", type='primary', disabled=True)
            st.error(":heavy_exclamation_mark: Required API keys not set. Please visit the 'API Keys' tab and set the Google Vision OCR API key and at least one LLM key.")

    with col_run_2:
        st.subheader('Run Tests', help="")
        st.write('We include a single image for testing. If you want to test all of the available prompts and LLMs on a different set of images, copy your images into `../VoucherVision/demo/demo_images`.')
        if st.button("Test GPT"):
            progress_report.set_n_overall(TestOptionsGPT.get_length())
            test_results, JSON_results = run_demo_tests_GPT(progress_report)
            with col_test:
                display_test_results(test_results, JSON_results, 'gpt')
            st.balloons()

        if st.button("Test PaLM2"):
            progress_report.set_n_overall(TestOptionsPalm.get_length())
            test_results, JSON_results = run_demo_tests_Palm(progress_report)
            with col_test:
                display_test_results(test_results, JSON_results, 'palm')
            st.balloons()

    with col_run_3:
        st.subheader('Check GPU')
        if st.button("GPU"):
            success, info = test_GPU()

            if success:
                st.balloons()
                for message in info:
                    st.success(message)
            else:
                for message in info:
                    st.error(message)

def content_tab_settings():
    st.header('Project')
    col_project_1, col_project_2 = st.columns([4,2])

    st.write("---")
    st.header('Input Images')
    col_local_1, col_local_2 = st.columns([4,2])              

    # st.write("---")
    # st.header('Modules')
    # col_m1, col_m2 = st.columns(2)

    st.write("---")
    st.header('Cropped Components')    
    col_cropped_1, col_cropped_2 = st.columns([4,4])        

    os.path.join(st.session_state.dir_home, )
    ### Project
    with col_project_1:
        st.session_state.config['leafmachine']['project']['run_name'] = st.text_input("Run name", st.session_state.config['leafmachine']['project'].get('run_name', ''))
        st.session_state.config['leafmachine']['project']['dir_output'] = st.text_input("Output directory", st.session_state.config['leafmachine']['project'].get('dir_output', ''))
    
    ### Input Images Local
    with col_local_1:
        st.session_state.config['leafmachine']['project']['dir_images_local'] = st.text_input("Input images directory", st.session_state.config['leafmachine']['project'].get('dir_images_local', ''))
        st.session_state.config['leafmachine']['project']['continue_run_from_partial_xlsx'] = st.text_input("Continue run from partially completed project XLSX", st.session_state.config['leafmachine']['project'].get('continue_run_from_partial_xlsx', ''), disabled=True)
        st.write("---")
        st.subheader('LLM Version')
        st.markdown(
            """
            ***Note:*** GPT-4 is 20x more expensive than GPT-3.5  
            """
            )
        st.session_state.config['leafmachine']['LLM_version'] = st.selectbox("LLM version", ["GPT 4", "GPT 3.5", "Azure GPT 4", "Azure GPT 3.5", "PaLM 2"], index=["GPT 4", "GPT 3.5", "Azure GPT 4", "Azure GPT 3.5", "PaLM 2"].index(st.session_state.config['leafmachine'].get('LLM_version', 'Azure GPT 4')))

        st.write("---")
        st.subheader('Prompt Version')
        versions, default_version = get_prompt_versions(st.session_state.config['leafmachine']['LLM_version'])

        if versions:
            selected_version = st.session_state.config['leafmachine']['project'].get('prompt_version', default_version)
            if selected_version not in versions:
                selected_version = default_version
            st.session_state.config['leafmachine']['project']['prompt_version'] = st.selectbox("Prompt Version", versions, index=versions.index(selected_version))

        # if st.session_state.config['leafmachine']['LLM_version'] in ["GPT 4", "GPT 3.5", "Azure GPT 4", "Azure GPT 3.5",]:
        #     st.session_state.config['leafmachine']['project']['prompt_version'] = st.selectbox("Prompt Version", ["Version 1", "Version 1 No Domain Knowledge", "Version 2"], index=["Version 1", "Version 1 No Domain Knowledge", "Version 2"].index(st.session_state.config['leafmachine']['project'].get('prompt_version', "Version 2")))
        # elif st.session_state.config['leafmachine']['LLM_version'] in ["PaLM 2",]:
        #     st.session_state.config['leafmachine']['project']['prompt_version'] = st.selectbox("Prompt Version", ["Version 1 PaLM 2", "Version 1 PaLM 2 No Domain Knowledge", "Version 2 PaLM 2"], index=["Version 1 PaLM 2", "Version 1 PaLM 2 No Domain Knowledge", "Version 2 PaLM 2"].index(st.session_state.config['leafmachine']['project'].get('prompt_version', "Version 2 PaLM 2")))

    ### Modules
    # with col_m1:
    #     st.session_state.config['leafmachine']['modules']['specimen_crop'] = st.checkbox("Specimen Close-up", st.session_state.config['leafmachine']['modules'].get('specimen_crop', True),disabled=True)

    ### cropped_components
    # with col_cropped_1:
    #     st.session_state.config['leafmachine']['cropped_components']['do_save_cropped_annotations'] = st.checkbox("Save cropped components as images", st.session_state.config['leafmachine']['cropped_components'].get('do_save_cropped_annotations', True), disabled=True)
    #     st.session_state.config['leafmachine']['cropped_components']['save_per_image'] = st.checkbox("Save cropped components grouped by specimen", st.session_state.config['leafmachine']['cropped_components'].get('save_per_image', False), disabled=True)
    #     st.session_state.config['leafmachine']['cropped_components']['save_per_annotation_class'] = st.checkbox("Save cropped components grouped by type", st.session_state.config['leafmachine']['cropped_components'].get('save_per_annotation_class', True), disabled=True)
    #     st.session_state.config['leafmachine']['cropped_components']['binarize_labels'] = st.checkbox("Binarize labels", st.session_state.config['leafmachine']['cropped_components'].get('binarize_labels', False), disabled=True)
    #     st.session_state.config['leafmachine']['cropped_components']['binarize_labels_skeletonize'] = st.checkbox("Binarize and skeletonize labels", st.session_state.config['leafmachine']['cropped_components'].get('binarize_labels_skeletonize', False), disabled=True)
    
    with col_cropped_1:
        default_crops = st.session_state.config['leafmachine']['cropped_components'].get('save_cropped_annotations', ['leaf_whole'])
        st.write("Prior to transcription, use LeafMachine2 to crop all labels from input images to create label collages for each specimen image. (Requires GPU)")
        st.session_state.config['leafmachine']['use_RGB_label_images'] = st.checkbox("Use LeafMachine2 label collage for transcriptions", st.session_state.config['leafmachine'].get('use_RGB_label_images', False))

        st.session_state.config['leafmachine']['cropped_components']['save_cropped_annotations'] = st.multiselect("Components to crop",  
                ['ruler', 'barcode','label', 'colorcard','map','envelope','photo','attached_item','weights',
                'leaf_whole', 'leaf_partial', 'leaflet', 'seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many', 'bud','specimen','roots','wood'],default=default_crops)
    with col_cropped_2:
        ba = os.path.join(st.session_state.dir_home,'demo', 'ba','ba2.png')
        image = Image.open(ba)
        st.image(image, caption='LeafMachine2 Collage', output_format = "PNG")

def content_tab_component():
    st.header('Archival Components')
    ACD_version = st.selectbox("Archival Component Detector (ACD) Version", ["Version 2.1", "Version 2.2"])
    
    ACD_confidence_default = int(st.session_state.config['leafmachine']['archival_component_detector']['minimum_confidence_threshold'] * 100)
    ACD_confidence = st.number_input("ACD Confidence Threshold (%)", min_value=0, max_value=100,value=ACD_confidence_default)
    st.session_state.config['leafmachine']['archival_component_detector']['minimum_confidence_threshold'] = float(ACD_confidence/100)

    st.session_state.config['leafmachine']['archival_component_detector']['do_save_prediction_overlay_images'] = st.checkbox("Save Archival Prediction Overlay Images", st.session_state.config['leafmachine']['archival_component_detector'].get('do_save_prediction_overlay_images', True))
    
    st.session_state.config['leafmachine']['archival_component_detector']['ignore_objects_for_overlay'] = st.multiselect("Hide Archival Components in Prediction Overlay Images",  
                ['ruler', 'barcode','label', 'colorcard','map','envelope','photo','attached_item','weights',],
                default=[])

    # Depending on the selected version, set the configuration
    if ACD_version == "Version 2.1":
        st.session_state.config['leafmachine']['archival_component_detector']['detector_type'] = 'Archival_Detector'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_version'] = 'PREP_final'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_iteration'] = 'PREP_final'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_weights'] = 'best.pt'
    elif ACD_version == "Version 2.2": #TODO update this to version 2.2
        st.session_state.config['leafmachine']['archival_component_detector']['detector_type'] = 'Archival_Detector'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_version'] = 'PREP_final'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_iteration'] = 'PREP_final'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_weights'] = 'best.pt'


def content_tab_processing():
    st.header('Processing Options')
    col_processing_1, col_processing_2 = st.columns([2,2,])
    with col_processing_1:
        st.subheader('Compute Options')
        st.session_state.config['leafmachine']['project']['num_workers'] = st.number_input("Number of CPU workers", value=st.session_state.config['leafmachine']['project'].get('num_workers', 1), disabled=True)
        st.session_state.config['leafmachine']['project']['batch_size'] = st.number_input("Batch size", value=st.session_state.config['leafmachine']['project'].get('batch_size', 500), help='Sets the batch size for the LeafMachine2 cropping. If computer RAM is filled, lower this value to ~100.')
    with col_processing_2:
        st.subheader('Misc')
        st.session_state.config['leafmachine']['project']['prefix_removal'] = st.text_input("Remove prefix from catalog number", st.session_state.config['leafmachine']['project'].get('prefix_removal', ''))
        st.session_state.config['leafmachine']['project']['suffix_removal'] = st.text_input("Remove suffix from catalog number", st.session_state.config['leafmachine']['project'].get('suffix_removal', ''))
        st.session_state.config['leafmachine']['project']['catalog_numerical_only'] = st.checkbox("Require 'Catalog Number' to be numerical only", st.session_state.config['leafmachine']['project'].get('catalog_numerical_only', True))
    
    ### Logging and Image Validation - col_v1
    st.header('Logging and Image Validation')    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.session_state.config['leafmachine']['do']['check_for_illegal_filenames'] = st.checkbox("Check for illegal filenames", st.session_state.config['leafmachine']['do'].get('check_for_illegal_filenames', True))
        st.session_state.config['leafmachine']['do']['check_for_corrupt_images_make_vertical'] = st.checkbox("Check for corrupt images", st.session_state.config['leafmachine']['do'].get('check_for_corrupt_images_make_vertical', True))
        
        st.session_state.config['leafmachine']['print']['verbose'] = st.checkbox("Print verbose", st.session_state.config['leafmachine']['print'].get('verbose', True))
        st.session_state.config['leafmachine']['print']['optional_warnings'] = st.checkbox("Show optional warnings", st.session_state.config['leafmachine']['print'].get('optional_warnings', True))

    with col_v2:
        log_level = st.session_state.config['leafmachine']['logging'].get('log_level', None)
        log_level_display = log_level if log_level is not None else 'default'
        selected_log_level = st.selectbox("Logging Level", ['default', 'DEBUG', 'INFO', 'WARNING', 'ERROR'], index=['default', 'DEBUG', 'INFO', 'WARNING', 'ERROR'].index(log_level_display))
        
        if selected_log_level == 'default':
            st.session_state.config['leafmachine']['logging']['log_level'] = None
        else:
            st.session_state.config['leafmachine']['logging']['log_level'] = selected_log_level

def content_tab_domain():
    st.header('Embeddings Database')
    col_emb_1, col_emb_2 = st.columns([4,2])  
    with col_emb_1:
        st.markdown(
            """
            VoucherVision includes the option of using domain knowledge inside of the dynamically generated prompts. The OCR text is queried against a database of existing label transcriptions. The most similar existing transcriptions act as an example of what the LLM should emulate and are shown to the LLM as JSON objects. VoucherVision uses cosine similarity search to return the most similar existing transcription.
            - Note: Using domain knowledge may increase the chance that foreign text is included in the final transcription  
            - Disabling this feature will show the LLM multiple examples of an empty JSON skeleton structure instead
            - Enabling this option requires a GPU with at least 8GB of VRAM
            - The domain knowledge files can be located in the directory "../VoucherVision/domain_knowledge". On first run the embeddings database must be created, which takes time. If the database creation runs each time you use VoucherVision, then something is wrong.
            """
            )
            
        st.write(f"Domain Knowledge is only available for the following prompts:")
        for available_prompts in PROMPTS_THAT_NEED_DOMAIN_KNOWLEDGE:
            st.markdown(f"- {available_prompts}")
        
        if st.session_state.config['leafmachine']['project']['prompt_version'] in PROMPTS_THAT_NEED_DOMAIN_KNOWLEDGE:
            st.session_state.config['leafmachine']['project']['use_domain_knowledge'] = st.checkbox("Use domain knowledge", True, disabled=True)
        else:
            st.session_state.config['leafmachine']['project']['use_domain_knowledge'] = st.checkbox("Use domain knowledge", False, disabled=True)

        st.write("")
        if st.session_state.config['leafmachine']['project']['use_domain_knowledge']:
            st.session_state.config['leafmachine']['project']['embeddings_database_name'] = st.text_input("Embeddings database name (only use underscores)", st.session_state.config['leafmachine']['project'].get('embeddings_database_name', ''))
            st.session_state.config['leafmachine']['project']['build_new_embeddings_database'] = st.checkbox("Build *new* embeddings database", st.session_state.config['leafmachine']['project'].get('build_new_embeddings_database', False))
            st.session_state.config['leafmachine']['project']['path_to_domain_knowledge_xlsx'] = st.text_input("Path to domain knowledge CSV file (will be used to create new embeddings database)", st.session_state.config['leafmachine']['project'].get('path_to_domain_knowledge_xlsx', ''))
        else:
            st.session_state.config['leafmachine']['project']['embeddings_database_name'] = st.text_input("Embeddings database name (only use underscores)", st.session_state.config['leafmachine']['project'].get('embeddings_database_name', ''), disabled=True)
            st.session_state.config['leafmachine']['project']['build_new_embeddings_database'] = st.checkbox("Build *new* embeddings database", st.session_state.config['leafmachine']['project'].get('build_new_embeddings_database', False), disabled=True)
            st.session_state.config['leafmachine']['project']['path_to_domain_knowledge_xlsx'] = st.text_input("Path to domain knowledge CSV file (will be used to create new embeddings database)", st.session_state.config['leafmachine']['project'].get('path_to_domain_knowledge_xlsx', ''), disabled=True)

def render_expense_report_summary():
    expense_summary = st.session_state.expense_summary
    expense_report = st.session_state.expense_report
    st.header('Expense Report Summary')

    if expense_summary:
        st.metric(label="Total Cost", value=f"${round(expense_summary['total_cost_sum'], 4):,}")
        col1, col2 = st.columns(2)

        # Run count and total costs
        with col1:
            st.metric(label="Run Count", value=expense_summary['run_count'])
            st.metric(label="Tokens In", value=f"{expense_summary['tokens_in_sum']:,}")

        # Token information
        with col2:
            st.metric(label="Total Images", value=expense_summary['n_images_sum'])
            st.metric(label="Tokens Out", value=f"{expense_summary['tokens_out_sum']:,}")


        # Calculate cost proportion per image for each API version
        st.subheader('Average Cost per Image by API Version')
        cost_labels = []
        cost_values = []
        total_images = 0
        cost_per_image_dict = {}
        # Iterate through the expense report to accumulate costs and image counts
        for index, row in expense_report.iterrows():
            api_version = row['api_version']
            total_cost = row['total_cost']
            n_images = row['n_images']
            total_images += n_images  # Keep track of total images processed
            if api_version not in cost_per_image_dict:
                cost_per_image_dict[api_version] = {'total_cost': 0, 'n_images': 0}
            cost_per_image_dict[api_version]['total_cost'] += total_cost
            cost_per_image_dict[api_version]['n_images'] += n_images

        api_versions = list(cost_per_image_dict.keys())
        colors = [COLORS_EXPENSE_REPORT[version] if version in COLORS_EXPENSE_REPORT else '#DDDDDD' for version in api_versions]
        
        # Calculate the cost per image for each API version
        for version, cost_data in cost_per_image_dict.items():
            total_cost = cost_data['total_cost']
            n_images = cost_data['n_images']
            # Calculate the cost per image for this version
            cost_per_image = total_cost / n_images if n_images > 0 else 0
            cost_labels.append(version)
            cost_values.append(cost_per_image)
        # Generate the pie chart
        cost_pie_chart = go.Figure(data=[go.Pie(labels=cost_labels, values=cost_values, hole=.3)])
        # Update traces for custom text in hoverinfo, displaying cost with a dollar sign and two decimal places
        cost_pie_chart.update_traces(
            marker=dict(colors=colors),
            text=[f"${value:.2f}" for value in cost_values],  # Formats the cost as a string with a dollar sign and two decimals
            textinfo='percent+label',
            hoverinfo='label+percent+text'  # Adds custom text (formatted cost) to the hover information
        )
        st.plotly_chart(cost_pie_chart, use_container_width=True)



        st.subheader('Proportion of Total Cost by API Version')
        cost_labels = []
        cost_proportions = []
        total_cost_by_version = {}
        # Sum the total cost for each API version
        for index, row in expense_report.iterrows():
            api_version = row['api_version']
            total_cost = row['total_cost']
            if api_version not in total_cost_by_version:
                total_cost_by_version[api_version] = 0
            total_cost_by_version[api_version] += total_cost
        # Calculate the combined total cost for all versions
        combined_total_cost = sum(total_cost_by_version.values())
        # Calculate the proportion of total cost for each API version
        for version, total_cost in total_cost_by_version.items():
            proportion = (total_cost / combined_total_cost) * 100 if combined_total_cost > 0 else 0
            cost_labels.append(version)
            cost_proportions.append(proportion)
        # Generate the pie chart
        cost_pie_chart = go.Figure(data=[go.Pie(labels=cost_labels, values=cost_proportions, hole=.3)])
        # Update traces for custom text in hoverinfo
        cost_pie_chart.update_traces(
            marker=dict(colors=colors),
            text=[f"${cost:.2f}" for cost in total_cost_by_version.values()],  # This will format the cost to 2 decimal places
            textinfo='percent+label',
            hoverinfo='label+percent+text'  # This tells Plotly to show the label, percent, and custom text (cost) on hover
        )
        st.plotly_chart(cost_pie_chart, use_container_width=True)

        # API version usage percentages pie chart
        st.subheader('Runs by API Version')
        api_versions = list(expense_summary['api_version_percentages'].keys())
        percentages = [expense_summary['api_version_percentages'][version] for version in api_versions]
        pie_chart = go.Figure(data=[go.Pie(labels=api_versions, values=percentages, hole=.3)])
        pie_chart.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        pie_chart.update_traces(marker=dict(colors=colors),)
        st.plotly_chart(pie_chart, use_container_width=True)

    else:
        st.error('No expense report data available.')

def sidebar_content():
    try:
        st.session_state.expense_summary, st.session_state.expense_report = summarize_expense_report(os.path.join(st.session_state.dir_home,'expense_report','expense_report.csv'))
        render_expense_report_summary()  
    except:
        st.header('Expense Report Summary')
        st.write('Available after first run...')
        
    # # Check if the expense summary is available in the session state
    # if 'expense' not in st.session_state or st.session_state.expense is None:
    #     st.sidebar.write('No expense report data available.')
    #     return
    
    # # Retrieve the expense report summary
    # expense_summary = st.session_state.expense

    # # Display the expense report summary
    # st.sidebar.markdown('**Run Count**: ' + str(expense_summary['run_count']))

    # # API version usage percentages
    # st.sidebar.markdown('**API Version Usage**:')
    # for version, percentage in expense_summary['api_version_percentages'].items():
    #     st.sidebar.markdown(f'- {version}: {percentage:.2f}%')

    # # Summary of costs and tokens
    # st.sidebar.markdown('**Total Cost**: $' + str(round(expense_summary['total_cost_sum'], 4)))
    # st.sidebar.markdown('**Tokens In**: ' + str(expense_summary['tokens_in_sum']))
    # st.sidebar.markdown('**Tokens Out**: ' + str(expense_summary['tokens_out_sum']))
    # # st.sidebar.markdown('**Rate In**: $' + str(round(expense_summary['rate_in_sum'], 2)) + ' per 1000 tokens')
    # # st.sidebar.markdown('**Rate Out**: $' + str(round(expense_summary['rate_out_sum'], 2)) + ' per 1000 tokens')
    # st.sidebar.markdown('**Cost In**: $' + str(round(expense_summary['cost_in_sum'], 4)))
    # st.sidebar.markdown('**Cost Out**: $' + str(round(expense_summary['cost_out_sum'], 4)))

def main():
    with st.sidebar:
        sidebar_content()
    # Main App
    content_header()

    tab_settings, tab_prompt, tab_domain, tab_component, tab_processing, tab_private, tab_delete = st.tabs(["Project Settings", "Prompt Builder", "Domain Knowledge","Component Detector", "Processing Options", "API Keys", "Space-Saver"])

    with tab_settings:
        content_tab_settings()

    with tab_prompt:
        if st.button("Build Custom LLM Prompt"):
            st.session_state.proceed_to_build_llm_prompt = True
            st.rerun()
        
    with tab_component:
        content_tab_component()

    with tab_domain:
        content_tab_domain()

    with tab_processing:
        content_tab_processing()

    with tab_private:
        if st.button("Edit API Keys"):
            st.session_state.proceed_to_private = True
            st.rerun()

    with tab_delete:
        create_space_saver()

st.set_page_config(layout="wide", page_icon='img/icon.ico', page_title='VoucherVision')

# Default YAML file path
if 'config' not in st.session_state:
    st.session_state.config, st.session_state.dir_home = build_VV_config()
    setup_streamlit_config(st.session_state.dir_home)

if 'proceed_to_main' not in st.session_state:
    st.session_state.proceed_to_main = False  # New state variable to control the flow

if 'proceed_to_build_llm_prompt' not in st.session_state:
    st.session_state.proceed_to_build_llm_prompt = False  # New state variable to control the flow
if 'proceed_to_private' not in st.session_state:
    st.session_state.proceed_to_private = False  # New state variable to control the flow

if 'private_file' not in st.session_state:
    st.session_state.private_file = does_private_file_exist()
    if st.session_state.private_file:
        st.session_state.proceed_to_main = True

# Initialize session_state variables if they don't exist
if 'prompt_info' not in st.session_state:
    st.session_state['prompt_info'] = {}
if 'rules' not in st.session_state:
    st.session_state['rules'] = {}

if not st.session_state.private_file:
    create_private_file()
elif st.session_state.proceed_to_build_llm_prompt:
    build_LLM_prompt_config()
elif st.session_state.proceed_to_private:
    create_private_file()
elif st.session_state.proceed_to_main:
    main()