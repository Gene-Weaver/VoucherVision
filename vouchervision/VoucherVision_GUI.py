import streamlit as st
import yaml, os, json, random, time, re, torch, random, warnings, uuid
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
from itertools import chain
from PIL import Image
import pandas as pd
from typing import Union
from streamlit_extras.let_it_rain import rain
from annotated_text import annotated_text
from vouchervision.LeafMachine2_Config_Builder import write_config_file
from vouchervision.VoucherVision_Config_Builder import build_VV_config, run_demo_tests_GPT, run_demo_tests_Palm , TestOptionsGPT, TestOptionsPalm, check_if_usable, run_api_tests
from vouchervision.vouchervision_main import voucher_vision, voucher_vision_OCR_test
from vouchervision.general_utils import test_GPU, get_cfg_from_full_path, summarize_expense_report, create_google_ocr_yaml_config, validate_dir
from vouchervision.model_maps import ModelMaps
from vouchervision.API_validation import APIvalidation

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
        self.current_overall_step = 0
        self.overall_bar.progress(0)
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

class JSONReport:
    def __init__(self, col_updates, col_json, col_json_WFO, col_json_GEO, col_json_map):
        self.plant_list = [':evergreen_tree:', ':deciduous_tree:',':palm_tree:',
                      ':maple_leaf:',':fallen_leaf:',':mushroom:',':leaves:',
                      ':cactus:',':seedling:',':tulip:',':sunflower:',':hibiscus:',
                      ':cherry_blossom:',':rose:',]
        self.location_list = [':earth_africa:',':earth_americas:',':earth_asia:',]
        self.book_list = [':bookmark_tabs:',':ledger:',':notebook:',':clipboard:',':scroll:',
                          ':notebook_with_decorative_cover:',':green_book:',':blue_book:',
                          ':open_book:',':closed_book:',':book:',
                          ':orange_book:',':books:',':memo:',':pencil:',
                          ]

        # Create placeholders for each JSON component
        self.col_updates = col_updates
        self.col_json = col_json
        self.col_json_WFO = col_json_WFO
        self.col_json_GEO = col_json_GEO
        self.col_json_map = col_json_map

        self.update_main = col_updates.empty()

        self.update_left = col_json.empty()
        self.header_json = col_json.empty()
        self.json_placeholder = col_json.empty()

        self.update_middle = col_json_WFO.empty()
        self.header_json_WFO = col_json_WFO.empty()
        self.json_WFO_placeholder = col_json_WFO.empty()

        self.update_right = col_json_GEO.empty()
        self.header_json_GEO = col_json_GEO.empty()
        self.json_GEO_placeholder = col_json_GEO.empty()

        self.update_map = col_json_map.empty()
        self.header_json_map = col_json_map.empty()
        self.json_map = col_json_map.empty()


        self.json = None
        self.json_WFO = None
        self.json_GEO = None

        self.text_main = ''
        self.text_middle = ''
        self.text_right = ''

        self.header_text_main = None
        self.header_text_middle = None
        self.header_text_right = None

    
    def set_JSON(self, json_main, json_WFO, json_GEO):
        i_plant = random.randint(0,len(self.plant_list)-1)
        i_location = random.randint(0,len(self.location_list)-1)
        i_book = random.randint(0,len(self.book_list)-1)
        self.json = json_main
        self.json_WFO = json_WFO
        self.json_GEO = json_GEO

        # Update placeholders with new JSON data
        self.header_text_main = None
        self.header_text_middle = None
        self.header_text_right = None

        self.update_main.subheader(f':loudspeaker: {self.text_main}')
        self.update_left.subheader(f'{self.book_list[i_book]}', divider='rainbow')
        self.update_middle.subheader(f'{self.plant_list[i_plant]}', divider='rainbow')
        self.update_right.subheader(f'{self.location_list[i_location]}', divider='rainbow')
        self.update_map.subheader(f':world_map:', divider='rainbow')

        self.header_json.markdown('**LLM-derived information from the OCR text**')
        self.header_json_WFO.markdown('World Flora Online')
        self.header_json_GEO.markdown('Geolocate')
        self.header_json_map.markdown(f':large_purple_circle: :violet[Geolocated]  :large_green_circle: :green[From OCR Text]')

        self.json_placeholder.json(self.json)
        self.json_WFO_placeholder.json(self.json_WFO)
        self.json_GEO_placeholder.json(self.json_GEO)

        # If GEO data is available, plot on the map
        # Clear the existing content in the map placeholder
        # Clear the existing content in the map placeholder
        self.json_map.empty()
        map_points = []
        map_data = []
        # Function to safely convert to float
        def safe_float_convert(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        # Check and process first point's data
        lat = safe_float_convert(self.json_GEO.get("GEO_decimal_lat")) if self.json_GEO else None
        lon = safe_float_convert(self.json_GEO.get("GEO_decimal_long")) if self.json_GEO else None

        if lat is not None and lon is not None:
            map_points.append({'lat': lat, 'lon': lon, 'color': '#8800ff' , 'size': [50000]})

        # Check and process second point's data
        lat_verbatim = safe_float_convert(self.json.get("decimalLatitude")) if self.json else None
        lon_verbatim = safe_float_convert(self.json.get("decimalLongitude")) if self.json else None

        if lat_verbatim is not None and lon_verbatim is not None:
            map_points.append({'lat': lat_verbatim, 'lon': lon_verbatim, 'color': '#00c227' , 'size': [25000]})

        # Convert the list of points to a DataFrame
        map_data = pd.DataFrame(map_points)

        # Display the map if map_data is not empty
        if not map_data.empty:
            with self.json_map:
                st.map(map_data, zoom=4, size='size', color='color')

    def set_text(self, text_main=None, text_middle=None, text_right=None):
        if text_main:
            self.text_main = text_main
            self.update_main.subheader(f':loudspeaker: {self.text_main}')
        if text_middle:
            self.text_middle = text_middle
            self.update_middle.subheader('', divider='rainbow')
        if text_right:
            self.text_right = text_right
            self.update_right.subheader(self.text_right, divider='rainbow')

    def clear_JSON(self):
        self.json = None
        self.json_WFO = None
        self.json_GEO = None

        # Clear the content in the placeholders
        self.json_placeholder.empty()
        self.json_WFO_placeholder.empty()
        self.json_GEO_placeholder.empty()

    def format_json(self, json_obj):
        try:
            return json.dumps(json.loads(json_obj), indent=4, sort_keys=False)
        except:
            return json.dumps(json_obj, indent=4, sort_keys=False)

    
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
                formatted_json = json.dumps(JSON_results[idx], indent=4, sort_keys=False)
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

def refresh():
    st.write('')

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

def format_json(json_obj):
    try:
        return json.dumps(json.loads(json_obj), indent=4, sort_keys=False)
    except:
        return json.dumps(json_obj, indent=4, sort_keys=False)
    
def get_prompt_versions(LLM_version):
    yaml_files = [f for f in os.listdir(os.path.join(st.session_state.dir_home, 'custom_prompts')) if f.endswith('.yaml')]

    return yaml_files

def get_private_file():
    dir_home = os.path.dirname(os.path.dirname(__file__))
    path_cfg_private = os.path.join(dir_home, 'PRIVATE_DATA.yaml')
    return get_cfg_from_full_path(path_cfg_private)



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
        st.session_state['prompt_author'] = st.session_state['prompt_info'].get('prompt_author', st.session_state['default_prompt_author']) 
        st.session_state['prompt_author_institution'] = st.session_state['prompt_info'].get('prompt_author_institution', st.session_state['default_prompt_author_institution']) 
        st.session_state['prompt_name'] = st.session_state['prompt_info'].get('prompt_name', st.session_state['default_prompt_name']) 
        st.session_state['prompt_version'] = st.session_state['prompt_info'].get('prompt_version', st.session_state['default_prompt_version']) 
        st.session_state['prompt_description'] = st.session_state['prompt_info'].get('prompt_description', st.session_state['default_prompt_description']) 
        st.session_state['instructions'] = st.session_state['prompt_info'].get('instructions', st.session_state['default_instructions']) 
        st.session_state['json_formatting_instructions'] = st.session_state['prompt_info'].get('json_formatting_instructions', st.session_state['default_json_formatting_instructions'] )
        st.session_state['rules'] = st.session_state['prompt_info'].get('rules', {})
        st.session_state['mapping'] = st.session_state['prompt_info'].get('mapping', {})
        st.session_state['LLM'] = st.session_state['prompt_info'].get('LLM', 'General Purpose')

        # Placeholder:
        st.session_state['assigned_columns'] = list(chain.from_iterable(st.session_state['mapping'].values())) 

def save_prompt_yaml(filename):
    yaml_content = {
        'prompt_author': st.session_state['prompt_author'],
        'prompt_author_institution': st.session_state['prompt_author_institution'],
        'prompt_name': st.session_state['prompt_name'],
        'prompt_version': st.session_state['prompt_version'],
        'prompt_description': st.session_state['prompt_description'],
        'LLM': st.session_state['LLM'],
        'instructions': st.session_state['instructions'],
        'json_formatting_instructions': st.session_state['json_formatting_instructions'],
        'rules': st.session_state['rules'],
        'mapping': st.session_state['mapping'],
    }
    
    dir_prompt = os.path.join(st.session_state.dir_home, 'custom_prompts')
    filepath = os.path.join(dir_prompt, f"{filename}.yaml")

    with open(filepath, 'w') as file:
        yaml.safe_dump(dict(yaml_content), file, sort_keys=False)

    st.success(f"Prompt saved as '{filename}.yaml'.")

def check_unique_mapping_assignments():
    print(st.session_state['assigned_columns'])
    if len(st.session_state['assigned_columns']) != len(set(st.session_state['assigned_columns'])):
        st.error("Each column name must be assigned to only one category.")
        return False
    elif not st.session_state['assigned_columns']:
        st.error("No columns have been mapped.")
        return False
    elif  len(st.session_state['assigned_columns']) != len(st.session_state['rules'].keys()):
        incomplete = [item for item in list(st.session_state['rules'].keys()) if item not in st.session_state['assigned_columns']]
        st.warning(f"These columns have been mapped: {st.session_state['assigned_columns']}")
        st.error(f"However, these columns must be mapped before the prompt is complete: {incomplete}")
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
        st.session_state['prompt_author'] = st.session_state['default_prompt_author']
        st.session_state['prompt_author_institution'] = st.session_state['default_prompt_author_institution']
        st.session_state['prompt_name'] = st.session_state['prompt_name']
        st.session_state['prompt_version'] = st.session_state['prompt_version']
        st.session_state['prompt_description'] = st.session_state['default_prompt_description']
        st.session_state['instructions'] = st.session_state['default_instructions']
        st.session_state['json_formatting_instructions'] = st.session_state['default_json_formatting_instructions'] 
        st.session_state['rules'] = {}
        st.session_state['LLM'] = 'General Purpose'
        
        st.session_state['assigned_columns'] = []

        st.session_state['prompt_info'] = {
            'prompt_author': st.session_state['prompt_author'],
            'prompt_author_institution': st.session_state['prompt_author_institution'],
            'prompt_name': st.session_state['prompt_name'],
            'prompt_version': st.session_state['prompt_version'],
            'prompt_description': st.session_state['prompt_description'],
            'instructions': st.session_state['instructions'],
            'json_formatting_instructions': st.session_state['json_formatting_instructions'],
            'rules': st.session_state['rules'],
            'mapping': st.session_state['mapping'],
            'LLM': st.session_state['LLM']
        }

def build_LLM_prompt_config():
    col_main1, col_main2 = st.columns([10,2])
    with col_main1:
        st.session_state.logo_path = os.path.join(st.session_state.dir_home, 'img','logo.png')
        st.session_state.logo = Image.open(st.session_state.logo_path)
        st.image(st.session_state.logo, width=250)
    with col_main2:
        if st.button('Exit',key='exist button 2'):
                st.session_state.proceed_to_build_llm_prompt = False
                st.session_state.proceed_to_main = True
                st.rerun()

    st.session_state['assigned_columns'] = []
    st.session_state['default_prompt_author'] = 'unknown'
    st.session_state['default_prompt_author_institution'] = 'unknown'
    st.session_state['default_prompt_name'] = 'custom_prompt'
    st.session_state['default_prompt_version'] = 'v-1-0'
    st.session_state['default_prompt_author_institution'] = 'unknown'
    st.session_state['default_prompt_description'] = 'unknown'
    st.session_state['default_LLM'] = 'General Purpose'
    st.session_state['default_instructions'] = """1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below.
2. Map the unstructured OCR text to the appropriate JSON key and populate the field given the user-defined rules.
3. JSON key values are permitted to remain empty strings if the corresponding information is not found in the unstructured OCR text.
4. Duplicate dictionary fields are not allowed.
5. Ensure all JSON keys are in camel case.
6. Ensure new JSON field values follow sentence case capitalization.
7. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
8. Ensure output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
9. Only return a JSON dictionary represented as a string. You should not explain your answer."""
    st.session_state['default_json_formatting_instructions'] = """This section provides rules for formatting each JSON value organized by the JSON key."""
    
    # Start building the Streamlit app
    col_prompt_main_left, ___, col_prompt_main_right = st.columns([6,1,3])

    
    with col_prompt_main_left:
        
        st.title("Custom LLM Prompt Builder")
        st.subheader('About')
        st.write("This form allows you to craft a prompt for your specific task. You can also edit the JSON yaml files directly, but please try loading the prompt back into this form to ensure that the formatting is correct. If this form cannot load your manually edited JSON yaml file, then it will not work in VoucherVision.")
        st.subheader(':rainbow[How it Works]')
        st.write("1. Edit this page until you are happy with your instructions. We recommend looking at the basic structure, writing down your prompt inforamtion in a Word document so that it does not randomly disappear, and then copying and pasting that info into this form once your whole prompt structure is defined.")
        st.write("2. After you enter all of your prompt instructions, click 'Save' and give your file a name.")
        st.write("3. This file will be saved as a yaml configuration file in the `..VoucherVision/custom_prompts` folder.")
        st.write("4. When you go back the main VoucherVision page you will now see your custom prompt available in the 'Prompt Version' dropdown menu.")
        
        st.write("---")
        st.header('Load an Existing Prompt Template')
        st.write("By default, this form loads the minimum required transcription fields but does not provide rules for each field. You can also load an existing prompt as a template, editing or deleting values as needed.")

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
                
        
        # Prompt Author Information
        st.write("---")
        st.header("Prompt Author Information")
        st.write("We value community contributions! Please provide your name(s) (or pseudonym if you prefer) for credit. If you leave this field blank, it will say 'unknown'.")
        if 'prompt_author' not in st.session_state:# != st.session_state['default_prompt_author']:
            st.session_state['prompt_author'] = st.text_input("Enter names of prompt author(s)", value=st.session_state['default_prompt_author'],key=uuid.uuid4())
        else:
            st.session_state['prompt_author'] = st.text_input("Enter names of prompt author(s)", value=st.session_state['prompt_author'],key=uuid.uuid4())

        # Institution 
        st.write("Please provide your institution name. If you leave this field blank, it will say 'unknown'.")
        if 'prompt_author_institution' not in st.session_state:
            st.session_state['prompt_author_institution'] = st.text_input("Enter name of institution", value=st.session_state['default_prompt_author_institution'],key=uuid.uuid4())
        else:
            st.session_state['prompt_author_institution'] = st.text_input("Enter name of institution", value=st.session_state['prompt_author_institution'],key=uuid.uuid4())

        # Prompt name 
        st.write("Please provide a simple name for your prompt. If you leave this field blank, it will say 'custom_prompt'.")
        if 'prompt_name' not in st.session_state:
            st.session_state['prompt_name'] = st.text_input("Enter prompt name", value=st.session_state['default_prompt_name'],key=uuid.uuid4())
        else:
            st.session_state['prompt_name'] = st.text_input("Enter prompt name", value=st.session_state['prompt_name'],key=uuid.uuid4())
      
        # Prompt verion  
        st.write("Please provide a version identifier for your prompt. If you leave this field blank, it will say 'v-1-0'.")
        if 'prompt_version' not in st.session_state:
            st.session_state['prompt_version'] = st.text_input("Enter prompt version", value=st.session_state['default_prompt_version'],key=uuid.uuid4())
        else:
            st.session_state['prompt_version'] = st.text_input("Enter prompt version", value=st.session_state['prompt_version'],key=uuid.uuid4())
            
       
        st.write("Please provide a description of your prompt and its intended task. Is it designed for a specific collection? Taxa? Database structure?")
        if 'prompt_description' not in st.session_state:
            st.session_state['prompt_description'] = st.text_input("Enter description of prompt", value=st.session_state['default_prompt_description'],key=uuid.uuid4())
        else:
            st.session_state['prompt_description'] = st.text_input("Enter description of prompt", value=st.session_state['prompt_description'],key=uuid.uuid4())
       
        st.write('---')
        st.header("Set LLM Model Type")
        # Define the options for the dropdown
        llm_options_general = ["General Purpose",
                       "OpenAI GPT Models","Google PaLM2 Models","Google Gemini Models","MistralAI Models",]
        llm_options_all = ModelMaps.get_models_gui_list()

        if 'LLM' not in st.session_state:
            st.session_state['LLM'] = st.session_state['default_LLM']
    
        if st.session_state['LLM']:
            llm_options = llm_options_general + llm_options_all + [st.session_state['LLM']]
        else:
            llm_options = llm_options_general + llm_options_all
        # Create the dropdown and set the value to session_state['LLM']
        st.write("Which LLM is this prompt designed for? This will not restrict its use to a specific LLM, but some prompts will behave differently across models.")
        st.write("SLTPvA prompts have been validated with all supported LLMs, but perfornce may vary. If you design a prompt to work best with a specific model, then you can indicate the model here.")
        st.write("For general purpose prompts (like the SLTPvA prompts) just use the 'General Purpose' option.")
        st.session_state['LLM'] = st.selectbox('Set LLM', llm_options, index=llm_options.index(st.session_state.get('LLM', 'General Purpose')))

        st.write('---')
        # Instructions Section
        st.header("Instructions")
        st.write("These are the general instructions that guide the LLM through the transcription task. We recommend using the default instructions unless you have a specific reason to change them.")
        
        if 'instructions' not in st.session_state:
            st.session_state['instructions'] = st.text_area("Enter guiding instructions", value=st.session_state['default_instructions'].strip(), height=350,key=uuid.uuid4())
        else:
            st.session_state['instructions'] = st.text_area("Enter guiding instructions", value=st.session_state['instructions'].strip(), height=350,key=uuid.uuid4())


        st.write('---')

        # Column Instructions Section
        st.header("JSON Formatting Instructions")
        st.write("The following section tells the LLM how we want to structure the JSON dictionary. We do not recommend changing this section because it would likely result in unstable and inconsistent behavior.")
        if 'json_formatting_instructions' not in st.session_state:
            st.session_state['json_formatting_instructions'] = st.text_area("Enter general JSON guidelines", value=st.session_state['default_json_formatting_instructions'],key=uuid.uuid4())
        else:
            st.session_state['json_formatting_instructions'] = st.text_area("Enter general JSON guidelines", value=st.session_state['json_formatting_instructions'],key=uuid.uuid4())
        





        st.write('---')
        col_left, col_right = st.columns([6,4])

        null_value_rules = ''
        c_name = "EXAMPLE_COLUMN_NAME"
        c_value = "REPLACE WITH DESCRIPTION"

        with col_left:
            st.subheader('Add/Edit Columns')
            st.markdown("The pre-populated fields are REQUIRED for downstream validation steps. They must be in all prompts.")
            
            # Initialize rules in session state if not already present
            if 'rules' not in st.session_state or not st.session_state['rules']:
                for required_col in st.session_state['required_fields']:
                    st.session_state['rules'][required_col] = c_value

            


            # Layout for adding a new column name
            # col_text, col_textbtn = st.columns([8, 2])
            # with col_text:
            st.session_state['new_column_name'] = st.text_input("Enter a new column name:")
            # with col_textbtn:
            # st.write('##')
            if st.button("Add New Column") and st.session_state['new_column_name']:
                if st.session_state['new_column_name'] not in st.session_state['rules']:
                    st.session_state['rules'][st.session_state['new_column_name']] = c_value
                    st.success(f"New column '{st.session_state['new_column_name']}' added. Now you can edit its properties.")
                    st.session_state['new_column_name'] = ''
                else:
                    st.error("Column name already exists. Please enter a unique column name.")
                    st.session_state['new_column_name'] = ''
            

            # Get columns excluding the protected "catalogNumber"
            st.write('#')
            # required_columns = [col for col in st.session_state['rules'] if col not in st.session_state['required_fields']]
            editable_columns = [col for col in st.session_state['rules'] if col not in ["catalogNumber"]]
            removable_columns = [col for col in st.session_state['rules'] if col not in st.session_state['required_fields']]

            st.session_state['current_rule'] = st.selectbox("Select a column to edit:", [""] + editable_columns)
            # column_name = st.selectbox("Select a column to edit:", editable_columns)

            

            # if 'current_rule' not in st.session_state:
            #     st.session_state['current_rule'] = current_rule


            


            # Form for input fields
            with st.form(key='rule_form'):
                # format_options = ["verbatim transcription", "spell check transcription", "boolean yes no", "boolean 1 0", "integer", "[list]", "yyyy-mm-dd"]
                # current_rule["format"] = st.selectbox("Format:", format_options, index=format_options.index(current_rule["format"]) if current_rule["format"] else 0)
                # current_rule["null_value"] = st.text_input("Null value:", value=current_rule["null_value"])
                if st.session_state['current_rule']:
                    current_rule_description = st.text_area("Description of category:", value=st.session_state['rules'][st.session_state['current_rule']])
                else:
                    current_rule_description = ''
                commit_button = st.form_submit_button("Commit Column")

            # default_rule = {
            #     "format": format_options[0],  # default format
            #     "null_value": "",  # default null value
            #     "description": "",  # default description
            # }
            # if st.session_state['current_rule'] != st.session_state['current_rule']:
            #     # Column has changed. Update the session_state selected column.
            #     st.session_state['current_rule'] = st.session_state['current_rule']
                # # Reset the current rule to the default for this new column, or a blank rule if not set.
                # current_rule = st.session_state['rules'][st.session_state['current_rule']].get(current_rule, c_value)

            # Handle commit action
            if commit_button and st.session_state['current_rule']:
                # Commit the rules to the session state.
                st.session_state['rules'][st.session_state['current_rule']] = current_rule_description
                st.success(f"Column '{st.session_state['current_rule']}' added/updated in rules.")

                # Force the form to reset by clearing the fields from the session state
                st.session_state.pop('current_rule', None)  # Clear the selected column to force reset

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
            delete_column_name = st.selectbox("Select a column to delete:", [""] + removable_columns)
            # with del_colbtn:
            # st.write('##')
            if st.button("Delete Column") and delete_column_name:
                del st.session_state['rules'][delete_column_name]
                st.success(f"Column '{delete_column_name}' removed from rules.")


            

        with col_right:
            # Display the current state of the JSON rules
            st.subheader('Formatted Columns')
            st.json(st.session_state['rules'])

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
            all_column_names = list(st.session_state['rules'].keys())

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
            'prompt_author': st.session_state['prompt_author'],
            'prompt_author_institution': st.session_state['prompt_author_institution'],
            'prompt_name': st.session_state['prompt_name'],
            'prompt_version': st.session_state['prompt_version'],
            'prompt_description': st.session_state['prompt_description'],
            'LLM': st.session_state['LLM'],
            'instructions': st.session_state['instructions'],
            'json_formatting_instructions': st.session_state['json_formatting_instructions'],
            'rules': st.session_state['rules'],
            'mapping': st.session_state['mapping'],
        }
        st.json(st.session_state['prompt_info'])

def show_header_welcome():
    st.session_state.logo_path = os.path.join(st.session_state.dir_home, 'img','logo.png')
    st.session_state.logo = Image.open(st.session_state.logo_path)
    st.image(st.session_state.logo, width=250)

def determine_n_images():
    try:
        # Check if 'dir_uploaded_images' key exists and it is not empty
        if 'dir_uploaded_images' in st and st['dir_uploaded_images']:
            dir_path = st['dir_uploaded_images']  # This would be the path to the directory
            return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        else:
            return None
    except:
        return None

def save_api_status(present_keys, missing_keys, date_of_check):
    with open(os.path.join(st.session_state.dir_home,'api_status.yaml'), 'w') as file:
        yaml.dump({'present_keys': present_keys, 'missing_keys': missing_keys, "date": date_of_check}, file)

def load_api_status():
    try:
        with open(os.path.join(st.session_state.dir_home,'api_status.yaml'), 'r') as file:
            status = yaml.safe_load(file)
            return status.get('present_keys', []), status.get('missing_keys', []), status.get('date', [])
    except FileNotFoundError:
        return None, None, None
    
def display_api_key_status():
    if not st.session_state['API_checked']:
        present_keys, missing_keys, date_of_check = load_api_status()
        if present_keys is None and missing_keys is None:
            st.session_state['API_checked'] = False
        else:
            # Convert keys to annotations (similar to what you do in check_api_key_status)
            present_annotations = [(key, " ", "#059c1b") for key in present_keys]  # Adjust as needed
            missing_annotations = [(key, " ", "#525252") for key in missing_keys]  # Adjust as needed

            st.session_state['present_annotations'] = present_annotations
            st.session_state['missing_annotations'] = missing_annotations
            st.session_state['date_of_check'] = date_of_check
            st.session_state['API_checked'] = True

    # Check if the API status has already been retrieved
    if 'API_checked' not in st.session_state or not st.session_state['API_checked'] or st.session_state['API_rechecked']:
        st.session_state['present_annotations'], st.session_state['missing_annotations'], st.session_state['date_of_check'] = check_api_key_status()
        st.session_state['API_checked'] = True
        st.session_state['API_rechecked'] = False

    st.markdown(f"Last checked on {st.session_state['date_of_check']}")
    # Display present keys horizontally
    if 'present_annotations' in st.session_state and st.session_state['present_annotations']:
        annotated_text(*st.session_state['present_annotations'])

    # Display missing keys horizontally
    if 'missing_annotations' in st.session_state and st.session_state['missing_annotations']:
        annotated_text(*st.session_state['missing_annotations'])


def check_api_key_status():
    path_cfg_private = os.path.join(st.session_state.dir_home, 'PRIVATE_DATA.yaml')
    cfg_private = get_cfg_from_full_path(path_cfg_private)

    API_Validator = APIvalidation(cfg_private, st.session_state.dir_home)
    present_keys, missing_keys, date_of_check = API_Validator.report_api_key_status()  # Assuming this function returns two lists

    # Prepare annotations for present keys
    present_annotations = []
    missing_annotations = []
    for key in present_keys:
        if "Valid" in key:
            show_text = key.split('(')[0]
            present_annotations.append((show_text, "ready!", "#059c1b"))  # Green for valid
        elif "Invalid" in key:
            show_text = key.split('(')[0]
            present_annotations.append((show_text, "error", "#870307"))  # Red for invalid

    # Prepare annotations for missing keys
    for key in missing_keys:
        show_text = key.split('(')[0]
        missing_annotations.append((show_text, "n/a", " ", "#c4c4c4"))  # Red for invalid

    # Save API key status
    save_api_status(present_keys, missing_keys, date_of_check) 
    
    return present_annotations, missing_annotations, date_of_check
   

def convert_cost_dict_to_table(cost, name):
    # Convert the dictionary to a pandas DataFrame for nicer display
    df = pd.DataFrame.from_dict(cost, orient='index')
    df.reset_index(inplace=True)
    df.columns = [str(name), 'Input', 'Output'] 


    # Apply color gradient
    cm = sns.light_palette("green", as_cmap=True)
    styled_df = df.style.background_gradient(cmap=cm, subset=['Input', 'Output'])
    return styled_df

def get_all_cost_tables():
    warnings.filterwarnings('ignore', message=".*is_sparse is deprecated.*")
    CostMap = ModelMaps
    cost_names = CostMap.get_all_mapping_cost()

    path_api_cost = os.path.join(st.session_state.dir_home,'api_cost','api_cost.yaml')
    with open(path_api_cost, 'r') as file:
        cost_data = yaml.safe_load(file)

    cost_openai = {}
    cost_azure = {}
    cost_google = {}
    cost_mistral = {}
    cost_local = {}
    for key, value in cost_names.items():
        parts = value.split("_")
        if 'LOCAL' in parts:
            cost_local[key] = cost_data.get(value,'')
        elif 'AZURE' in parts:
            cost_azure[key] = cost_data.get(value,'')
        elif 'GPT' in parts:
            cost_openai[key] = cost_data.get(value,'')
        elif 'PALM2' in parts or 'GEMINI' in parts:
            cost_google[key] = cost_data.get(value,'')
        elif 'MISTRAL' in parts:
            cost_mistral[key] = cost_data.get(value,'')

    styled_cost_openai = convert_cost_dict_to_table(cost_openai, "OpenAI")
    styled_cost_azure = convert_cost_dict_to_table(cost_azure, "OpenAI (Azure Endpoints)")
    styled_cost_google = convert_cost_dict_to_table(cost_google, "Google (VertexAI)")
    styled_cost_mistral = convert_cost_dict_to_table(cost_mistral, "MistralAI")
    styled_cost_local = convert_cost_dict_to_table(cost_local, "Local Models")

    return cost_openai, styled_cost_openai, cost_azure, styled_cost_azure, cost_google, styled_cost_google, cost_mistral, styled_cost_mistral, cost_local, styled_cost_local


def content_header():
    col_logo, col_run_1, col_run_2, col_run_3, col_run_4, col_run_5 = st.columns([2,2,2,2,2,2])

    
    col_test = st.container()

    st.subheader("Overall Progress")
    col_run_info_1 = st.columns([1])[0]
    col_updates_1, col_updates_2 = st.columns([5,1])
    col_json, col_json_WFO, col_json_GEO, col_json_map = st.columns([2, 2, 2, 2])

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
        json_report = JSONReport(col_updates_1, col_json, col_json_WFO, col_json_GEO, col_json_map)
        

    with col_logo:
        show_header_welcome()
    
    with col_run_1:
        # st.subheader('Run VoucherVision')
        N_STEPS = 6

        if determine_n_images():
            st.session_state['processing_add_on'] = f" {determine_n_images()} Images"
        else:
            st.session_state['processing_add_on'] = ''

        if check_if_usable():
            if st.button(f"Start Processing{st.session_state['processing_add_on']}", type='primary',use_container_width=True):
                st.session_state['formatted_json'] = None
                st.session_state['formatted_json_WFO'] = None
                st.session_state['formatted_json_GEO'] = None
                # Define number of overall steps
                progress_report.set_n_overall(N_STEPS)
                progress_report.update_overall(f"Starting VoucherVision...")
            
                # First, write the config file.
                write_config_file(st.session_state.config, st.session_state.dir_home, filename="VoucherVision.yaml")

                path_custom_prompts = os.path.join(st.session_state.dir_home,'custom_prompts',st.session_state.config['leafmachine']['project']['prompt_version'])
                # Call the machine function.
                st.session_state['formatted_json'], st.session_state['formatted_json_WFO'], st.session_state['formatted_json_GEO'], total_cost, n_failed_OCR, n_failed_LLM_calls = voucher_vision(None,
                                                                                                                st.session_state.dir_home, 
                                                                                                                path_custom_prompts, 
                                                                                                                None, 
                                                                                                                progress_report,
                                                                                                                json_report,
                                                                                                                path_api_cost=os.path.join(st.session_state.dir_home,'api_cost','api_cost.yaml'), 
                                                                                                                is_real_run=True)
                
                if n_failed_OCR > 0:
                    st.error(f"Caution:heavy_exclamation_mark: :loudspeaker: {n_failed_LLM_calls} images had a no extractable OCR text :eyes:")

                if n_failed_LLM_calls > 0:
                    st.error(f"Caution:heavy_exclamation_mark: :loudspeaker: {n_failed_LLM_calls} images had a failed LLM API call :eyes:")
                    st.error(f"Make sure that you have access to the chosen LLM API model. Sometimes certain OpenAI accounts do not have access to all models, for example")
                
                if total_cost:
                    st.success(f":money_with_wings: This run cost :heavy_dollar_sign:{total_cost:.4f}")
                else:
                    st.info(f":money_with_wings: This run cost :heavy_dollar_sign:{total_cost:.4f}")
                st.balloons()

        else:
            st.button("Start Processing", type='primary', disabled=True)
            st.error(":heavy_exclamation_mark: Required API keys not set. Please visit the 'API Keys' tab and set the Google Vision OCR API key and at least one LLM key.")
      
        if st.session_state['formatted_json']:
            json_report.set_JSON(st.session_state['formatted_json'], st.session_state['formatted_json_WFO'], st.session_state['formatted_json_GEO'])
   
    with col_run_5:
        with st.expander("View Messages and Updates"):
            st.info("***Note:*** If you use VoucherVision frequently, you can change the default values that are auto-populated in the form below. In a text editor or IDE, edit the first few rows in the file `../VoucherVision/vouchervision/VoucherVision_Config_Builder.py`")

 
    
    with col_run_1:
        ct_left, ct_right = st.columns([1,1])
    with ct_left:
        st.button("Refresh", on_click=refresh, use_container_width=True)
    with ct_right:
        if st.button('FAQs', use_container_width=True):
            pass

    # with col_run_2:
    #     if st.button("Test GPT"):
    #         progress_report.set_n_overall(TestOptionsGPT.get_length())
    #         test_results, JSON_results = run_demo_tests_GPT(progress_report)
    #         with col_test:
    #             display_test_results(test_results, JSON_results, 'gpt')
    #         st.balloons()

    #     if st.button("Test PaLM2"):
    #         progress_report.set_n_overall(TestOptionsPalm.get_length())
    #         test_results, JSON_results = run_demo_tests_Palm(progress_report)
    #         with col_test:
    #             display_test_results(test_results, JSON_results, 'palm')
    #         st.balloons()


    with col_run_2:
        if st.button('Save Current Settings',use_container_width=True):
            if st.session_state.settings_filename:
                config_file_path = os.path.join(st.session_state.dir_home, 'settings', st.session_state['settings_filename'] + '.yaml')
                with open(config_file_path, 'w') as file:
                    yaml.dump(st.session_state.config, file, default_flow_style=False)
                with col_run_4:
                    st.success(f'Current settings saved to {config_file_path}')
            else:
                with col_run_4:
                    st.error('Missing settings file name. Settings not saved.')
                    # st.session_state.config
    with col_run_3:
        st.session_state['settings_filename'] = st.text_input('Setting File Name',placeholder="Settings fileame",label_visibility='collapsed',value=None)



    with col_run_2:
        if st.button('Load Settings',use_container_width=True):
            if st.session_state['loaded_settings_filename']:
                path_load_settings = os.path.join(st.session_state['dir_settings'],st.session_state['loaded_settings_filename'])
                if os.path.exists(path_load_settings) and not None:
                    with open(path_load_settings, 'r') as file:
                        loaded_config = yaml.safe_load(file)
                    st.session_state.config, st.session_state.dir_home = build_VV_config(loaded_cfg=loaded_config)
                    with col_run_4:
                        st.success(f'Loaded settings from {path_load_settings}')
                else:
                    st.error(f'Path to settings file does not exist: {path_load_settings}')
            else:
                with col_run_4:
                    st.warning(f'Filename not selected')


    with col_run_3:
        st.session_state['settings_choice_null'] = 'Select previous settings...'
        st.session_state['dir_settings'] = os.path.join(st.session_state.dir_home, 'settings')
        all_settings_files = [st.session_state['settings_choice_null']] + [f for f in os.listdir(st.session_state['dir_settings']) if f.endswith('.yaml')]
        settings_choice = st.selectbox('Load Previous Settings', all_settings_files,label_visibility='collapsed')
        if settings_choice != st.session_state['settings_choice_null']:
            st.session_state['loaded_settings_filename'] = settings_choice            
        

    with col_run_2:
        if st.button("Check GPU Status",use_container_width=True):
            success, info = test_GPU()

            if success:
                st.balloons()
                with col_run_4:
                    for message in info:
                        st.success(message)
            else:
                with col_run_4:
                    for message in info:
                        st.error(message)











def content_project_settings():
     
    st.header('Project Settings')
    col_project_1, col_project_2 = st.columns([11,1])
    ### Project
    with col_project_1:
        st.session_state.config['leafmachine']['project']['run_name'] = st.text_input("Run name", st.session_state.config['leafmachine']['project'].get('run_name', ''))
        st.session_state.config['leafmachine']['project']['dir_output'] = st.text_input("Output directory", st.session_state.config['leafmachine']['project'].get('dir_output', ''))
    


def content_input_images():
    st.header('Input Images')
    col_local_1, col_local_2 = st.columns([11,1])  
    with col_local_1:
        ### Input Images Local
        st.session_state.config['leafmachine']['project']['dir_images_local'] = st.text_input("Input images directory", st.session_state.config['leafmachine']['project'].get('dir_images_local', ''))
        st.session_state.config['leafmachine']['project']['continue_run_from_partial_xlsx'] = st.text_input("Continue run from partially completed project XLSX", st.session_state.config['leafmachine']['project'].get('continue_run_from_partial_xlsx', ''), disabled=True)
        


def content_llm_cost():
    st.write("---")
    st.header('LLM Cost Calculator')
    # ( n_in/1000 * Input + n_out/1000 * Output ) * n_img = COST
    calculator_1,calculator_2,calculator_3,calculator_4,calculator_5 = st.columns([1,1,1,1,1])     

    st.subheader('Cost Matrix')
    st.markdown('The table shows the cost of each LLM API per 1,000 tokens. An average VoucherVision call uses 2,000 input tokens and receives 500 output tokens.')
    col_cost_1, col_cost_2, col_cost_3, col_cost_4, col_cost_5 = st.columns([1,1,1,1,1])    

    # Load all cost tables if not already done
    if 'all_llm_cost' not in st.session_state:
        st.session_state['all_llm_cost'] = True
        st.session_state['cost_openai'], st.session_state['styled_cost_openai'], st.session_state['cost_azure'], st.session_state['styled_cost_azure'], st.session_state['cost_google'], st.session_state['styled_cost_google'], st.session_state['cost_mistral'], st.session_state['styled_cost_mistral'], st.session_state['cost_local'], st.session_state['styled_cost_local'] = get_all_cost_tables()

    with calculator_1:
        # Combine all model names into a single list
        model_names = []
        for df in [st.session_state['cost_openai'], st.session_state['cost_azure'], st.session_state['cost_google'], st.session_state['cost_mistral'], st.session_state['cost_local']]:
            for key in df.keys():
                model_names.append(key)

        # Create a dropdown for model selection
        selected_model = st.selectbox("Select a model", options=model_names)

    with calculator_2:
        # Create input fields for n_in, n_out, n_img
        n_in = st.number_input("Tokens In", min_value=0, value=2000, step=50)
    with calculator_3:
        n_out = st.number_input("Tokens Out", min_value=0, value=500, step=50)
    with calculator_4:
        n_img = st.number_input("Number of Images", min_value=0, value=1000, step=100)

    # Function to find the model's Input and Output values
    def find_model_values(model, all_dfs):
        for df in all_dfs:
            if model in df.keys():
                return df[model]['in'], df[model]['out']
        return None, None

    # Calculate and display cost when button is pressed
    input_value, output_value = find_model_values(selected_model, 
                                                [st.session_state['cost_openai'], st.session_state['cost_azure'], st.session_state['cost_google'], st.session_state['cost_mistral'], st.session_state['cost_local']])
    if input_value is not None and output_value is not None:
        cost = (n_in/1000 * input_value + n_out/1000 * output_value) * n_img
    with calculator_5:
        st.text_input("Total Cost", f"${round(cost,2)}") # selected_model
    
    with col_cost_1:
        rounding = 4
        st.dataframe(st.session_state.styled_cost_openai.format(precision=rounding), hide_index=True,)
    with col_cost_2:
        st.dataframe(st.session_state.styled_cost_azure.format(precision=rounding), hide_index=True,)
    with col_cost_3:
        st.dataframe(st.session_state.styled_cost_google.format(precision=rounding), hide_index=True,)
    with col_cost_4:
        st.dataframe(st.session_state.styled_cost_mistral.format(precision=rounding), hide_index=True,)
    with col_cost_5:
        st.dataframe(st.session_state.styled_cost_local.format(precision=rounding), hide_index=True,)



def content_prompt_and_llm_version():
    st.header('Prompt Version')
    col_prompt_1, col_prompt_2 = st.columns([4,2])              
    with col_prompt_1:
        available_prompts = get_prompt_versions(st.session_state.config['leafmachine']['LLM_version'])
        

        if available_prompts:
            default_version = available_prompts[0]  ######### Can be configured by user #################################################################
            selected_version = st.session_state.config['leafmachine']['project'].get('prompt_version', default_version)
            if selected_version not in available_prompts:
                selected_version = default_version
            st.session_state.config['leafmachine']['project']['prompt_version'] = st.selectbox("Prompt Version", available_prompts, index=available_prompts.index(selected_version),label_visibility='collapsed')

    with col_prompt_2:
        if st.button("Build Custom LLM Prompt"):
            st.session_state.proceed_to_build_llm_prompt = True
            st.rerun()

    st.header('LLM Version')
    col_llm_1, col_llm_2 = st.columns([4,2])  
     
    with col_llm_1:
        GUI_MODEL_LIST = ModelMaps.get_models_gui_list()
        st.session_state.config['leafmachine']['LLM_version'] = st.selectbox("LLM version", GUI_MODEL_LIST, index=GUI_MODEL_LIST.index(st.session_state.config['leafmachine'].get('LLM_version', ModelMaps.MODELS_GUI_DEFAULT)))
    


def content_api_check():
    # In your Streamlit layout
    # Create two columns for the header and the button
    col_llm_2a, col_llm_2b = st.columns([6, 2])  # Adjust the ratio as needed

    # Place the header in the first column
    with col_llm_2a:
        st.header('Available APIs')

        # Display API key status
        display_api_key_status()
    
        # Place the button in the second column, right-justified
        # with col_llm_2b:
        if st.button("Re-Check API Keys"):
            st.session_state['API_checked'] = False
            st.session_state['API_rechecked'] = True
        # with col_llm_2c:
        if st.button("Edit API Keys"):
            st.session_state.proceed_to_private = True
            st.rerun()
            

    
    

def content_collage_overlay():
    st.write("---")
    st.header('LeafMachine2 Label Collage')    
    col_cropped_1, col_cropped_2 = st.columns([4,4])   

    st.write("---")
    st.header('OCR Overlay Image')    
    col_ocr_1, col_ocr_2 = st.columns([4,4])        
    
    demo_text_h = f"Google_OCR_Handwriting:\nHERBARIUM OF MARCUS W. LYON , JR . Tracaulon sagittatum Indiana : Porter Co. incal Springs edge wet subdunal woods 1927 TX 11 Ilowers pink UNIVERSITE HERBARIUM MICH University of Michigan Herbarium 1439649 copyright reserved PERSICARIA FEB 2 6 1965 cm "
    demo_text_tr = f"trOCR:\nherbarium of marcus w. lyon jr. : : : tracaulon sagittatum indiana porter co. incal springs TX 11 Ilowers pink  1439649 copyright reserved D H U Q "
    demo_text_p = f"Google_OCR_Printed:\nTracaulon sagittatum Indiana : Porter Co. incal Springs edge wet subdunal woods 1927  Ilowers pink 1439649 copyright reserved PERSICARIA FEB 2 6 1965 cm "
    demo_text_b = demo_text_h + '\n' + demo_text_p
    demo_text_trb = demo_text_h + '\n' + demo_text_p + '\n' + demo_text_tr
    demo_text_trh = demo_text_h + '\n' + demo_text_tr
    demo_text_trp = demo_text_p + '\n' + demo_text_tr

    with col_cropped_1:
        default_crops = st.session_state.config['leafmachine']['cropped_components']['save_cropped_annotations']
        st.write("Prior to transcription, use LeafMachine2 to crop all labels from input images to create label collages for each specimen image. (Requires GPU)")
        # Set the options for the radio button
        options = {
            "Use LeafMachine2 label collage for transcriptions": "use_RGB_label_images",
            "Use specimen collage for transcriptions": "use_specimen_collage"
        }

        # Create the radio button with the available options
        selected_option = st.radio(
            "Select the transcription method:",
            options=list(options.keys()),
            index=0 if st.session_state.config['leafmachine'].get('use_RGB_label_images', False) else 1
        )

        # Update the session state based on the selected option
        st.session_state.config['leafmachine']['use_RGB_label_images'] = (selected_option == "Use LeafMachine2 label collage for transcriptions")
        st.session_state.config['leafmachine']['project']['use_specimen_collage'] = (selected_option == "Use specimen collage for transcriptions")

        option_selected_crops = st.multiselect(label="Components to crop",  
                options=['ruler', 'barcode','label', 'colorcard','map','envelope','photo','attached_item','weights',
                'leaf_whole', 'leaf_partial', 'leaflet', 'seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many', 'bud','specimen','roots','wood'],default=default_crops)
        st.session_state.config['leafmachine']['cropped_components']['save_cropped_annotations'] = option_selected_crops
    with col_cropped_2:
        # Load the image only if it's not already in the session state
        if "demo_collage" not in st.session_state:
            # ba = os.path.join(st.session_state.dir_home, 'demo', 'ba', 'ba2.png')
            ba = os.path.join(st.session_state.dir_home, 'demo', 'ba', 'ba2.jpg')
            st.session_state["demo_collage"] = Image.open(ba)

        # Display the image
        # st.image(st.session_state["demo_collage"], caption='LeafMachine2 Collage', output_format="PNG")
        st.image(st.session_state["demo_collage"], caption='LeafMachine2 Collage', output_format="JPEG")
        


    with col_ocr_1:
        options = [":rainbow[Printed + Handwritten]", "Printed", "Use both models"]
        captions = [
            "Works well for both printed and handwritten text", 
            "Works for printed text", 
            "Adds both OCR versions to the LLM prompt"
        ]

        st.write('This will plot bounding boxes around all text that Google Vision was able to detect. If there are no boxes around text, then the OCR failed, so that missing text will not be seen by the LLM when it is creating the JSON object. The created image will be viewable in the VoucherVisionEditor.')
        
        do_create_OCR_helper_image = st.checkbox("Create image showing an overlay of the OCR detections",value=st.session_state.config['leafmachine']['do_create_OCR_helper_image'])
        st.session_state.config['leafmachine']['do_create_OCR_helper_image'] = do_create_OCR_helper_image
        
        
        do_use_trOCR = st.checkbox("Supplement Google Vision OCR with trOCR (handwriting OCR) via 'microsoft/trocr-large-handwritten'", value=st.session_state.config['leafmachine']['project']['do_use_trOCR'],disabled=st.session_state['lacks_GPU'])
        st.session_state.config['leafmachine']['project']['do_use_trOCR'] = do_use_trOCR

        # Get the current OCR option from session state
        OCR_option = st.session_state.config['leafmachine']['project']['OCR_option']

        # Map the OCR option to the index in options list
        # You need to define the mapping based on your application's logic
        option_to_index = {
            'hand': 0,  
            'normal': 1,  
            'both': 2,
        }
        default_index = option_to_index.get(OCR_option, 0)  # Default to 0 if option not found

        # Create the radio button
        OCR_option_select = st.radio(
            "Select the Google Vision OCR version.",
            options,
            index=default_index,
            help="",captions=captions,
        )
        st.session_state.config['leafmachine']['project']['OCR_option'] = OCR_option_select

        if OCR_option_select == ":rainbow[Printed + Handwritten]":
            OCR_option = 'hand'
        elif OCR_option_select == "Printed":
            OCR_option = 'normal'
        elif OCR_option_select == "Use both models":
            OCR_option = 'both'
        else:
            raise

        st.session_state.config['leafmachine']['project']['OCR_option'] = OCR_option
        st.markdown("Below is an example of what the LLM would see given the choice of OCR ensemble. One, two, or three version of OCR can be fed into the LLM prompt. Typically, 'printed + handwritten' works well. If you have a GPU then you can enable trOCR.")
        if (OCR_option == 'hand') and not do_use_trOCR:
            st.text_area(label='HandwrittenPrinted',placeholder=demo_text_h,disabled=True, label_visibility='visible')
        elif (OCR_option == 'normal') and not do_use_trOCR:
            st.text_area(label='Printed',placeholder=demo_text_p,disabled=True, label_visibility='visible')
        elif (OCR_option == 'both') and not do_use_trOCR:
            st.text_area(label='HandwrittenPrinted + Printed',placeholder=demo_text_b,disabled=True, label_visibility='visible')
        elif (OCR_option == 'both') and do_use_trOCR:
            st.text_area(label='HandwrittenPrinted + Printed + trOCR',placeholder=demo_text_trb,disabled=True, label_visibility='visible')
        elif (OCR_option == 'normal') and do_use_trOCR:
            st.text_area(label='Printed + trOCR',placeholder=demo_text_trp,disabled=True, label_visibility='visible')
        elif (OCR_option == 'hand') and do_use_trOCR:
            st.text_area(label='HandwrittenPrinted + trOCR',placeholder=demo_text_trh,disabled=True, label_visibility='visible')
        
    with col_ocr_2:
        if "demo_overlay" not in st.session_state:
            # ocr = os.path.join(st.session_state.dir_home,'demo', 'ba','ocr.png')
            ocr = os.path.join(st.session_state.dir_home,'demo', 'ba','ocr.jpg')
            st.session_state["demo_overlay"] = Image.open(ocr)
        
        # st.image(st.session_state["demo_overlay"], caption='OCR Overlay Images', output_format = "PNG")
        st.image(st.session_state["demo_overlay"], caption='OCR Overlay Images', output_format = "JPEG")



def content_archival_components():
    st.write("---")
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



def content_processing_options():
    st.write("---")
    st.header('Processing Options')
    col_processing_1, col_processing_2 = st.columns([2,2,])
    with col_processing_1:
        st.subheader('Compute Options')
        st.session_state.config['leafmachine']['project']['num_workers'] = st.number_input("Number of CPU workers", value=st.session_state.config['leafmachine']['project'].get('num_workers', 1), disabled=False)
        st.session_state.config['leafmachine']['project']['batch_size'] = st.number_input("Batch size", value=st.session_state.config['leafmachine']['project'].get('batch_size', 500), help='Sets the batch size for the LeafMachine2 cropping. If computer RAM is filled, lower this value to ~100.')
    with col_processing_2:
        st.subheader('Filename Prefix Handling')
        st.session_state.config['leafmachine']['project']['prefix_removal'] = st.text_input("Remove prefix from catalog number", st.session_state.config['leafmachine']['project'].get('prefix_removal', ''),placeholder="e.g. MICH-V-")
        st.session_state.config['leafmachine']['project']['suffix_removal'] = st.text_input("Remove suffix from catalog number", st.session_state.config['leafmachine']['project'].get('suffix_removal', ''),placeholder="e.g. _B")
        st.session_state.config['leafmachine']['project']['catalog_numerical_only'] = st.checkbox("Require 'Catalog Number' to be numerical only", st.session_state.config['leafmachine']['project'].get('catalog_numerical_only', True))
    
    ### Logging and Image Validation - col_v1
    st.write("---")
    st.header('Logging and Image Validation')    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        option_check_illegal = st.checkbox("Check for illegal filenames", value=st.session_state.config['leafmachine']['do']['check_for_illegal_filenames'])
        st.session_state.config['leafmachine']['do']['check_for_illegal_filenames'] = option_check_illegal
       
        st.session_state.config['leafmachine']['do']['check_for_corrupt_images_make_vertical'] = st.checkbox("Check for corrupt images", st.session_state.config['leafmachine']['do'].get('check_for_corrupt_images_make_vertical', True),disabled=True)
        
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
    st.write("---")
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
        for available_prompts in ModelMaps.PROMPTS_THAT_NEED_DOMAIN_KNOWLEDGE:
            st.markdown(f"- {available_prompts}")
        
        if st.session_state.config['leafmachine']['project']['prompt_version'] in ModelMaps.PROMPTS_THAT_NEED_DOMAIN_KNOWLEDGE:
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



def content_space_saver():
    st.write("---")
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



#################################################################################################################################################
# render_expense_report_summary #################################################################################################################
#################################################################################################################################################
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
        colors = [ModelMaps.COLORS_EXPENSE_REPORT[version] if version in ModelMaps.COLORS_EXPENSE_REPORT else '#DDDDDD' for version in api_versions]
        
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
            text=[f"${value:.4f}" for value in cost_values],  # Formats the cost as a string with a dollar sign and two decimals
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
            text=[f"${cost:.4f}" for cost in total_cost_by_version.values()],  # This will format the cost to 2 decimal places
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



def content_less_used():
    st.write('---')
    st.write(':octagonal_sign: ***NOTE:*** Settings below are not relevant for most projects. Some settings below may not be reflected in saved settings files and would need to be set each time.')

#################################################################################################################################################
# Sidebar #######################################################################################################################################
#################################################################################################################################################
def sidebar_content():
    if not os.path.exists(os.path.join(st.session_state.dir_home,'expense_report')):
        validate_dir(os.path.join(st.session_state.dir_home,'expense_report'))
    expense_report_path = os.path.join(st.session_state.dir_home, 'expense_report', 'expense_report.csv')

    if os.path.exists(expense_report_path):
        # File exists, proceed with summarization
        st.session_state.expense_summary, st.session_state.expense_report = summarize_expense_report(expense_report_path)
        render_expense_report_summary()  
    else:
        # File does not exist, handle this case appropriately
        # For example, you could set the session state variables to None or an empty value
        st.session_state.expense_summary, st.session_state.expense_report = None, None
        st.header('Expense Report Summary')
        st.write('Available after first run...')


#################################################################################################################################################
# Routing Function ##############################################################################################################################
#################################################################################################################################################
def main():
    with st.sidebar:
        sidebar_content()
    # Main App
    content_header()

    col1, col2 = st.columns([1,1])
    with col1:
        content_project_settings()
    with col2:
        content_input_images()
    st.write("---")
    col3, col4 = st.columns([1,1])
    with col3:
        content_prompt_and_llm_version()
    with col4:
        content_api_check()
    content_llm_cost()
    content_collage_overlay()
    content_processing_options()
    content_less_used()
    content_archival_components()
    content_space_saver()
    # content_tab_domain()



#################################################################################################################################################
# Initializations ###############################################################################################################################
#################################################################################################################################################
        
st.set_page_config(layout="wide", page_icon='img/icon.ico', page_title='VoucherVision')


# Default YAML file path
if 'config' not in st.session_state:
    st.session_state.config, st.session_state.dir_home = build_VV_config(loaded_cfg=None)
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


if 'processing_add_on' not in st.session_state:
    st.session_state['processing_add_on'] = ''


if 'formatted_json' not in st.session_state:
    st.session_state['formatted_json'] = None
if 'formatted_json_WFO' not in st.session_state:
    st.session_state['formatted_json_WFO'] = None
if 'formatted_json_GEO' not in st.session_state:
    st.session_state['formatted_json_GEO'] = None


if 'lacks_GPU' not in st.session_state:
    st.session_state['lacks_GPU'] = not torch.cuda.is_available()


if 'API_key_validation' not in st.session_state:
    st.session_state['API_key_validation'] = False
if 'present_annotations' not in st.session_state:
    st.session_state['present_annotations'] = None
if 'missing_annotations' not in st.session_state:
    st.session_state['missing_annotations'] = None
if 'date_of_check' not in st.session_state:
    st.session_state['date_of_check'] = None
if 'API_checked' not in st.session_state:
    st.session_state['API_checked'] = False
if 'API_rechecked' not in st.session_state:
    st.session_state['API_rechecked'] = False 


if 'cost_openai' not in st.session_state:
    st.session_state['cost_openai'] = None
if 'cost_azure' not in st.session_state:
    st.session_state['cost_azure'] = None
if 'cost_google' not in st.session_state:
    st.session_state['cost_google'] = None
if 'cost_mistral' not in st.session_state:
    st.session_state['cost_mistral'] = None
if 'cost_local' not in st.session_state:
    st.session_state['cost_local'] = None

if 'settings_filename' not in st.session_state:
    st.session_state['settings_filename'] = None
if 'loaded_settings_filename' not in st.session_state:
    st.session_state['loaded_settings_filename'] = None


# Initialize session_state variables if they don't exist
if 'prompt_info' not in st.session_state:
    st.session_state['prompt_info'] = {}
if 'rules' not in st.session_state:
    st.session_state['rules'] = {}
if 'required_fields' not in st.session_state:
    st.session_state['required_fields'] = ['catalogNumber','order','family','scientificName',
                                           'scientificNameAuthorship','genus','subgenus','specificEpithet','infraspecificEpithet',
                                           'verbatimEventDate','eventDate',
                                           'country','stateProvince','county','municipality','locality','decimalLatitude','decimalLongitude','verbatimCoordinates',]

# These are the fields that are in SLTPvA that are not required by another parsing valication function:
#     "identifiedBy": "M.W. Lyon, Jr.",
#     "recordedBy": "University of Michigan Herbarium",
#     "recordNumber": "",
#     "habitat": "wet subdunal woods",
#     "occurrenceRemarks": "Indiana : Porter Co.",
#     "degreeOfEstablishment": "",
#     "minimumElevationInMeters": "",
#     "maximumElevationInMeters": ""



if 'proceed_to_build_llm_prompt' not in st.session_state:
    st.session_state.proceed_to_build_llm_prompt = False  
if 'proceed_to_component_detector' not in st.session_state:
    st.session_state.proceed_to_component_detector = False  
if 'proceed_to_parsing_options' not in st.session_state:
    st.session_state.proceed_to_parsing_options = False  
if 'proceed_to_api_keys' not in st.session_state:
    st.session_state.proceed_to_api_keys = False  
if 'proceed_to_space_saver' not in st.session_state:
    st.session_state.proceed_to_space_saver = False  

#################################################################################################################################################
# Main ##########################################################################################################################################
#################################################################################################################################################


if not st.session_state.private_file:
    create_private_file()
elif st.session_state.proceed_to_build_llm_prompt:
    build_LLM_prompt_config()
elif st.session_state.proceed_to_private:
    create_private_file()
elif st.session_state.proceed_to_main:
    main()