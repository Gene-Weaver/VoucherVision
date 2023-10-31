import streamlit as st
import yaml, os, json
from PIL import Image
import pandas as pd
from vouchervision.LeafMachine2_Config_Builder import write_config_file
from vouchervision.VoucherVision_Config_Builder import build_VV_config, run_demo_tests_GPT, run_demo_tests_Palm , TestOptionsGPT, TestOptionsPalm
from vouchervision.vouchervision_main import voucher_vision
from vouchervision.general_utils import load_config_file_testing, test_GPU, get_cfg_from_full_path

PROMPTS_THAT_NEED_DOMAIN_KNOWLEDGE = ["Version 1","Version 1 PaLM 2"]

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

def get_prompt_versions(LLM_version):
    if LLM_version in ["GPT 4", "GPT 3.5", "Azure GPT 4", "Azure GPT 3.5"]:
        return (["Version 1", "Version 1 No Domain Knowledge", "Version 2"], "Version 2")
    elif LLM_version in ["PaLM 2",]:
        return (["Version 1 PaLM 2", "Version 1 PaLM 2 No Domain Knowledge", "Version 2 PaLM 2"], "Version 2 PaLM 2")
    else:
        # Handle other cases or raise an error
        return ([], None)

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


def create_private_file():

    if st.session_state.private_file:
        cfg_private = get_private_file()
        create_private_file_0(cfg_private)
    else:
        st.title("VoucherVision")
        create_private_file_0()

def create_private_file_0(cfg_private=None):
    col_private,_,__= st.columns([6,2,2])

    if cfg_private is None:
        cfg_private = {}
        
        cfg_private['openai'] = {}
        cfg_private['openai']['openai_api_key'] = ''
        cfg_private['openai']['API_VERSION'] = ''
        cfg_private['openai']['OPENAI_API_KEY'] =''
        cfg_private['openai']['openai_api_base'] =''
        cfg_private['openai']['OPENAI_organization'] =''
        cfg_private['openai']['openai_api_type'] =''

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

        google_vision = st.text_input("Full path to Google Cloud JSON API key file", cfg_private['google_cloud'].get('path_json_file', ''),
                                                 help='This API Key is in the form of a JSON file. Please save the JSON file in a safe directory. DO NOT store the JSON key inside of the VoucherVision directory.',
                                                 placeholder = 'e.g. C:/Documents/Secret_Files/google_API/application_default_credentials.json',
                                                 type='password')

        
        st.write("---")
        st.subheader("OpenAI")
        st.markdown("API key for first-party OpenAI API. Create an account with OpenAI [here](https://platform.openai.com/signup), then create an API key [here](https://platform.openai.com/account/api-keys).")
        openai_api_key = st.text_input("openai_api_key", cfg_private['openai'].get('openai_api_key', ''),
                                                 help='The actual API key. Likely to be a string of 2 character, a dash, and then a 48-character string: sk-XXXXXXXX...',
                                                 placeholder = 'e.g. sk-XXXXXXXX...',
                                                 type='password')

        st.write("---")
        st.subheader("OpenAI - Azure")
        st.markdown("This version OpenAI relies on Azure servers directly as is intended for private enterprise instances of OpenAI's services, such as [UM-GPT](https://its.umich.edu/computing/ai). Administrators will provide you with the following information.")
        azure_openai_api_version = st.text_input("azure_openai_api_version", cfg_private['openai'].get('API_VERSION', ''),
                                                 help='API Version e.g. "2023-05-15"',
                                                 placeholder = 'e.g. 2023-05-15',
                                                 type='password')
        azure_openai_api_key = st.text_input("azure_openai_api_key", cfg_private['openai'].get('OPENAI_API_KEY', ''),
                                                 help='The actual API key. Likely to be a 32-character string',
                                                 placeholder = 'e.g. 12333333333333333333333333333332',
                                                 type='password')
        azure_openai_api_base = st.text_input("azure_openai_api_base", cfg_private['openai'].get('openai_api_base', ''),
                                                 help='The base url for the API e.g. "https://api.umgpt.umich.edu/azure-openai-api"',
                                                 placeholder = 'e.g. https://api.umgpt.umich.edu/azure-openai-api',
                                                 type='password')
        azure_openai_organization = st.text_input("azure_openai_organization", cfg_private['openai'].get('OPENAI_organization', ''),
                                                 help='Your organization code. Likely a short string',
                                                 placeholder = 'e.g. 123456',
                                                 type='password')
        azure_openai_api_type = st.text_input("azure_openai_api_type", cfg_private['openai'].get('openai_api_type', ''),
                                                 help='The API type. Typically "azure"',
                                                 placeholder = 'e.g. azure',
                                                 type='password')

        
        st.write("---")
        st.subheader("Google PaLM 2")
        st.markdown('Follow these [instructions](https://developers.generativeai.google/tutorials/setup) to generate an API key for PaLM 2. You may need to also activate an account with [MakerSuite](https://makersuite.google.com/app/apikey) and enable "early access."')
        google_palm = st.text_input("Google PaLM 2 API Key", cfg_private['google_palm'].get('google_palm_api', ''),
                                                 help='The MakerSuite API key e.g. a 32-character string',
                                                 placeholder='e.g. SATgthsykuE64FgrrrrEervr3S4455t_geyDeGq',
                                                 type='password')

        st.button("Set API Keys",type='primary', on_click=save_changes_to_API_keys, args=[cfg_private,openai_api_key,azure_openai_api_version,azure_openai_api_key,
                                                                    azure_openai_api_base,azure_openai_organization,azure_openai_api_type,google_vision,google_palm])
            

def save_changes_to_API_keys(cfg_private,openai_api_key,azure_openai_api_version,azure_openai_api_key,
                             azure_openai_api_base,azure_openai_organization,azure_openai_api_type,google_vision,google_palm):
    # Update the configuration dictionary with the new values
    cfg_private['openai']['openai_api_key'] = openai_api_key

    cfg_private['openai']['API_VERSION'] = azure_openai_api_version
    cfg_private['openai']['OPENAI_API_KEY'] = azure_openai_api_key
    cfg_private['openai']['openai_api_base'] = azure_openai_api_base
    cfg_private['openai']['OPENAI_organization'] = azure_openai_organization
    cfg_private['openai']['openai_api_type'] = azure_openai_api_type

    cfg_private['google_cloud']['path_json_file'] = google_vision

    cfg_private['google_palm']['google_palm_api'] = google_palm
    # Call the function to write the updated configuration to the YAML file
    write_config_file(cfg_private, st.session_state.dir_home, filename="PRIVATE_DATA.yaml")
    st.session_state.private_file = does_private_file_exist()





def main():
    # Main App
    st.title("VoucherVision")

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
        

    with col_run_1:
        st.subheader('Run VoucherVision')
        if st.button("Start Processing", type='primary'):
            # First, write the config file.
            write_config_file(st.session_state.config, st.session_state.dir_home, filename="VoucherVision.yaml")
            
            # Call the machine function.
            last_JSON_response = voucher_vision(None, st.session_state.dir_home, None, progress_report)
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
        st.write("If you use VoucherVision frequently, you can change the default values that are auto-populated in the form below. In a text editor or IDE, edit the first few rows in the file `../VoucherVision/vouchervision/VoucherVision_Config_Builder.py`")

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


        


    tab_settings, tab_domain, tab_component, tab_processing, tab_private, tab_delete = st.tabs(["Project Settings", "Domain Knowledge","Component Detector", "Processing Options", "API Keys", "Space-Saver"])




    # Tab 1: General Settings
    with tab_settings:
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
                Select an LLM.
                - GPT-4 is 20x more expensive than GPT-3.5  
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
        



    with tab_component:
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



    with tab_domain:
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
            
        


    ### Processing Options
    with tab_processing:
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


    with tab_private:
        create_private_file()

    with tab_delete:
        create_space_saver()



st.set_page_config(layout="wide", page_icon='img/icon_VV.ico', page_title='VoucherVision')

# Default YAML file path
if 'config' not in st.session_state:
    st.session_state.config, st.session_state.dir_home = build_VV_config()
    setup_streamlit_config(st.session_state.dir_home)
if 'private_file' not in st.session_state:
    st.session_state.private_file = does_private_file_exist()

if not st.session_state.private_file:
    create_private_file()
else:
    main()