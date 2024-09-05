import os, yaml
import streamlit as st
from PIL import Image
from itertools import chain

from vouchervision.model_maps import ModelMaps
from vouchervision.utils_hf import check_prompt_yaml_filename

st.set_page_config(layout="wide", page_icon='img/icon.ico', page_title='VV Prompt Builder',initial_sidebar_state="collapsed")

def create_download_button_yaml(file_path, selected_yaml_file, key_val):
    file_label = f"Download {selected_yaml_file}"
    with open(file_path, 'rb') as f:
        st.download_button(
            label=file_label,
            data=f,
            file_name=os.path.basename(file_path),
            mime='application/x-yaml',use_container_width=True,key=key_val,
        )


# def upload_local_prompt_to_server(dir_prompt):
#     uploaded_file = st.file_uploader("Upload a custom prompt file", type=['yaml'])
#     if uploaded_file is not None:
#         # Check the file extension
#         file_name = uploaded_file.name
#         if file_name.endswith('.yaml'):
#             file_path = os.path.join(dir_prompt, file_name)
            
#             # Save the file
#             with open(file_path, 'wb') as f:
#                 f.write(uploaded_file.getbuffer())
#             st.success(f"Saved file {file_name} in {dir_prompt}")
#         else:
#             st.error("Please upload a .yaml file that you previously created using this Prompt Builder tool.")
def upload_local_prompt_to_server(dir_prompt):
    uploaded_file = st.file_uploader("Upload a custom prompt file", type=['yaml'])
    if uploaded_file is not None:
        # Check the file extension
        file_name = uploaded_file.name
        if file_name.endswith('.yaml'):
            file_path = os.path.join(dir_prompt, file_name)
            
            # Save the file
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved file {file_name} in {dir_prompt}")
            
            # Update the prompt list
            st.session_state['yaml_files'] = [f for f in os.listdir(dir_prompt) if f.endswith('.yaml')]
        else:
            st.error("Please upload a .yaml file that you previously created using this Prompt Builder tool.")



def save_prompt_yaml(filename, col):
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

    with col: # added
        create_download_button_yaml(filepath, filename,key_val=2456237465) # added


def load_prompt_yaml(filename):
    st.session_state['user_clicked_load_prompt_yaml'] = filename
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

        # print(st.session_state['mapping'].values())
        # print(chain.from_iterable(st.session_state['mapping'].values()))
        # print(list(chain.from_iterable(st.session_state['mapping'].values())))
        st.session_state['assigned_columns'] = list(chain.from_iterable(st.session_state['mapping'].values())) 


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
    

def build_LLM_prompt_config():
    col_main1, col_main2 = st.columns([10,2])
    with col_main1:
        st.session_state.logo_path = os.path.join(st.session_state.dir_home, 'img','logo.png')
        st.session_state.logo = Image.open(st.session_state.logo_path)
        st.image(st.session_state.logo, width=250)
    with col_main2:
        try:
            st.page_link('app.py', label="Home", icon="🏠")
            st.page_link(os.path.join("pages","faqs.py"), label="FAQs", icon="❔")
            st.page_link(os.path.join("pages","report_bugs.py"), label="Report a Bug", icon="⚠️")
        except:
            st.page_link(os.path.join(os.path.dirname(os.path.dirname(__file__)),'app.py'), label="Home", icon="🏠")
            st.page_link(os.path.join(os.path.dirname(os.path.dirname(__file__)),"pages","faqs.py"), label="FAQs", icon="❔")
            st.page_link(os.path.join(os.path.dirname(os.path.dirname(__file__)),"pages","report_bugs.py"), label="Report a Bug", icon="⚠️")

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
        st.write("5. The LLM ***only*** sees information from the 'instructions', 'rules', and 'json_formatting_instructions' sections. All other information is for versioning and integration with VoucherVisionEditor.")
        
        st.write("---")
        st.header('Load an Existing Prompt Template')
        st.write("By default, this form loads the minimum required transcription fields but does not provide rules for each field. You can also load an existing prompt as a template, editing or deleting values as needed.")

        dir_prompt = os.path.join(st.session_state.dir_home, 'custom_prompts')
        yaml_files = [f for f in os.listdir(dir_prompt) if f.endswith('.yaml')]
        col_load_text, col_load_btn, col_load_btn2 = st.columns([8,2,2])
        with col_load_text:
        # Dropdown for selecting a YAML file
            st.session_state['selected_yaml_file'] = st.selectbox('Select a prompt .YAML file to load:', [''] + yaml_files)
        with col_load_btn:
            st.write('##')
            # Button to load the selected prompt
            st.button('Load Prompt', on_click=btn_load_prompt, args=[st.session_state['selected_yaml_file'], dir_prompt],use_container_width=True)
                
        with col_load_btn2:
            if st.session_state['selected_yaml_file']:
                # Construct the full path to the file
                download_file_path = os.path.join(dir_prompt, st.session_state['selected_yaml_file'] )
                # Create the download button
                st.write('##')
                create_download_button_yaml(download_file_path, st.session_state['selected_yaml_file'],key_val=345798)


        upload_local_prompt_to_server(dir_prompt)

        # Prompt Author Information
        st.write("---")
        st.header("Prompt Author Information")
        st.write("We value community contributions! Please provide your name(s) (or pseudonym if you prefer) for credit. If you leave this field blank, it will say 'unknown'.")
        if 'prompt_author' not in st.session_state:# != st.session_state['default_prompt_author']:
            st.session_state['prompt_author'] = st.text_input("Enter names of prompt author(s)", value=st.session_state['default_prompt_author'],key=1111)
        else:
            st.session_state['prompt_author'] = st.text_input("Enter names of prompt author(s)", value=st.session_state['prompt_author'],key=1112)

        # Institution 
        st.write("Please provide your institution name. If you leave this field blank, it will say 'unknown'.")
        if 'prompt_author_institution' not in st.session_state:
            st.session_state['prompt_author_institution'] = st.text_input("Enter name of institution", value=st.session_state['default_prompt_author_institution'],key=1113)
        else:
            st.session_state['prompt_author_institution'] = st.text_input("Enter name of institution", value=st.session_state['prompt_author_institution'],key=1114)

        # Prompt name 
        st.write("Please provide a simple name for your prompt. If you leave this field blank, it will say 'custom_prompt'.")
        if 'prompt_name' not in st.session_state:
            st.session_state['prompt_name'] = st.text_input("Enter prompt name", value=st.session_state['default_prompt_name'],key=1115)
        else:
            st.session_state['prompt_name'] = st.text_input("Enter prompt name", value=st.session_state['prompt_name'],key=1116)
      
        # Prompt verion  
        st.write("Please provide a version identifier for your prompt. If you leave this field blank, it will say 'v-1-0'.")
        if 'prompt_version' not in st.session_state:
            st.session_state['prompt_version'] = st.text_input("Enter prompt version", value=st.session_state['default_prompt_version'],key=1117)
        else:
            st.session_state['prompt_version'] = st.text_input("Enter prompt version", value=st.session_state['prompt_version'],key=1118)
            
       
        st.write("Please provide a description of your prompt and its intended task. Is it designed for a specific collection? Taxa? Database structure?")
        if 'prompt_description' not in st.session_state:
            st.session_state['prompt_description'] = st.text_input("Enter description of prompt", value=st.session_state['default_prompt_description'],key=1119)
        else:
            st.session_state['prompt_description'] = st.text_input("Enter description of prompt", value=st.session_state['prompt_description'],key=11111)
       
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
            st.session_state['instructions'] = st.text_area("Enter guiding instructions", value=st.session_state['default_instructions'].strip(), height=350,key=111112)
        else:
            st.session_state['instructions'] = st.text_area("Enter guiding instructions", value=st.session_state['instructions'].strip(), height=350,key=111112)


        st.write('---')

        # Column Instructions Section
        st.header("JSON Formatting Instructions")
        st.write("The following section tells the LLM how we want to structure the JSON dictionary. We do not recommend changing this section because it would likely result in unstable and inconsistent behavior.")
        if 'json_formatting_instructions' not in st.session_state:
            st.session_state['json_formatting_instructions'] = st.text_area("Enter general JSON guidelines", value=st.session_state['default_json_formatting_instructions'],key=111114)
        else:
            st.session_state['json_formatting_instructions'] = st.text_area("Enter general JSON guidelines", value=st.session_state['json_formatting_instructions'],key=111115)
        





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

            # Handle commit action
            if commit_button and st.session_state['current_rule']:
                # Commit the rules to the session state.
                st.session_state['rules'][st.session_state['current_rule']] = current_rule_description
                st.success(f"Column '{st.session_state['current_rule']}' added/updated in rules.")

                # Force the form to reset by clearing the fields from the session state
                st.session_state.pop('current_rule', None)  # Clear the selected column to force reset

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

        st.write('---')

        col_left_mapping, col_right_mapping = st.columns([6,4])
        with col_left_mapping:
            st.header("Mapping")
            st.write("Assign each column name to a single category.")
            st.session_state['refresh_mapping'] = False

            # Dynamically create a list of all column names that can be assigned
            # This assumes that the column names are the keys in the dictionary under 'rules'
            all_column_names = list(st.session_state['rules'].keys())

            categories = ['TAXONOMY', 'GEOGRAPHY', 'LOCALITY', 'COLLECTING', 'MISC']
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
                            save_prompt_yaml(new_filename, col_left_save)
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
        if st.session_state['user_clicked_load_prompt_yaml'] is None: # see if user has loaded a yaml to edit
            st.session_state['show_prompt_name_e'] = f"Prompt Status  :arrow_forward:  Building prompt from scratch"
            if st.session_state['prompt_name']:
                st.session_state['show_prompt_name_w'] = f"New Prompt Name  :arrow_forward:  {st.session_state['prompt_name']}.yaml"
            else:
                st.session_state['show_prompt_name_w'] = f"New Prompt Name  :arrow_forward:  [PLEASE SET NAME]"
        else:
            st.session_state['show_prompt_name_e'] = f"Prompt Status: Editing  :arrow_forward:  {st.session_state['selected_yaml_file']}"
            if st.session_state['prompt_name']:
                st.session_state['show_prompt_name_w'] = f"New Prompt Name  :arrow_forward:  {st.session_state['prompt_name']}.yaml"
            else:
                st.session_state['show_prompt_name_w'] = f"New Prompt Name  :arrow_forward:  [PLEASE SET NAME]"

        st.subheader(f'Full Prompt')
        st.write(st.session_state['show_prompt_name_e'])
        st.write(st.session_state['show_prompt_name_w'])
        st.write("---")
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

build_LLM_prompt_config()