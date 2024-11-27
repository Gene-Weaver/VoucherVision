import streamlit as st
import yaml, os, json, random, time, re, torch, random, warnings, shutil, sys, glob
import seaborn as sns
import plotly.graph_objs as go
from PIL import Image
import pandas as pd
from io import BytesIO
# from streamlit_extras.let_it_rain import rain
from annotated_text import annotated_text

from vouchervision.LeafMachine2_Config_Builder import write_config_file
from vouchervision.VoucherVision_Config_Builder import build_VV_config, TestOptionsGPT, TestOptionsPalm, check_if_usable
from vouchervision.vouchervision_main import voucher_vision
from vouchervision.general_utils import test_GPU, get_cfg_from_full_path, summarize_expense_report, validate_dir, install_qwen_requirements
from vouchervision.model_maps import ModelMaps
from vouchervision.API_validation import APIvalidation
from vouchervision.utils_hf import setup_streamlit_config, save_uploaded_file, save_uploaded_local, save_uploaded_file_local, report_violation
from vouchervision.data_project import convert_pdf_to_jpg
from vouchervision.utils_LLM import check_system_gpus
from vouchervision.OCR_google_cloud_vision import SafetyCheck

import cProfile
import pstats
#################################################################################################################################################
# Initializations ###############################################################################################################################
#################################################################################################################################################
st.set_page_config(layout="wide", page_icon='img/icon.ico', page_title='VoucherVision',initial_sidebar_state="collapsed")

# Parse the 'is_hf' argument and set it in session state
if 'is_hf' not in st.session_state:
    is_hf_os = os.getenv('IS_HF', '').lower()  # Get the environment variable and convert to lowercase for uniformity
    print(f"=== os.getenv('IS_HF', '').lower() ===> {is_hf_os} ===")
    if is_hf_os in ['1', 'true']:  # Check against string representations of truthy values
        st.session_state['is_hf'] = True
    else:
        st.session_state['is_hf'] = False

print(f"=== is_hf {st.session_state['is_hf']} ===")


# Default YAML file path
if 'config' not in st.session_state:
    st.session_state.config, st.session_state.dir_home = build_VV_config(loaded_cfg=None)
    setup_streamlit_config(st.session_state.dir_home)

# st.session_state['is_hf'] = True

########################################################################################################
###  Global constants                                                                               ####
########################################################################################################
MAX_GALLERY_IMAGES = 20
GALLERY_IMAGE_SIZE = 96


########################################################################################################
###  Init funcs                                                                                     ####
########################################################################################################
def does_private_file_exist():
    dir_home = os.path.dirname(__file__)
    path_cfg_private = os.path.join(dir_home, 'PRIVATE_DATA.yaml')
    return os.path.exists(path_cfg_private)


########################################################################################################
###  Streamlit inits                         [FOR SAVE FILE]                                        ####
########################################################################################################




########################################################################################################
###  Streamlit inits                         [routing]                                              ####
########################################################################################################
if st.session_state['is_hf']:
    if 'proceed_to_main' not in st.session_state:
        st.session_state.proceed_to_main = True
    
    if 'proceed_to_private' not in st.session_state:
        st.session_state.proceed_to_private = False 

    if 'private_file' not in st.session_state:
        st.session_state.private_file = True 
else:
    if 'proceed_to_main' not in st.session_state:
        st.session_state.proceed_to_main = False  # New state variable to control the flow

    if 'private_file' not in st.session_state:
        st.session_state.private_file = does_private_file_exist()
        if st.session_state.private_file:
            st.session_state.proceed_to_main = True

    if 'proceed_to_private' not in st.session_state:
        st.session_state.proceed_to_private = False  # New state variable to control the flow


if 'proceed_to_build_llm_prompt' not in st.session_state:
    st.session_state.proceed_to_build_llm_prompt = False  # New state variable to control the flow
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
if 'proceed_to_faqs' not in st.session_state:
    st.session_state.proceed_to_faqs = False 


########################################################################################################
###  Streamlit inits                         [basics]                                               ####
########################################################################################################
if 'processing_add_on' not in st.session_state:
    st.session_state['processing_add_on'] = 0


if 'capability_score' not in st.session_state:
    st.session_state['num_gpus'], st.session_state['gpu_dict'], st.session_state['total_vram_gb'], st.session_state['capability_score'] = check_system_gpus()


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
if 'API_checked' not in st.session_state:
    st.session_state['API_checked'] = False
if 'API_rechecked' not in st.session_state:
    st.session_state['API_rechecked'] = False 


if 'present_annotations' not in st.session_state:
    st.session_state['present_annotations'] = None
if 'missing_annotations' not in st.session_state:
    st.session_state['missing_annotations'] = None
if 'model_annotations' not in st.session_state:
    st.session_state['model_annotations'] = None
if 'date_of_check' not in st.session_state:
    st.session_state['date_of_check'] = None


if 'support_qwen' not in st.session_state:
    st.session_state['support_qwen'] = False
if 'support_qwen_check' not in st.session_state:
    st.session_state['support_qwen_check'] = False


if 'json_report' not in st.session_state:
    st.session_state['json_report'] = False 
if 'hold_output' not in st.session_state:
    st.session_state['hold_output'] = False


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
if 'zip_filepath' not in st.session_state:
    st.session_state['zip_filepath'] = None


########################################################################################################
###  Streamlit inits                         [prompt builder]                                       ####
########################################################################################################
# These are the fields that are in SLTPvA that are not required by another parsing valication function:
#     "identifiedBy": "M.W. Lyon, Jr.",
#     "recordedBy": "University of Michigan Herbarium",
#     "recordNumber": "",
#     "habitat": "wet subdunal woods",
#     "occurrenceRemarks": "Indiana : Porter Co.",
#     "degreeOfEstablishment": "",
#     "minimumElevationInMeters": "",
#     "maximumElevationInMeters": ""
if 'required_fields' not in st.session_state:
    st.session_state['required_fields'] = ['catalogNumber','order','family','scientificName',
                                           'scientificNameAuthorship','genus','subgenus','specificEpithet','infraspecificEpithet',
                                           'verbatimEventDate','eventDate',
                                           'country','stateProvince','county','municipality','locality','decimalLatitude','decimalLongitude','verbatimCoordinates',]
if 'prompt_info' not in st.session_state:
    st.session_state['prompt_info'] = {}
if 'rules' not in st.session_state:
    st.session_state['rules'] = {}


########################################################################################################
###  Streamlit inits                         [gallery]                                              ####
########################################################################################################
if 'uploader_idk' not in st.session_state:
    st.session_state['uploader_idk'] = 1
if 'input_list_small' not in st.session_state:
    st.session_state['input_list_small'] = []  
if 'input_list' not in st.session_state:
    st.session_state['input_list'] = []
if 'user_clicked_load_prompt_yaml' not in st.session_state:
    st.session_state['user_clicked_load_prompt_yaml'] = None
if 'new_prompt_yaml_filename' not in st.session_state:
    st.session_state['new_prompt_yaml_filename'] = None
if 'view_local_gallery' not in st.session_state:
    st.session_state['view_local_gallery'] = False
if 'dir_images_local_TEMP' not in st.session_state:
    st.session_state['dir_images_local_TEMP'] = False
if 'dir_uploaded_images' not in st.session_state:
    st.session_state['dir_uploaded_images'] = os.path.join(st.session_state.dir_home,'uploads')
    validate_dir(os.path.join(st.session_state.dir_home,'uploads'))
if 'dir_uploaded_images_small' not in st.session_state:
    st.session_state['dir_uploaded_images_small'] = os.path.join(st.session_state.dir_home,'uploads_small')
    validate_dir(os.path.join(st.session_state.dir_home,'uploads_small'))




########################################################################################################
###  CONTENT                                  []                                             ####
########################################################################################################
@st.cache_data
def show_gallery_small():
    st.image(st.session_state['input_list_small'], width=GALLERY_IMAGE_SIZE)  

@st.cache_data
def show_gallery_small_hf(images_to_display):
    print(images_to_display)
    st.image(images_to_display)


@st.cache_data
def load_gallery(converted_files, uploaded_file):
    for file_name in converted_files:   
        if file_name.lower().endswith('.jpg'):
            jpg_file_path = os.path.join(st.session_state['dir_uploaded_images'], file_name)
            st.session_state['input_list'].append(jpg_file_path)

            # Optionally, create a thumbnail for the gallery
            img = Image.open(jpg_file_path)
            img.thumbnail((GALLERY_IMAGE_SIZE, GALLERY_IMAGE_SIZE), Image.Resampling.LANCZOS)
            file_path_small = save_uploaded_file(st.session_state['dir_uploaded_images_small'], uploaded_file, img)
            st.session_state['input_list_small'].append(file_path_small)



def handle_image_upload_and_gallery_hf(uploaded_files):
    SAFE = SafetyCheck(st.session_state['is_hf'])
    if uploaded_files:
        
        # Clear input image gallery and input list
        clear_image_uploads()

        ind_small = 0
        for uploaded_file in uploaded_files:

            if SAFE.check_for_inappropriate_content(uploaded_file):
                clear_image_uploads()
                report_violation(uploaded_file.name, is_hf=st.session_state['is_hf'])
                st.error("Warning: You uploaded an image that violates our terms of service.")
                return True
            
            # Print out details of the uploaded file for debugging
            # st.write(f"Uploaded file: {uploaded_file.name}")
            # st.write(f"File size: {len(uploaded_file.getvalue())} bytes")

            # Check if the uploaded file is not empty
            if len(uploaded_file.getvalue()) == 0:
                st.error(f"The uploaded file {uploaded_file.name} is empty.")
                continue

            # Save the uploaded file (PDF or image)
            file_path = save_uploaded_file(st.session_state['dir_uploaded_images'], uploaded_file)
            
            if not file_path:
                st.error(f"Failed to process the file: {uploaded_file.name}")
                continue  # Skip to the next file
            
            # Determine the file type
            if uploaded_file.name.lower().endswith('.pdf'):
                try:
                    # Convert each page of the PDF to an image
                    n_pages = convert_pdf_to_jpg(file_path, st.session_state['dir_uploaded_images'], dpi=200)
                    if n_pages == 0:
                        st.error(f"No pages were converted from the PDF: {uploaded_file.name}")
                        continue  # Skip to the next file

                    # Update the input list for each page image
                    converted_files = os.listdir(st.session_state['dir_uploaded_images'])
                    for file_name in converted_files:   
                        if file_name.split('.')[1].lower() in ['jpg', 'jpeg']:
                            ind_small += 1
                            jpg_file_path = os.path.join(st.session_state['dir_uploaded_images'], file_name)
                            st.session_state['input_list'].append(jpg_file_path)

                            if ind_small < MAX_GALLERY_IMAGES + 5:
                                # Create a thumbnail for the gallery
                                img = Image.open(jpg_file_path)
                                img.thumbnail((GALLERY_IMAGE_SIZE, GALLERY_IMAGE_SIZE), Image.Resampling.LANCZOS)
                                file_path_small = save_uploaded_file(st.session_state['dir_uploaded_images_small'], jpg_file_path, img)
                                st.session_state['input_list_small'].append(file_path_small)

                except Exception as e:
                    st.error(f"Failed to process PDF file {uploaded_file.name}. Error: {e}")
                    continue  # Skip to the next file

            else:
                # Handle JPG/JPEG files (existing process)
                ind_small += 1
                st.session_state['input_list'].append(file_path)
                if ind_small < MAX_GALLERY_IMAGES + 5:
                    img = Image.open(file_path)
                    img.thumbnail((GALLERY_IMAGE_SIZE, GALLERY_IMAGE_SIZE), Image.Resampling.LANCZOS)
                    file_path_small = save_uploaded_file(st.session_state['dir_uploaded_images_small'], uploaded_file, img)
                    st.session_state['input_list_small'].append(file_path_small)

        # After processing all files
        st.session_state.config['leafmachine']['project']['dir_images_local'] = st.session_state['dir_uploaded_images']
        st.info(f"Processing images from {st.session_state.config['leafmachine']['project']['dir_images_local']}")
    
    if st.session_state['input_list_small']:
        if len(st.session_state['input_list_small']) > MAX_GALLERY_IMAGES:
            images_to_display = st.session_state['input_list_small'][:MAX_GALLERY_IMAGES]
        else:
            images_to_display = st.session_state['input_list_small']
        show_gallery_small_hf(images_to_display)
    
    return False

# def handle_image_upload_and_gallery_hf(uploaded_files): # not working with pdfs
#     SAFE = SafetyCheck(st.session_state['is_hf'])
#     if uploaded_files:
        
#         # Clear input image gallery and input list
#         clear_image_uploads()

#         ind_small = 0
#         for uploaded_file in uploaded_files:

#             if SAFE.check_for_inappropriate_content(uploaded_file):
#                 clear_image_uploads()
#                 report_violation(uploaded_file.name, is_hf=st.session_state['is_hf'])
#                 st.error("Warning: You uploaded an image that violates our terms of service.")
#                 return True

            
#             # Determine the file type
#             if uploaded_file.name.lower().endswith('.pdf'):
#                 # Handle PDF files
#                 file_path = save_uploaded_file(st.session_state['dir_uploaded_images'], uploaded_file)
#                 # Convert each page of the PDF to an image
#                 n_pages = convert_pdf_to_jpg(file_path, st.session_state['dir_uploaded_images'], dpi=200)#st.session_state.config['leafmachine']['project']['dir_images_local'])
#                 # Update the input list for each page image
#                 converted_files = os.listdir(st.session_state['dir_uploaded_images'])
#                 for file_name in converted_files:   
#                     if file_name.split('.')[1].lower() in ['jpg','jpeg']:
#                         ind_small += 1
#                         jpg_file_path = os.path.join(st.session_state['dir_uploaded_images'], file_name)
#                         st.session_state['input_list'].append(jpg_file_path)

#                         if ind_small < MAX_GALLERY_IMAGES +5:
#                             # Optionally, create a thumbnail for the gallery
#                             img = Image.open(jpg_file_path)
#                             img.thumbnail((GALLERY_IMAGE_SIZE, GALLERY_IMAGE_SIZE), Image.Resampling.LANCZOS)
#                             try:
#                                 file_path_small = save_uploaded_file(st.session_state['dir_uploaded_images_small'], file_name, img)
#                             except:
#                                 file_path_small = save_uploaded_file_local(st.session_state['dir_uploaded_images_small'],st.session_state['dir_uploaded_images_small'], file_name, img)
#                             st.session_state['input_list_small'].append(file_path_small)
                
#             else:
#                 ind_small += 1
#                 # Handle JPG/JPEG files (existing process)
#                 file_path = save_uploaded_file(st.session_state['dir_uploaded_images'], uploaded_file)
#                 st.session_state['input_list'].append(file_path)
#                 if ind_small < MAX_GALLERY_IMAGES +5:
#                     img = Image.open(file_path)
#                     img.thumbnail((GALLERY_IMAGE_SIZE, GALLERY_IMAGE_SIZE), Image.Resampling.LANCZOS)
#                     file_path_small = save_uploaded_file(st.session_state['dir_uploaded_images_small'], uploaded_file, img)
#                     st.session_state['input_list_small'].append(file_path_small)

#         # After processing all files
#         st.session_state.config['leafmachine']['project']['dir_images_local'] = st.session_state['dir_uploaded_images']
#         st.info(f"Processing images from {st.session_state.config['leafmachine']['project']['dir_images_local']}")
    
#     if st.session_state['input_list_small']:
#         if len(st.session_state['input_list_small']) > MAX_GALLERY_IMAGES:
#             # Only take the first 100 images from the list
#             images_to_display = st.session_state['input_list_small'][:MAX_GALLERY_IMAGES]
#         else:
#             # If there are less than 100 images, take them all
#             images_to_display = st.session_state['input_list_small']
#         show_gallery_small_hf(images_to_display)
    
#     return False


def handle_image_upload_and_gallery():
    
    if st.session_state['view_local_gallery'] and st.session_state['input_list_small'] and (st.session_state['dir_images_local_TEMP'] == st.session_state.config['leafmachine']['project']['dir_images_local']):
        if MAX_GALLERY_IMAGES <= st.session_state['processing_add_on']:
            info_txt = f"Showing {MAX_GALLERY_IMAGES} out of {st.session_state['processing_add_on']} images"
        else:
            info_txt = f"Showing {st.session_state['processing_add_on']} out of {st.session_state['processing_add_on']} images"
        st.info(info_txt)
        try:
            show_gallery_small()
        except:
            pass

    elif not st.session_state['view_local_gallery'] and st.session_state['input_list_small'] and (st.session_state['dir_images_local_TEMP'] == st.session_state.config['leafmachine']['project']['dir_images_local']):
        pass
    elif not st.session_state['view_local_gallery'] and not st.session_state['input_list_small'] and (st.session_state['dir_images_local_TEMP'] == st.session_state.config['leafmachine']['project']['dir_images_local']):
        pass
    # elif st.session_state['input_list_small'] and (st.session_state['dir_images_local_TEMP'] != st.session_state.config['leafmachine']['project']['dir_images_local']):
    elif (st.session_state['dir_images_local_TEMP'] != st.session_state.config['leafmachine']['project']['dir_images_local']):
        has_pdf = False
        clear_image_uploads()

        directory = st.session_state.config['leafmachine']['project']['dir_images_local']
        for input_file in os.listdir():
            fpath = os.path.join(directory, input_file)

            if os.path.isfile(fpath):
                if input_file.split('.')[1].lower() in ['jpg','jpeg']:
                    pass
                
                elif input_file.split('.')[1].lower() in ['pdf',]:
                    has_pdf = True
                    # Handle PDF files
                    file_path = save_uploaded_file_local(st.session_state.config['leafmachine']['project']['dir_images_local'], st.session_state['dir_uploaded_images'], input_file)
                    # Convert each page of the PDF to an image
                    n_pages = convert_pdf_to_jpg(file_path, st.session_state['dir_uploaded_images'], dpi=200)#st.session_state.config['leafmachine']['project']['dir_images_local'])
                else:
                    pass
            else:
                pass
                # st.warning("Inputs must be '.PDF' or '.jpg' or '.jpeg'")
        if has_pdf:
            st.session_state.config['leafmachine']['project']['dir_images_local'] = st.session_state['dir_uploaded_images']

        dir_images_local = st.session_state.config['leafmachine']['project']['dir_images_local']
        count_n_imgs = list_jpg_files(dir_images_local)
        st.session_state['processing_add_on'] = count_n_imgs
        # print(st.session_state['processing_add_on'])
        st.session_state['dir_images_local_TEMP'] = st.session_state.config['leafmachine']['project']['dir_images_local']
        print("rerun")
        st.rerun()


def content_input_images(col_left, col_right):
    
    st.write('---')
    # col1, col2 = st.columns([2,8])
    with col_left:
        st.header('Input Images')
        if not st.session_state.is_hf:

            ### Input Images Local
            st.session_state.config['leafmachine']['project']['dir_images_local'] = st.text_input("Input images directory", st.session_state.config['leafmachine']['project'].get('dir_images_local', ''))
        
            st.session_state.config['leafmachine']['project']['continue_run_from_partial_xlsx'] = st.text_input("Continue run from partially completed project XLSX", st.session_state.config['leafmachine']['project'].get('continue_run_from_partial_xlsx', ''), disabled=True)
        else:
            pass
    
    with col_left:
        if st.session_state.is_hf:
            st.session_state['dir_uploaded_images'] = os.path.join(st.session_state.dir_home,'uploads')
            st.session_state['dir_uploaded_images_small'] = os.path.join(st.session_state.dir_home,'uploads_small')
            uploaded_files = st.file_uploader("Upload Images", type=['jpg', 'jpeg','pdf'], accept_multiple_files=True, key=st.session_state['uploader_idk'])
            st.button("Use Test Image",help="This will clear any uploaded images and load the 1 provided test image.",on_click=use_test_image)
    
    with col_right:
        if st.session_state.is_hf:
            result = handle_image_upload_and_gallery_hf(uploaded_files)

        else:
            st.session_state['view_local_gallery'] = st.toggle("View Image Gallery",)
            handle_image_upload_and_gallery()

def list_jpg_files(directory_path):
    jpg_count = 0
    clear_image_gallery()
    st.session_state['input_list_small'] = []

    if not os.path.isdir(directory_path):
        return None
    
    jpg_count = count_jpg_images(directory_path)

    jpg_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
                if len(jpg_files) == MAX_GALLERY_IMAGES:
                    break
        if len(jpg_files) == MAX_GALLERY_IMAGES:
            break
            
    for simg in jpg_files:
        simg2 = Image.open(simg)
        simg2.thumbnail((GALLERY_IMAGE_SIZE, GALLERY_IMAGE_SIZE), Image.Resampling.LANCZOS)  
        file_path_small = save_uploaded_local(st.session_state['dir_uploaded_images_small'], simg, simg2)
        st.session_state['input_list_small'].append(file_path_small)
    return jpg_count


def count_jpg_images(directory_path):
    if not os.path.isdir(directory_path):
        return None

    jpg_count = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_count += 1

    return jpg_count


def create_download_button(zip_filepath, col, key):
    with col:
        # labal_n_images = f"Download Results for {st.session_state['processing_add_on']} Images"
        labal_n_images = f"Download Results"
        with open(zip_filepath, 'rb') as f:
            bytes_io = BytesIO(f.read())
        st.download_button(
            label=labal_n_images,
            type='primary',
            data=bytes_io,
            file_name=os.path.basename(zip_filepath),
            mime='application/zip',
            use_container_width=True,key=key,
        )


def delete_directory(dir_path):
    try:
        shutil.rmtree(dir_path)
        st.session_state['input_list'] = []
        st.session_state['input_list_small'] = []
        # st.success(f"Deleted previously uploaded images, making room for new images: {dir_path}")
    except OSError as e:
        st.error(f"Error: {dir_path} : {e.strerror}")


def clear_image_gallery():
    delete_directory(st.session_state['dir_uploaded_images_small'])
    validate_dir(st.session_state['dir_uploaded_images_small'])

def clear_image_uploads():
    delete_directory(st.session_state['dir_uploaded_images'])
    delete_directory(st.session_state['dir_uploaded_images_small'])
    validate_dir(st.session_state['dir_uploaded_images'])
    validate_dir(st.session_state['dir_uploaded_images_small'])


def use_test_image():
    st.info(f"Processing images from {os.path.join(st.session_state.dir_home,'demo','demo_images')}")
    st.session_state.config['leafmachine']['project']['dir_images_local'] = os.path.join(st.session_state.dir_home,'demo','demo_images')
    n_images = len([f for f in os.listdir(st.session_state.config['leafmachine']['project']['dir_images_local']) if os.path.isfile(os.path.join(st.session_state.config['leafmachine']['project']['dir_images_local'], f))])
    st.session_state['processing_add_on'] = n_images
    clear_image_uploads()
    st.session_state['uploader_idk'] += 1
    for file in os.listdir(st.session_state.config['leafmachine']['project']['dir_images_local']):
        try:
            file_path = save_uploaded_file(os.path.join(st.session_state.dir_home,'demo','demo_images'), file)
        except:
            file_path = save_uploaded_file_local(os.path.join(st.session_state.dir_home,'demo','demo_images'),os.path.join(st.session_state.dir_home,'demo','demo_images'), file)

        st.session_state['input_list'].append(file_path)

        img = Image.open(file_path)
        img.thumbnail((GALLERY_IMAGE_SIZE, GALLERY_IMAGE_SIZE), Image.Resampling.LANCZOS)  
        try:
            file_path_small = save_uploaded_file(st.session_state['dir_uploaded_images_small'], file, img)
        except:
            file_path_small = save_uploaded_file_local(st.session_state['dir_uploaded_images_small'],st.session_state['dir_uploaded_images_small'], file, img)
        st.session_state['input_list_small'].append(file_path_small)


def refresh():
    st.session_state['uploader_idk'] += 1
    st.write('')



    

# def display_image_gallery():
#     # Initialize the container
#     con_image = st.empty()
    
#     # Start the div for the image grid
#     img_grid_html = """
#     <div style='display: flex; flex-wrap: wrap; align-items: flex-start; overflow-y: auto; max-height: 400px; gap: 10px;'>
#     """
    
#     # Loop through each image in the input list
#     # with con_image.container():
#     for image_path in st.session_state['input_list']:
#         # Open the image and create a thumbnail
#         img = Image.open(image_path)
#         img.thumbnail((120, 120), Image.Resampling.LANCZOS)  

#         # Convert the image to base64
#         base64_image = image_to_base64(img)

#         # Append the image to the grid HTML
#         # img_html = f"""
#         #     <div style='display: flex; flex-wrap: wrap; overflow-y: auto; max-height: 400px;'>
#         #         <img src='data:image/jpeg;base64,{base64_image}' alt='Image' style='max-width: 100%; height: auto;'>
#         #     </div>
#         #     """
#         img_html = f"""
#                 <img src='data:image/jpeg;base64,{base64_image}' alt='Image' style='max-width: 100%; height: auto;'>
#             """
#         img_grid_html += img_html
#         # st.markdown(img_html, unsafe_allow_html=True)

    
#     # Close the div for the image grid
#     img_grid_html += "</div>"
    
#     # Display the image grid in the container
#     with con_image.container():
#         st.markdown(img_grid_html, unsafe_allow_html=True)

#     # The CSS to make the images display inline and be responsive
#     css = """
#     <style>
#         .scrollable-image-container img {
#             max-width: 100%;
#             height: auto;
#         }
#     </style>
#     """
#     # Apply the CSS
#     st.markdown(css, unsafe_allow_html=True)
########################################################################################################
########################################################################################################
########################################################################################################
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
                st.map(map_data, zoom=4, size='size', color='color', use_container_width=True)

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
    # rain_emojis(test_results)



def add_emoji_delay():
    time.sleep(0.3)



# def rain_emojis(test_results):
#     # test_results = {
#     #     'test1': True,   # Test passed
#     #     'test2': True,   # Test passed
#     #     'test3': True,   # Test passed
#     #     'test4': False,  # Test failed
#     #     'test5': False,  # Test failed
#     #     'test6': False,  # Test failed
#     #     'test7': False,  # Test failed
#     #     'test8': False,  # Test failed
#     #     'test9': False,  # Test failed
#     #     'test10': False,  # Test failed
#     # }
#     success_emojis = ["🥇", "🏆", "🍾", "🙌"]
#     failure_emojis = ["💔", "😭"]

#     success_count = sum(1 for result in test_results.values() if result)
#     failure_count = len(test_results) - success_count

#     chosen_emoji = random.choice(success_emojis)
#     for _ in range(success_count):
#         rain(
#             emoji=chosen_emoji,
#             font_size=72,
#             falling_speed=4,
#             animation_length=2,
#         )
#         add_emoji_delay()

#     chosen_emoji = random.choice(failure_emojis)
#     for _ in range(failure_count):
#         rain(
#             emoji=chosen_emoji,
#             font_size=72,
#             falling_speed=5,
#             animation_length=1,
#         )
#         add_emoji_delay()



def format_json(json_obj):
    try:
        return json.dumps(json.loads(json_obj), indent=4, sort_keys=False)
    except:
        return json.dumps(json_obj, indent=4, sort_keys=False)
    


def get_prompt_versions(LLM_version):
    yaml_files = [f for f in os.listdir(os.path.join(st.session_state.dir_home, 'custom_prompts')) if f.endswith('.yaml')]

    return yaml_files



def get_private_file():
    dir_home = os.path.dirname(__file__)
    path_cfg_private = os.path.join(dir_home, 'PRIVATE_DATA.yaml')
    return get_cfg_from_full_path(path_cfg_private)

def blog_text_and_image(text=None, fullpath=None, width=700):
    if text:
        st.markdown(f"{text}")
    if fullpath:
        st.session_state.logo = Image.open(fullpath)
        st.image(st.session_state.logo, width=width)

def blog_text(text_bold, text):
    st.markdown(f"- **{text_bold}**{text}")
def blog_text_plain(text_bold, text):
    st.markdown(f"**{text_bold}** {text}")

def create_private_file(): 
    section_left = 2
    section_mid = 6
    section_right = 2
    
    st.session_state.proceed_to_main = False
    st.title("VoucherVision")
    _, col_private,__= st.columns([section_left,section_mid, section_right])

    

    if st.session_state.private_file:
        cfg_private = get_private_file()
    else:
        cfg_private = {}
        cfg_private['openai'] = {}
        cfg_private['openai']['OPENAI_API_KEY'] =''
        
        cfg_private['openai_azure'] = {}
        cfg_private['openai_azure']['OPENAI_API_KEY_AZURE'] = ''
        cfg_private['openai_azure']['OPENAI_API_VERSION'] = ''
        cfg_private['openai_azure']['OPENAI_API_BASE'] =''
        cfg_private['openai_azure']['OPENAI_ORGANIZATION'] =''
        cfg_private['openai_azure']['OPENAI_API_TYPE'] =''

        cfg_private['google'] = {}
        cfg_private['google']['GOOGLE_APPLICATION_CREDENTIALS'] =''
        cfg_private['google']['GOOGLE_PALM_API'] =''
        cfg_private['google']['GOOGLE_PROJECT_ID'] =''
        cfg_private['google']['GOOGLE_LOCATION'] =''

        cfg_private['mistral'] = {}
        cfg_private['mistral']['MISTRAL_API_KEY'] =''

        cfg_private['here'] = {}
        cfg_private['here']['APP_ID'] =''
        cfg_private['here']['API_KEY'] =''

        cfg_private['open_cage_geocode'] = {}
        cfg_private['open_cage_geocode']['API_KEY'] =''

        cfg_private['huggingface'] = {}

    with col_private:
        st.header("Set API keys")
        st.warning("To commit changes to API keys you must press the 'Set API Keys' button at the bottom of the page.")
        st.write("Before using VoucherVision you must set your API keys. All keys are stored locally on your computer and are never made public.")
        st.write("API keys are stored in `../VoucherVision/PRIVATE_DATA.yaml`.")
        st.write("Deleting this file will allow you to reset API keys. Alternatively, you can edit the keys in the user interface or by manually editing the `.yaml` file in a text editor.")
        st.write("Leave keys blank if you do not intend to use that service.")
        st.info("Note: You can manually edit these API keys later by opening the /PRIVATE_DATA.yaml file in a plain text editor.")

        st.write("---")
        st.subheader("Hugging Face  (*Required For Local LLMs*)")
        st.markdown("VoucherVision relies on LLM models from Hugging Face. Some models are 'gated', meaning that you have to agree to the creator's usage guidelines.")
        st.markdown("""Create a [Hugging Face account](https://huggingface.co/join). Once your account is created, in your profile settings [navigate to 'Access Tokens'](https://huggingface.co/settings/tokens) and click 'Create new token'. Create a token that has 'Read' privileges. Copy the token into the field below.""")

        hugging_face_token = st.text_input(label = 'Hugging Face token', value = cfg_private['huggingface'].get('hf_token', ''),
                                                placeholder = 'e.g. hf_GNRLIUBnvfkjvnf....',
                                                help ="This is your Hugging Face access token. It only needs Read access. Please see https://huggingface.co/settings/tokens",
                                                type='password')

        st.write("---")
        st.subheader("Google Vision  (*Required*) / Google PaLM 2 / Google Gemini")
        st.markdown("VoucherVision currently uses [Google Vision API](https://cloud.google.com/vision/docs/ocr) for OCR. Generating an API key for this is more involved than the others. [Please carefully follow the instructions outlined here to create and setup your account.](https://cloud.google.com/vision/docs/setup) ")
        st.markdown("""Once your account is created, [visit this page](https://console.cloud.google.com) and create a project. Then follow these instructions:""")

        with st.expander("**View Google API Instructions**"):
        
            blog_text_and_image(text="Select your project, then in the search bar, search for `vertex ai` and select the option in the photo below.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_00.PNG'))
            
            blog_text_and_image(text="On the main overview page, click `Enable All Recommended APIs`. Sometimes this button may be hidden. In that case, enable all of the suggested APIs listed on this page.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_0.PNG'))
            
            blog_text_and_image(text="Sometimes this button may be hidden. In that case, enable all of the suggested APIs listed on this page.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_2.PNG'))
            
            blog_text_and_image(text="Make sure that all APIs are enabled.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_1.PNG'))
            
            blog_text_and_image(text="Find the `Vision AI API` service and go to its page.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_3.PNG'))
            
            blog_text_and_image(text="Find the `Vision AI API` service and go to its page. This is the API service required to use OCR in VoucherVision and must be enabled.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_6.PNG'))
            
            blog_text_and_image(text="You can also search for the Vertex AI Vision service.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_4.PNG'))
            
            blog_text_and_image(text=None, 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_5.PNG'))
            
            st.subheader("Getting a Google JSON authentication key")
            st.write(f"Google uses a JSON file to store additional authentication information. Save this file in a safe, private location and assign the `GOOGLE_APPLICATION_CREDENTIALS` value to the file path. For Hugging Face, copy the contents of the JSON file including the curly brackets and paste it as the secret value.")
            st.write("To download your JSON key...")
            blog_text_and_image(text="Open the navigation menu. Click on the hamburger menu (three horizontal lines) in the top left corner. Go to IAM & Admin. ", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_7.PNG'),width=300)
            
            blog_text_and_image(text="In the navigation pane, hover over `IAM & Admin` and then click on `Service accounts`.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_8.PNG'))
            
            blog_text_and_image(text="Find the default Compute Engine service account, select it.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_9.PNG'))
            
            blog_text_and_image(text="Click `Add Key`.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_10.PNG'))
            
            blog_text_and_image(text="Select `JSON` and click create. This will download your key. Store this in a safe location. The file path to this safe location is the value that you enter into the `GOOGLE_APPLICATION_CREDENTIALS` value.", 
                                fullpath=os.path.join(st.session_state.dir_home, 'demo','google','google_api_11.PNG'))
            
            blog_text(text_bold="Store Safely", text=": This file contains sensitive data that can be used to authenticate and bill your Google Cloud account. Never commit it to public repositories or expose it in any way. Always keep it safe and secure.")

            st.write("Below is an example of the JSON key.")
            st.json({
                "type": "service_account",
                "project_id": "NAME OF YOUR PROJECT",
                "private_key_id": "XXXXXXXXXXXXXXXXXXXXXXXX",
                "private_key": "-----BEGIN PRIVATE KEY-----\naaaaaaaaaaa\n-----END PRIVATE KEY-----\n",
                "client_email": "EMAIL-ADDRESS@developer.gserviceaccount.com",
                "client_id": "ID NUMBER",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "A LONG URL",
                "universe_domain": "googleapis.com"
                })
            
            blog_text('Google project ID', ': The project ID is the "project_id"  value from the JSON file.')
            blog_text('Google project location', ': The project location specifies the location of the Google server that your project resources will utilize. It should not really make a difference which location you choose. We use `us-central1`, but you might want to choose a location closer to where you live. [please see this page for a list of available regions](https://cloud.google.com/vertex-ai/docs/general/locations)')

            
        google_application_credentials = st.text_input(label = 'Full path to Google Cloud JSON API key file', value = cfg_private['google'].get('GOOGLE_APPLICATION_CREDENTIALS', ''),
                                                placeholder = 'e.g. C:/Documents/Secret_Files/google_API/application_default_credentials.json',
                                                help ="This API Key is in the form of a JSON file. Please save the JSON file in a safe directory. DO NOT store the JSON key inside of the VoucherVision directory.",
                                                type='password')
        google_project_location = st.text_input(label = 'Google project location', value = cfg_private['google'].get('GOOGLE_LOCATION', ''),
                                                placeholder = 'e.g. us-central1',
                                                help ="This is the location of where your Google services are operating.",
                                                type='password')
        google_project_id = st.text_input(label = 'Google project ID', value = cfg_private['google'].get('GOOGLE_PROJECT_ID', ''),
                                                placeholder = 'e.g. my-project-name',
                                                help ="This is the value in the `project_id` field in your JSON key.",
                                                type='password')

        
        st.write("---")
        st.subheader("OpenAI")
        st.markdown("API key for first-party OpenAI API. Create an account with OpenAI [here](https://platform.openai.com/signup), then create an API key [here](https://platform.openai.com/account/api-keys).")
        openai_api_key = st.text_input("openai_api_key", cfg_private['openai'].get('OPENAI_API_KEY', ''),
                                                 help='The actual API key. Likely to be a string of 2 character, a dash, and then a 48-character string: sk-XXXXXXXX...',
                                                 placeholder = 'e.g. sk-XXXXXXXX...',
                                                 type='password')


        st.write("---")
        st.subheader("OpenAI - Azure")
        st.markdown("This version OpenAI relies on Azure servers directly as is intended for private enterprise instances of OpenAI's services, such as [UM-GPT](https://its.umich.edu/computing/ai). Administrators will provide you with the following information.")
        azure_openai_api_version = st.text_input("OPENAI_API_VERSION", cfg_private['openai_azure'].get('OPENAI_API_VERSION', ''),
                                                 help='API Version e.g. "2023-05-15"',
                                                 placeholder = 'e.g. 2023-05-15',
                                                 type='password')
        azure_openai_api_key = st.text_input("OPENAI_API_KEY_AZURE", cfg_private['openai_azure'].get('OPENAI_API_KEY_AZURE', ''),
                                                 help='The actual API key. Likely to be a 32-character string. This might also be called "endpoint."',
                                                 placeholder = 'e.g. 12333333333333333333333333333332',
                                                 type='password')
        azure_openai_api_base = st.text_input("OPENAI_API_BASE", cfg_private['openai_azure'].get('OPENAI_API_BASE', ''),
                                                 help='The base url for the API e.g. "https://api.umgpt.umich.edu/azure-openai-api"',
                                                 placeholder = 'e.g. https://api.umgpt.umich.edu/azure-openai-api',
                                                 type='password')
        azure_openai_organization = st.text_input("OPENAI_ORGANIZATION", cfg_private['openai_azure'].get('OPENAI_ORGANIZATION', ''),
                                                 help='Your organization code. Likely a short string.',
                                                 placeholder = 'e.g. 123456',
                                                 type='password')
        azure_openai_api_type = st.text_input("OPENAI_API_TYPE", cfg_private['openai_azure'].get('OPENAI_API_TYPE', ''),
                                                 help='The API type. Typically "azure"',
                                                 placeholder = 'e.g. azure',
                                                 type='password')

        # st.write("---")
        # st.subheader("Google PaLM 2 (Deprecated)")
        # st.write("Plea")
        # st.markdown('Follow these [instructions](https://developers.generativeai.google/tutorials/setup) to generate an API key for PaLM 2. You may need to also activate an account with [MakerSuite](https://makersuite.google.com/app/apikey) and enable "early access." If this is deprecated, then use the full Google API instructions above.')

        # google_palm = st.text_input("Google PaLM 2 API Key", cfg_private['google'].get('GOOGLE_PALM_API', ''),
        #                                          help='The MakerSuite API key e.g. a 32-character string',
        #                                          placeholder='e.g. SATgthsykuE64FgrrrrEervr3S4455t_geyDeGq',
        #                                          type='password')


        st.write("---")
        st.subheader("MistralAI")
        st.markdown('Follow these [instructions](https://console.mistral.ai/) to generate an API key for MistralAI.')
        mistral_API_KEY = st.text_input("MistralAI API Key", cfg_private['mistral'].get('MISTRAL_API_KEY', ''),
                                                 help='e.g. a 32-character string',
                                                 placeholder='e.g. SATgthsykuE64FgrrrrEervr3S4455t_geyDeGq',
                                                 type='password')
                

        st.write("---")
        st.subheader("HERE Geocoding")
        st.markdown('Follow these [instructions](https://platform.here.com/sign-up?step=verify-identity) to generate an API key for HERE.')
        here_APP_ID = st.text_input("HERE Geocoding App ID", cfg_private['here'].get('APP_ID', ''),
                                                 help='e.g. a 32-character string',
                                                 placeholder='e.g. SATgthsykuE64FgrrrrEervr3S4455t_geyDeGq',
                                                 type='password')
        here_API_KEY = st.text_input("HERE Geocoding API Key", cfg_private['here'].get('API_KEY', ''),
                                                 help='e.g. a 32-character string',
                                                 placeholder='e.g. SATgthsykuE64FgrrrrEervr3S4455t_geyDeGq',
                                                 type='password')



        st.button("Set API Keys",type='primary', on_click=save_changes_to_API_keys, 
                    args=[cfg_private,
                        openai_api_key,
                        hugging_face_token,
                        azure_openai_api_version, azure_openai_api_key, azure_openai_api_base, azure_openai_organization, azure_openai_api_type,
                        google_application_credentials, google_project_location, google_project_id,
                        mistral_API_KEY, 
                        here_APP_ID, here_API_KEY])
        if st.button('Proceed to VoucherVision'):
            st.session_state.private_file = does_private_file_exist()
            st.session_state.proceed_to_private = False
            st.session_state.proceed_to_main = True
            st.rerun()
       

def save_changes_to_API_keys(cfg_private,
                        openai_api_key,
                        hugging_face_token,
                        azure_openai_api_version, azure_openai_api_key, azure_openai_api_base, azure_openai_organization, azure_openai_api_type,
                        google_application_credentials, google_project_location, google_project_id,
                        mistral_API_KEY, 
                        here_APP_ID, here_API_KEY): 
    
    # Update the configuration dictionary with the new values
    cfg_private['huggingface']['hf_token'] = hugging_face_token 

    cfg_private['openai']['OPENAI_API_KEY'] = openai_api_key 

    cfg_private['openai_azure']['OPENAI_API_VERSION'] = azure_openai_api_version
    cfg_private['openai_azure']['OPENAI_API_KEY_AZURE'] = azure_openai_api_key
    cfg_private['openai_azure']['OPENAI_API_BASE'] = azure_openai_api_base
    cfg_private['openai_azure']['OPENAI_ORGANIZATION'] = azure_openai_organization
    cfg_private['openai_azure']['OPENAI_API_TYPE'] = azure_openai_api_type

    cfg_private['google']['GOOGLE_APPLICATION_CREDENTIALS'] = google_application_credentials
    cfg_private['google']['GOOGLE_PROJECT_ID'] = google_project_id
    cfg_private['google']['GOOGLE_LOCATION'] = google_project_location 

    cfg_private['mistral']['MISTRAL_API_KEY'] = mistral_API_KEY

    cfg_private['here']['APP_ID'] = here_APP_ID
    cfg_private['here']['API_KEY'] = here_API_KEY
    # Call the function to write the updated configuration to the YAML file
    write_config_file(cfg_private, st.session_state.dir_home, filename="PRIVATE_DATA.yaml")
    st.success(f"API Keys saved to {os.path.join(st.session_state.dir_home, 'PRIVATE_DATA.yaml')}")
    # st.session_state.private_file = does_private_file_exist()

# Function to load a YAML file and update session_state


### Updated to match HF version
# def save_prompt_yaml(filename):



@st.cache_data
def show_header_welcome():
    st.session_state.logo_path = os.path.join(st.session_state.dir_home, 'img','logo.png')
    st.session_state.logo = Image.open(st.session_state.logo_path)
    st.image(st.session_state.logo, width=250)

def determine_n_images():
    try:
        # Check if 'dir_uploaded_images' key exists in session state and it is not empty
        if 'dir_uploaded_images' in st.session_state and st.session_state['dir_uploaded_images']:
            dir_path = st.session_state['dir_uploaded_images']  # This would be the path to the directory
            # Count only files (not directories) in the specified directory
            count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            return count
        else:
            return None  # Return 0 if the directory path doesn't exist or is empty
    except Exception as e:
        print(e)
        return None
# def determine_n_images():
#     try:
#         # Check if 'dir_uploaded_images' key exists and it is not empty
#         if 'dir_uploaded_images' in st and st['dir_uploaded_images']:
#             dir_path = st['dir_uploaded_images']  # This would be the path to the directory
#             return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
#         else:
#             return None
#     except:
#         return None

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
    
def display_api_key_status(ccol):
    if not st.session_state['API_checked']:
        present_keys, missing_keys, date_of_check = load_api_status()
        if present_keys is None and missing_keys is None:
            st.session_state['API_checked'] = False
        else:
            # Convert keys to annotations (similar to what you do in check_api_key_status)
            present_annotations = []
            missing_annotations = []
            model_annotations = []
            for key in present_keys:
                if "[MODEL]" in key:
                    show_text = key.split(']')[1]
                    show_text = show_text.split('(')[0]
                    if 'Under Review' in key:
                        model_annotations.append((show_text, "under review", "#9C0586"))  # Green for valid
                    elif 'invalid' in key:
                        model_annotations.append((show_text, "error!", "#870307"))  # Green for valid
                    else:
                        model_annotations.append((show_text, "ready!", "#059c1b"))  # Green for valid

                elif "Valid" in key:
                    show_text = key.split('(')[0]
                    present_annotations.append((show_text, "ready!", "#059c1b"))  # Green for valid
                elif "Invalid" in key:
                    show_text = key.split('(')[0]
                    present_annotations.append((show_text, "error", "#870307"))  # Red for invalid

            st.session_state['present_annotations'] = present_annotations
            st.session_state['missing_annotations'] = missing_annotations
            st.session_state['model_annotations'] = model_annotations
            st.session_state['date_of_check'] = date_of_check
            st.session_state['API_checked'] = True
            # print('for')
            # print(st.session_state['present_annotations'])
            # print(st.session_state['missing_annotations'])
    else:
        # print('else')
        # print(st.session_state['present_annotations'])
        # print(st.session_state['missing_annotations'])
        pass

    # Check if the API status has already been retrieved
    if 'API_checked' not in st.session_state or not st.session_state['API_checked'] or st.session_state['API_rechecked']:
        with ccol:
            with st.spinner('Verifying APIs by sending short requests...'):
                check_api_key_status()
        st.session_state['API_checked'] = True
        st.session_state['API_rechecked'] = False

    st.markdown(f"Last checked on {st.session_state['date_of_check']}")
    # Display present keys horizontally
    if 'present_annotations' in st.session_state and st.session_state['present_annotations']:
        annotated_text(*st.session_state['present_annotations'])

    # Display missing keys horizontally
    if 'missing_annotations' in st.session_state and st.session_state['missing_annotations']:
        annotated_text(*st.session_state['missing_annotations'])
    
    if not st.session_state['is_hf']:
        st.markdown(f"Access to Hugging Face Models")
        
        if 'model_annotations' in st.session_state and st.session_state['model_annotations']:
            annotated_text(*st.session_state['model_annotations'])

    
    


def check_api_key_status():
    try:
        path_cfg_private = os.path.join(st.session_state.dir_home, 'PRIVATE_DATA.yaml')
        cfg_private = get_cfg_from_full_path(path_cfg_private)
    except:
        cfg_private = None

    API_Validator = APIvalidation(cfg_private, st.session_state.dir_home, st.session_state['is_hf'])
    present_keys, missing_keys, date_of_check = API_Validator.report_api_key_status() 

    # Prepare annotations for present keys
    present_annotations = []
    missing_annotations = []
    model_annotations = []
    for key in present_keys:
        if "[MODEL]" in key:
            show_text = key.split(']')[1]
            show_text = show_text.split('(')[0]
            if 'Under Review' in key:
                model_annotations.append((show_text, "under review", "#9C0586"))  # Green for valid
            elif 'invalid' in key:
                model_annotations.append((show_text, "error!", "#870307"))  # Green for valid
            else:
                model_annotations.append((show_text, "ready!", "#059c1b"))  # Green for valid

        elif "Valid" in key:
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

    st.session_state['present_annotations'] = present_annotations
    st.session_state['missing_annotations'] = missing_annotations
    st.session_state['model_annotations'] = model_annotations
    st.session_state['date_of_check'] = date_of_check
    

def convert_cost_dict_to_table(cost, name):
    # Convert the dictionary to a pandas DataFrame for nicer display
    df = pd.DataFrame.from_dict(cost, orient='index')
    df.reset_index(inplace=True)
    df.columns = [str(name), 'Input', 'Output'] 


    # Apply color gradient
    cm = sns.light_palette("green", as_cmap=True, reverse=True)
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
    cost_hyper = {}
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
        elif 'Hyperbolic' in parts:
            cost_hyper[key] = cost_data.get(value,'')
        elif ('MISTRAL' in parts) or ('MIXTRAL' in parts) or ('PIXTRAL' in parts) or ('MINISTRAL' in parts):
            cost_mistral[key] = cost_data.get(value,'')

    styled_cost_openai = convert_cost_dict_to_table(cost_openai, "OpenAI")
    styled_cost_azure = convert_cost_dict_to_table(cost_azure, "OpenAI (Azure Endpoints)")
    styled_cost_google = convert_cost_dict_to_table(cost_google, "Google (VertexAI)")
    styled_cost_mistral = convert_cost_dict_to_table(cost_mistral, "MistralAI")
    styled_cost_local = convert_cost_dict_to_table(cost_local, "Local Models")
    styled_cost_hyper = convert_cost_dict_to_table(cost_hyper, "Hyperbolic Hosted Models")

    return cost_openai, styled_cost_openai, cost_azure, styled_cost_azure, cost_google, styled_cost_google, cost_mistral, styled_cost_mistral, cost_local, styled_cost_local, cost_hyper, styled_cost_hyper


def content_header():
    col_logo, col_run_1, col_run_2, col_run_3, col_run_4 = st.columns([2,2,2,2,4])
    with col_run_4:
        with st.expander("View Messages and Updates"):
            st.info("***Note:*** If you use VoucherVision frequently, you can change the default values that are auto-populated in the form below. In a text editor or IDE, edit the first few rows in the file `../VoucherVision/vouchervision/VoucherVision_Config_Builder.py`")
        st.info("Please enable LeafMachine2 collage for full-sized images of herbarium vouchers, you will get better results! If your image is primarily text (like a flora or book page) then disable the collage.")
    
    col_test = st.container()

    st.subheader("Overall Progress")
    col_run_info_1 = st.columns([1])[0]
    col_updates_1, col_updates_2 = st.columns([5,1])
    col_json, col_json_WFO, col_json_GEO, col_json_map = st.columns([2, 2, 2, 2])

    with col_run_info_1:
        # Progress
        overall_progress_bar = st.progress(0)
        text_overall = st.empty()  # Placeholder for current step name
        st.subheader('Transcription Progress')
        batch_progress_bar = st.progress(0)
        text_batch = st.empty()  # Placeholder for current step name
        progress_report = ProgressReport(overall_progress_bar, batch_progress_bar, text_overall, text_batch)
        st.session_state['hold_output'] = st.toggle('View Final Transcription')

    with col_logo:
        show_header_welcome()
    
    with col_run_1:
        N_STEPS = 6

        if check_if_usable(is_hf=st.session_state['is_hf']):
            # b_text = f"Start Processing {st.session_state['processing_add_on']} Images" if st.session_state['processing_add_on'] > 1 else f"Start Processing {st.session_state['processing_add_on']} Image"
            # if st.session_state['processing_add_on'] == 0:
            b_text = f"Start Transcription"
            if st.button(b_text, type='primary',use_container_width=True):
                st.session_state['formatted_json'] = {}
                st.session_state['formatted_json_WFO'] = {}
                st.session_state['formatted_json_GEO'] = {}
                st.session_state['json_report'] = JSONReport(col_updates_1, col_json, col_json_WFO, col_json_GEO, col_json_map)
                st.session_state['json_report'].set_JSON(st.session_state['formatted_json'], st.session_state['formatted_json_WFO'], st.session_state['formatted_json_GEO'])
                
                # Define number of overall steps
                progress_report.set_n_overall(N_STEPS)
                progress_report.update_overall(f"Starting VoucherVision...")
            
                # First, write the config file.
                write_config_file(st.session_state.config, st.session_state.dir_home, filename="VoucherVision.yaml")

                path_custom_prompts = os.path.join(st.session_state.dir_home,'custom_prompts',st.session_state.config['leafmachine']['project']['prompt_version'])
                # Call the machine function.
                total_cost = 0.00
                n_failed_OCR = 0
                n_failed_LLM_calls = 0
                # try:
                voucher_vision_output = voucher_vision(None,
                                                    st.session_state.dir_home, 
                                                    path_custom_prompts, 
                                                    None, 
                                                    progress_report,
                                                    st.session_state['json_report'],
                                                    path_api_cost=os.path.join(st.session_state.dir_home,'api_cost','api_cost.yaml'),
                                                    is_hf = st.session_state['is_hf'], 
                                                    is_real_run=True)
                st.session_state['formatted_json'] = voucher_vision_output['last_JSON_response']
                st.session_state['formatted_json_WFO'] = voucher_vision_output['final_WFO_record']
                st.session_state['formatted_json_GEO'] = voucher_vision_output['final_GEO_record']
                total_cost = voucher_vision_output['total_cost']
                n_failed_OCR = voucher_vision_output['n_failed_OCR']
                n_failed_LLM_calls = voucher_vision_output['n_failed_LLM_calls']
                st.session_state['zip_filepath'] = voucher_vision_output['zip_filepath']
                # st.balloons()

                # except Exception as e:
                #     with col_run_4:
                #         st.error(f"Transcription failed. Error: {e}")

                if n_failed_OCR > 0:
                    with col_run_4:
                        st.error(f"Caution:heavy_exclamation_mark: :loudspeaker: {n_failed_LLM_calls} images had a no extractable OCR text :eyes:")

                if n_failed_LLM_calls > 0:
                    with col_run_4:
                        st.error(f"Caution:heavy_exclamation_mark: :loudspeaker: {n_failed_LLM_calls} images had a failed LLM API call :eyes:")
                        st.error(f"Make sure that you have access to the chosen LLM API model. Sometimes certain OpenAI accounts do not have access to all models, for example")
                
                if total_cost:
                    with col_run_4:
                        st.success(f":money_with_wings: This run cost :heavy_dollar_sign:{total_cost:.4f}")
                else:
                    with col_run_4:
                        st.info(f":money_with_wings: This run cost :heavy_dollar_sign:{total_cost:.4f}")
            if st.session_state['zip_filepath']:
                create_download_button(st.session_state['zip_filepath'], col_run_1,key=97863332)
        else:
            st.button("Start Transcription", type='primary', disabled=True)
            with col_run_4:
                st.error(":heavy_exclamation_mark: Required API keys not set. Please visit the 'API Keys' tab and set the Google Vision OCR API key and at least one LLM key.")
      
        if st.session_state['formatted_json']:
            if st.session_state['hold_output']:
                st.session_state['json_report'].set_JSON(st.session_state['formatted_json'], st.session_state['formatted_json_WFO'], st.session_state['formatted_json_GEO'])
                if st.session_state['zip_filepath']:
                    create_download_button(st.session_state['zip_filepath'], col_run_1,key=978633452)
    
 
    
    with col_run_1:
        ct_left, ct_right = st.columns([1,1])
    with ct_left:
        st.button("Refresh", on_click=refresh, use_container_width=True)
    with ct_right:
        try:
            st.page_link(os.path.join("pages","faqs.py"), label="FAQs", icon="❔")
        except:
            st.page_link(os.path.join(os.path.dirname(__file__),"pages","faqs.py"), label="FAQs", icon="❔")

      

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
                        st.warning(message)



def content_project_settings(col):
         ### Project
    with col:
        st.header('Project Settings')

        st.session_state.config['leafmachine']['project']['run_name'] = st.text_input("Run name", st.session_state.config['leafmachine']['project'].get('run_name', ''),key=63456)

        if not st.session_state.is_hf:
            st.session_state.config['leafmachine']['project']['dir_output'] = st.text_input("Output directory", st.session_state.config['leafmachine']['project'].get('dir_output', ''))
        

def content_tools():
    st.write("---")
    st.header('Validation Tools')    
    
    tool_WFO = st.session_state.config['leafmachine']['project']['tool_WFO']
    st.session_state.config['leafmachine']['project']['tool_WFO'] = st.checkbox(label="Enable World Flora Online taxonomy verification",
                                                                                      help="",
                                                                                      value=tool_WFO)
    
    tool_GEO = st.session_state.config['leafmachine']['project']['tool_GEO']
    st.session_state.config['leafmachine']['project']['tool_GEO'] = st.checkbox(label="Enable HERE geolocation hints",
                                                                                      help="",
                                                                                      value=tool_GEO)

    tool_wikipedia = st.session_state.config['leafmachine']['project']['tool_wikipedia']
    st.session_state.config['leafmachine']['project']['tool_wikipedia'] = st.checkbox(label="Enable Wikipedia verification",
                                                                                      help="",
                                                                                      value=tool_wikipedia)

def content_llm_cost():
    st.write("---")
    st.header('LLM Cost Calculator')
    # ( n_in/1000 * Input + n_out/1000 * Output ) * n_img = COST
    calculator_1,calculator_2,calculator_3,calculator_4,calculator_5 = st.columns([1,1,1,1,1])     

    st.subheader('Cost Matrix')
    st.markdown('The table shows the $USD cost of each LLM API per 1 million tokens. An average VoucherVision call uses 2,000 input tokens and receives 500 output tokens.')
    col_cost_1, col_cost_2, col_cost_3, col_cost_4 = st.columns([1,1,1,1])    

    # Load all cost tables if not already done
    if 'all_llm_cost' not in st.session_state:
        st.session_state['all_llm_cost'] = True
        st.session_state['cost_openai'], st.session_state['styled_cost_openai'], st.session_state['cost_azure'], st.session_state['styled_cost_azure'], st.session_state['cost_google'], st.session_state['styled_cost_google'], st.session_state['cost_mistral'], st.session_state['styled_cost_mistral'], st.session_state['cost_local'], st.session_state['styled_cost_local'], st.session_state['cost_hyper'], st.session_state['styled_cost_hyper'] = get_all_cost_tables()

    with calculator_1:
        # Combine all model names into a single list
        model_names = []
        for df in [st.session_state['cost_openai'], st.session_state['cost_azure'], st.session_state['cost_google'], st.session_state['cost_mistral'], st.session_state['cost_local'], st.session_state['cost_hyper']]:
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
    
    @st.cache_data
    def show_cost_matrix_1(rounding):
        st.dataframe(st.session_state.styled_cost_openai.format(precision=rounding), hide_index=True,)
    @st.cache_data
    def show_cost_matrix_2(rounding):
        st.dataframe(st.session_state.styled_cost_azure.format(precision=rounding), hide_index=True,)
    @st.cache_data
    def show_cost_matrix_3(rounding):
        st.dataframe(st.session_state.styled_cost_google.format(precision=rounding), hide_index=True,)
    @st.cache_data
    def show_cost_matrix_4(rounding):
        st.dataframe(st.session_state.styled_cost_mistral.format(precision=rounding), hide_index=True,)
    @st.cache_data
    def show_cost_matrix_5(rounding):
        st.dataframe(st.session_state.styled_cost_local.format(precision=rounding), hide_index=True,)
    @st.cache_data
    def show_cost_matrix_6(rounding):
        st.dataframe(st.session_state.styled_cost_hyper.format(precision=rounding), hide_index=True,)

    input_value, output_value = find_model_values(selected_model, 
                                                [st.session_state['cost_openai'], st.session_state['cost_azure'], st.session_state['cost_google'], st.session_state['cost_mistral'], st.session_state['cost_local']])
    if input_value is not None and output_value is not None:
        cost = (n_in/1000000 * input_value + n_out/1000000 * output_value) * n_img
    with calculator_5:
        st.text_input("Total Cost", f"${round(cost,2)}") # selected_model
    
    rounding = 2
    with col_cost_1:
        show_cost_matrix_1(rounding)
    with col_cost_2:
        show_cost_matrix_2(rounding)
        show_cost_matrix_3(rounding)
        show_cost_matrix_5(rounding)
    with col_cost_3:
        show_cost_matrix_6(rounding)
    with col_cost_4:
        show_cost_matrix_4(rounding)




def content_prompt_and_llm_version():
    st.info("Note: The default settings may not work for your particular image. If VoucherVision does not produce the results that you were expecting: 1) try disabling the LM2 collage 2) Then try enabling 2 copies of OCR, SLTPvB_long prompt, Azure GPT 4. We are currently building 'recipes' for different scenarios, please stay tuned!")
    st.warning("UPDATE :bell: May 25, 2024 - The default LLM used to be Azure GPT-3.5, which was served by the University of Michigan. However, UofM has sunset all but GPT-4 Turbo so that is now the default LLM. If you ran VV prior to this update and saw an empty result, that was the reason.")
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
        # if st.button("Build Custom LLM Prompt"):
        try:
            st.page_link(os.path.join("pages","prompt_builder.py"), label="Prompt Builder", icon="🚧")
        except:
            st.page_link(os.path.join(os.path.dirname(__file__),"pages","prompt_builder.py"), label="Prompt Builder", icon="🚧")


    # st.header('LLM Version')
    # col_llm_1, col_llm_2 = st.columns([4,2])  
     
    # with col_llm_1:
    #     GUI_MODEL_LIST = ModelMaps.get_models_gui_list()
    #     st.session_state.config['leafmachine']['LLM_version'] = st.selectbox("LLM version", GUI_MODEL_LIST, index=GUI_MODEL_LIST.index(st.session_state.config['leafmachine'].get('LLM_version', ModelMaps.MODELS_GUI_DEFAULT)))
        

    # Determine the default family based on the default model
    default_model = ModelMaps.MODELS_GUI_DEFAULT
    default_family = None
    for family, models in ModelMaps.MODEL_FAMILY.items():
        if default_model in models:
            default_family = family
            break

    st.header("LLM Version")

    col_llm_1, col_llm_2 = st.columns([4, 2])  
    with col_llm_1:
        # Step 1: Select Model Family with default family pre-selected
        family_list = list(ModelMaps.MODEL_FAMILY.keys())
        selected_family = st.selectbox("Select Model Family", family_list, index=family_list.index(default_family) if default_family else 0)

        # Step 2: Display Models based on selected family
        GUI_MODEL_LIST = ModelMaps.get_models_gui_list_family(selected_family)
        
        # Ensure the selected model is part of the current family; if not, use the default of this family
        selected_model_default = st.session_state.config['leafmachine'].get('LLM_version', default_model)
        if selected_model_default not in GUI_MODEL_LIST:
            selected_model_default = GUI_MODEL_LIST[0]

        selected_model = st.selectbox("LLM version", GUI_MODEL_LIST, index=GUI_MODEL_LIST.index(selected_model_default))
        
        # Update the session state with the selected model
        st.session_state.config['leafmachine']['LLM_version'] = selected_model



    
        
        st.markdown("""
Based on preliminary results, the following models perform the best. We are currently running tests of all possible OCR + LLM + Prompt combinations to create recipes for different workflows.
- Any Mistral model e.g., `Mistral Large`          
- `PaLM 2 text-bison@002`
- `GPT 4 Turbo 1106-preview`
- `GPT 3.5 Turbo`
- `LOCAL Mixtral 7Bx8 Instruct`
- `LOCAL Mixtral 7B Instruct`

Larger models (e.g., `GPT 4`, `Gemini Pro`) do not necessarily perform better for these tasks. MistralAI models exceeded our expectations and perform extremely well. PaLM 2 text-bison@001 also seems to consistently out-perform Gemini Pro.
                    
The `SLTPvA_short.yaml` prompt also seems to work better with smaller LLMs (e.g., Mistral Tiny). Alternatively, enable double OCR to help the LLM focus on the OCR text given a longer prompt.
                    
Models `GPT 3.5 Turbo` and `GPT 4 Turbo 0125-preview` enable OpenAI's [JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode), which helps prevent JSON errors. All models implement Langchain JSON parsing too, so JSON errors are rare for most models.""")


def content_api_check():
    # In your Streamlit layout
    # Create two columns for the header and the button
    col_llm_2a, col_llm_2b = st.columns([6, 2])  # Adjust the ratio as needed

    # Place the header in the first column
    with col_llm_2a:
        st.header('Available APIs')

        # Display API key status
        display_api_key_status(col_llm_2a)
    
        # Place the button in the second column, right-justified
        # with col_llm_2b:
        if st.button("Re-Check API Keys"):
            st.session_state['API_checked'] = False
            st.session_state['API_rechecked'] = True
            st.rerun()
        # with col_llm_2c:
        if not st.session_state.is_hf:
            if st.button("Edit API Keys"):
                st.session_state.proceed_to_private = True
                st.rerun()
                


def adjust_ocr_options_based_on_capability(capability_score, model_name='llava'):
    if model_name == 'llava':
        llava_models_requirements = {
            "liuhaotian/llava-v1.6-mistral-7b": {"full": 18, "4bit": 9},
            "liuhaotian/llava-v1.6-34b": {"full": 70, "4bit": 25},
            "liuhaotian/llava-v1.6-vicuna-13b": {"full": 33, "4bit": 15},
            "liuhaotian/llava-v1.6-vicuna-7b": {"full": 20, "4bit": 10},
        }
        if capability_score == 'no_gpu':
            return False
        else:
            capability_score_n = int(capability_score.split("_")[1].split("GB")[0])
            supported_models = [model for model, reqs in llava_models_requirements.items()
                                if reqs["full"] <= capability_score_n or reqs["4bit"] <= capability_score_n]

            # If no models are supported, disable the LLaVA option
            if not supported_models:
                # Assuming the LLaVA option is the last in your list
                return False  # Indicate LLaVA is not supported
            return True  # Indicate LLaVA is supported
    elif model_name == 'florence-2':
        florence_models_requirements = {
            "microsoft/Florence-2-large": {"full": 16,},
            "microsoft/Florence-2-base": {"full": 12,},
        }
        if capability_score == 'no_gpu':
            return False
        else:
            capability_score_n = int(capability_score.split("_")[1].split("GB")[0])
            supported_models = [model for model, reqs in florence_models_requirements.items()
                                if reqs["full"] <= capability_score_n]

            # If no models are supported, disable the model option
            if not supported_models:
                # Assuming the model option is the last in your list
                return False  # Indicate model is not supported
            return True  # Indicate model is supported
        
    elif model_name == 'Qwen7B':
        Qwen_models_requirements = {
            "Qwen/Qwen2-VL-7B-Instruct": {"full": 22,},
            "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8": {"full": 16,},
            "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4": {"full": 12,},
            "Qwen/Qwen2-VL-7B-Instruct-AWQ": {"full": 12},
        }
        if capability_score == 'no_gpu':
            return False
        else:
            capability_score_n = int(capability_score.split("_")[1].split("GB")[0])
            supported_models = [model for model, reqs in Qwen_models_requirements.items()
                                if reqs["full"] <= capability_score_n]

            # If no models are supported, disable the model option
            if not supported_models:
                # Assuming the model option is the last in your list
                return False  # Indicate model is not supported
            return True  # Indicate model is supported
        
    elif model_name == 'Qwen2B':
        Qwen_models_requirements = {
            "Qwen/Qwen2-VL-2B-Instruct": {"full": 10,},
            "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8": {"full": 8,},
            "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4": {"full": 8,},
            "Qwen/Qwen2-VL-2B-Instruct-AWQ": {"full": 8},
        }
        if capability_score == 'no_gpu':
            return False
        else:
            capability_score_n = int(capability_score.split("_")[1].split("GB")[0])
            supported_models = [model for model, reqs in Qwen_models_requirements.items()
                                if reqs["full"] <= capability_score_n]

            # If no models are supported, disable the model option
            if not supported_models:
                # Assuming the model option is the last in your list
                return False  # Indicate model is not supported
            return True  # Indicate model is supported
    



def content_ocr_method():
    st.write("---")
    st.header('OCR Methods')   
    with st.expander("Read about available OCR methods"):
        st.subheader("Overview")
        st.markdown("""VoucherVision can use the `Google Vision API`, `CRAFT` text detection + `trOCR`, and all `LLaVA v1.6` models. 
                    VoucherVision sends the OCR inside of the LLM prompt. We have found that sending multiple copies, or multiple version of 
                    the OCR text to the LLM helps the LLM maintain focus on the OCR text -- our prompts are quite long and the OCR text is reletively short. 
                    Below you can choose the OCR method/s. You can 'stack' all of the methods if you want, which may improve results because
                    different OCR methods have different strengths, giving the LLM more information to work with. Alternative.y, you can select a single method and 
                    send 2 copies to the LLM by enabling that option below.""")
        st.subheader("Google Vision API")
        st.markdown("""`Google Vision API` provides several OCR methods. We use the `document_text_detection()` service, designed to handle dense text blocks. 
                    The `Handwritten` option CAN also be used for printed and mixed labels, but it is also optimized for handwriting. `Handwritten` uses the Google Vision Beta service. 
                    This is the recommended default OCR method. `Printed` uses the regular Google Vision service and works well for general use. 
                    You can also supplement Google Vision OCR by enabling trOCR, which is optimized for handwriting. trOCR requires segmented word images, which is provided as part
                    of the Google Vision metadata. trOCR does not require a GPU, but it runs *much* faster with a GPU.""")
        st.subheader("LLaVA")
        st.markdown("""`LLaVA` can replace Google Vision APIs. It requires the use of LeafMachine2 collage, or images that are majority text. It may struggle with very
                    long texts. LLaVA models are multimodal, meaning that we can upload the image and the model will transcribe (and even parse) the text all at once. With VoucherVision, we 
                    support 4 different LLaVA models of varying sizes, some are much more capable than others. These models tend to outperform all other OCR methods for handwriting. 
                    LLaVA models are run locally and require powerful GPUs to implement. While LLaVA models are capable of handling both the OCR and text parsing tasks all in one step, 
                    this option only uses LLaVA to transcribe all of the text in the image and still uses a separate LLM to parse text in to categories. """)
        st.subheader("CRAFT + trOCR")
        st.markdown("""This pairing can replace Google Vision APIs and is computationally lighter than LLaVA. `CRAFT` locates text, segments lines of text, and feeds the segmentations 
                    to the `trOCR` transformer model. This pairing requires at least an 8 GB GPU. trOCR is a Microsoft model optimized for handwriting. The base model is not as accurate as 
                    LLaVA or Google Vision, but if you have a trOCR-based model, let us know and we will add support.""")

    c1, c2 = st.columns([4,4])   

    with c2:
        st.subheader("Local Methods")
        st.write("Local methods are free, but require a capable GPU. ")
        # Check if LLaVA models are supported based on capability score
        llava_supported = adjust_ocr_options_based_on_capability(st.session_state.capability_score, model_name='llava')
        florence_supported = adjust_ocr_options_based_on_capability(st.session_state.capability_score, model_name='florence-2')
        Qwen7B_supported = adjust_ocr_options_based_on_capability(st.session_state.capability_score, model_name='Qwen7B')
        Qwen2B_supported = adjust_ocr_options_based_on_capability(st.session_state.capability_score, model_name='Qwen2B')

        if llava_supported:
            st.success("LLaVA models are supported on this computer. A GPU with at least 12 GB of VRAM is available.")
        else:
            st.warning("LLaVA models are NOT supported on this computer. Requires a GPU with at least 12 GB of VRAM.")

        if florence_supported:
            st.success("Florence-2 models are supported on this computer. A GPU with at least 12 GB of VRAM is available.")
        else:
            st.warning("Florence-2 models are NOT supported on this computer. Requires a GPU with at least 12 GB of VRAM.")

        if Qwen7B_supported:
            st.success("Qwen-7B models are supported on this computer. A GPU with at least 12 GB of VRAM is available.")
            if not st.session_state.is_hf:
                if not st.session_state.support_qwen_check:

                    try:
                        from transformers import Qwen2VLForConditionalGeneration
                        st.session_state.support_qwen = True
                    except ImportError:
                        st.session_state.support_qwen = False
                    st.session_state.support_qwen_check = True

                if not st.session_state.support_qwen:
                    st.warning("Qwen models require a dev verison of the transformers package. Click the button to install, then restart VoucherVision.")
                    if st.button('Install packages required for Qwen-7B Models',help="As of 9 Sept. 2024 Qwen requires a dev version of the transformers package. Click the button to install it. Then restart VoucherVision."):
                        with st.spinner('Running: [pip install -U "git+https://github.com/huggingface/transformers"] AND [pip install "qwen-vl-utils"]'):
                            install_qwen_requirements()
                        st.error("Please restart VoucherVision to use Qwen models")
        else:
            st.warning("Qwen-7B models are NOT supported on this computer. Requires a GPU with at least 12 GB of VRAM.")

        if Qwen2B_supported:
            st.success("Qwen-2B models are supported on this computer. A GPU with at least 8 GB of VRAM is available.")
            if not st.session_state.is_hf:
                if not st.session_state.support_qwen_check:

                    try:
                        from transformers import Qwen2VLForConditionalGeneration
                        st.session_state.support_qwen = True
                    except ImportError:
                        st.session_state.support_qwen = False
                    st.session_state.support_qwen_check = True

                if not st.session_state.support_qwen:
                    st.warning("Qwen models require a dev verison of the transformers package. Click the button to install, then restart VoucherVision.")
                    if st.button('Install packages required for Qwen-2B Models',help="As of 9 Sept. 2024 Qwen requires a dev version of the transformers package. Click the button to install it. Then restart VoucherVision."):
                        with st.spinner('Running: [pip install -U "git+https://github.com/huggingface/transformers"] AND [pip install "qwen-vl-utils"]'):
                            install_qwen_requirements()
                        st.error("Please restart VoucherVision to use Qwen models")
        else:
            st.warning("Qwen-2B models are NOT supported on this computer. Requires a GPU with at least 8 GB of VRAM.")
            

    demo_text_h = f"Google_OCR_Handwriting:\nHERBARIUM OF MARCUS W. LYON , JR . Tracaulon sagittatum Indiana : Porter Co. incal Springs edge wet subdunal woods 1927 TX 11 Ilowers pink UNIVERSITE HERBARIUM MICH University of Michigan Herbarium 1439649 copyright reserved PERSICARIA FEB 2 6 1965 cm "
    demo_text_tr = f"trOCR:\nherbarium of marcus w. lyon jr. : : : tracaulon sagittatum indiana porter co. incal springs TX 11 Ilowers pink  1439649 copyright reserved D H U Q "
    demo_text_p = f"Google_OCR_Printed:\nTracaulon sagittatum Indiana : Porter Co. incal Springs edge wet subdunal woods 1927  Ilowers pink 1439649 copyright reserved PERSICARIA FEB 2 6 1965 cm "
    demo_text_b = demo_text_h + '\n' + demo_text_p
    demo_text_trb = demo_text_h + '\n' + demo_text_p + '\n' + demo_text_tr
    demo_text_trh = demo_text_h + '\n' + demo_text_tr
    demo_text_trp = demo_text_p + '\n' + demo_text_tr

    options = ["Google Vision Handwritten", 
                "Google Vision Printed", 
                "LOCAL Qwen-2-VL", 
                "LOCAL Florence-2", 
                "GPT-4o-mini", 
                "Hyperbolic Pixtral-12B-2409",
                "Hyperbolic Llama-3.2-90B-Vision-Instruct",
                "Hyperbolic Qwen2-VL-7B-Instruct",
                "Hyperbolic Qwen2-VL-72B-Instruct",
                "CRAFT + trOCR",
                "LLaVA", ]
    options_llava = ["llava-v1.6-mistral-7b", "llava-v1.6-34b", "llava-v1.6-vicuna-13b", "llava-v1.6-vicuna-7b",]
    options_llava_bit = ["full", "4bit",]
    captions_llava = [
        "Full Model: 18 GB VRAM, 4-bit: 9 GB VRAM", 
        "Full Model: 70 GB VRAM, 4-bit: 25 GB VRAM", 
        "Full Model: 33 GB VRAM, 4-bit: 15 GB VRAM",
        "Full Model: 20 GB VRAM, 4-bit: 10 GB VRAM",
    ]
    captions_llava_bit = ["Full Model","4-bit Quantization",]
    # Get the current OCR option from session state
    OCR_option = st.session_state.config['leafmachine']['project']['OCR_option']
    OCR_option_llava = st.session_state.config['leafmachine']['project']['OCR_option_llava']
    OCR_option_llava_bit = st.session_state.config['leafmachine']['project']['OCR_option_llava_bit']
    double_OCR = st.session_state.config['leafmachine']['project']['double_OCR']

    default_index = 0  # Default to 0 if option not found
    default_index_llava = 0  # Default to 0 if option not found
    default_index_llava_bit = 0

    # Map the OCR option to the index in options list
    # You need to define the mapping for multiple OCR options
    # based on your application's logic
    if len(OCR_option) == 1:
        OCR_option = OCR_option[0]
        try:
            default_index = options.index(OCR_option)
        except ValueError:
            pass

    with c1:
        st.subheader("API Methods (Google Vision)")
        st.write("Using APIs for OCR allows VoucherVision to run on most computers. You can use multiple OCR engines simultaneously.")

        st.session_state.config['leafmachine']['project']['double_OCR'] = st.checkbox(label="Send 2 copies of the OCR to the LLM",
                                                                                      help="This can help the LLMs focus attention on the OCR and not get lost in the longer instruction text",
                                                                                      value=double_OCR)

        # Create the radio button
        # OCR_option_select = st.radio(
        #     "Select the OCR Method",
        #     options,
        #     index=default_index,
        #     help="",captions=captions,
        # )
        default_values = [options[default_index]]
        OCR_option_select = st.multiselect(
            "Select the OCR Method(s)",
            options=options,
            default=default_values,
            help="Select one or more OCR methods."
        )
        # st.session_state.config['leafmachine']['project']['OCR_option'] = OCR_option_select

        # Handling multiple selections (Example logic)
        OCR_options = {
            "Google Vision Handwritten": 'hand',
            "Google Vision Printed": 'normal',
            "LOCAL Qwen-2-VL": "LOCAL Qwen-2-VL",
            "GPT-4o-mini": "GPT-4o-mini",
            
            "Hyperbolic Pixtral-12B-2409": "Pixtral-12B-2409",
            "Hyperbolic Llama-3.2-90B-Vision-Instruct": "Llama-3.2-90B-Vision-Instruct",
            "Hyperbolic Qwen2-VL-7B-Instruct": "Qwen2-VL-7B-Instruct",
            "Hyperbolic Qwen2-VL-72B-Instruct": "Qwen2-VL-72B-Instruct",

            "LOCAL Florence-2": 'LOCAL Florence-2',
            "CRAFT + trOCR": 'CRAFT',
            "LLaVA": 'LLaVA',
        }

        # Map selected options to their corresponding internal representations
        selected_OCR_options = [OCR_options[option] for option in OCR_option_select]
        print('Selected OCR options:',selected_OCR_options)
        # Assuming you need to use these mapped values elsewhere in your application
        st.session_state.config['leafmachine']['project']['OCR_option'] = selected_OCR_options


        



        
    if 'CRAFT' in selected_OCR_options:
        st.subheader('Options for :blue[CRAFT + trOCR]')
        st.write("Supplement Google Vision OCR with :blue[trOCR] (handwriting OCR) using `microsoft/trocr-base-handwritten`. This option requires Google Vision API and a GPU.")
        if 'CRAFT' in selected_OCR_options:
            do_use_trOCR = st.checkbox("Enable :blue[trOCR]", value=True, key="Enable trOCR1",disabled=True)#,disabled=st.session_state['lacks_GPU'])
        else:
            do_use_trOCR = st.checkbox("Enable :blue[trOCR]", value=st.session_state.config['leafmachine']['project']['do_use_trOCR'],key="Enable trOCR2")#,disabled=st.session_state['lacks_GPU'])
            st.session_state.config['leafmachine']['project']['do_use_trOCR'] = do_use_trOCR

        if do_use_trOCR:
            # st.session_state.config['leafmachine']['project']['trOCR_model_path'] = "microsoft/trocr-large-handwritten"
            default_trOCR_model_path = st.session_state.config['leafmachine']['project']['trOCR_model_path']
            user_input_trOCR_model_path = st.text_input(":blue[trOCR] Hugging Face model path. MUST be a fine-tuned version of 'microsoft/trocr-base-handwritten' or 'microsoft/trocr-large-handwritten', or a microsoft :blue[trOCR] model.", value=default_trOCR_model_path)
            if st.session_state.config['leafmachine']['project']['trOCR_model_path'] != user_input_trOCR_model_path:
                is_valid_mp = is_valid_huggingface_model_path(user_input_trOCR_model_path)
                if not is_valid_mp:
                    st.error(f"The Hugging Face model path {user_input_trOCR_model_path} is not valid. Please revise.")
                else:
                    st.session_state.config['leafmachine']['project']['trOCR_model_path'] = user_input_trOCR_model_path


    if "Qwen-2-VL" in selected_OCR_options:
        st.subheader('Options for :blue[Qwen-2-VL]')
        default_qwen_model_path = st.session_state.config['leafmachine']['project']['qwen_model_path']

        st.session_state.config['leafmachine']['project']['qwen_model_path'] = st.radio(
            "Select :blue[Qwen-2-VL] version.",
            ["Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B-Instruct-AWQ", ],
            captions=["Qwen2-VL-7B-Instruct requires at least 24GB of VRAM", "Qwen2-VL-7B-Instruct-AWQ requires 16GB of VRAM."])
    
    if "LOCAL Florence-2" in selected_OCR_options:
        st.subheader('Options for :green[Florence-2]')
        default_florence_model_path = st.session_state.config['leafmachine']['project']['florence_model_path']

        st.session_state.config['leafmachine']['project']['florence_model_path'] = st.radio(
            "Select :green[Florence-2] version.",
            ["microsoft/Florence-2-large", "microsoft/Florence-2-base", ],
            captions=["'large' requires at least 16GB of VRAM", "'base' requires 12GB of VRAM."])

    if "GPT-4o-mini" in selected_OCR_options:
        st.subheader('Options for :violet[GPT-4o-mini]')
        default_resolution = st.session_state.config['leafmachine']['project']['OCR_GPT_4o_mini_resolution']

        st.session_state.config['leafmachine']['project']['OCR_GPT_4o_mini_resolution'] = st.radio(
            "Select level of detail for :violet[GPT-4o-mini] OCR. We only recommend 'high' detail in most scenarios.",
            ["high", "low", ],
            captions=[f"$0.50 per 1,000", f"$5 - $10 per 1,000"])


    if 'LLaVA' in selected_OCR_options:
        st.subheader('Options for :red[LLaVA]')
        OCR_option_llava = st.radio(
            "Select the :red[LLaVA] version",
            options_llava,
            index=default_index_llava,
            help="",captions=captions_llava,
        )
        st.session_state.config['leafmachine']['project']['OCR_option_llava'] = OCR_option_llava

        OCR_option_llava_bit = st.radio(
            "Select the :red[LLaVA] quantization level",
            options_llava_bit,
            index=default_index_llava_bit,
            help="",captions=captions_llava_bit,
        )
        st.session_state.config['leafmachine']['project']['OCR_option_llava_bit'] = OCR_option_llava_bit
    st.write('---')
    
    

    # st.markdown("Below is an example of what the LLM would see given the choice of OCR ensemble. One, two, or three version of OCR can be fed into the LLM prompt. Typically, 'printed + handwritten' works well. If you have a GPU then you can enable trOCR.")
    # if (OCR_option == 'hand') and not do_use_trOCR:
    #     st.text_area(label='Handwritten/Printed',placeholder=demo_text_h,disabled=True, label_visibility='visible', height=150)
    # elif (OCR_option == 'normal') and not do_use_trOCR:
    #     st.text_area(label='Printed',placeholder=demo_text_p,disabled=True, label_visibility='visible', height=150)
    # elif (OCR_option == 'both') and not do_use_trOCR:
    #     st.text_area(label='Handwritten/Printed + Printed',placeholder=demo_text_b,disabled=True, label_visibility='visible', height=150)
    # elif (OCR_option == 'both') and do_use_trOCR:
    #     st.text_area(label='Handwritten/Printed + Printed + trOCR',placeholder=demo_text_trb,disabled=True, label_visibility='visible', height=150)
    # elif (OCR_option == 'normal') and do_use_trOCR:
    #     st.text_area(label='Printed + trOCR',placeholder=demo_text_trp,disabled=True, label_visibility='visible', height=150)
    # elif (OCR_option == 'hand') and do_use_trOCR:
    #     st.text_area(label='Handwritten/Printed + trOCR',placeholder=demo_text_trh,disabled=True, label_visibility='visible', height=150)

def is_valid_huggingface_model_path(model_path):
    from transformers import AutoConfig

    try:
        # Attempt to load the model configuration from Hugging Face Model Hub
        config = AutoConfig.from_pretrained(model_path)
        return True  # If the configuration loads successfully, the model path is valid
    except Exception as e:
        # If loading the model configuration fails, the model path is not valid
        return False
    
@st.cache_data
def show_collage():
    # Load the image only if it's not already in the session state
    if "demo_collage" not in st.session_state:
        # ba = os.path.join(st.session_state.dir_home, 'demo', 'ba', 'ba2.png')
        ba = os.path.join(st.session_state.dir_home, 'demo', 'ba', 'ba2.png')
        st.session_state["demo_collage"] = Image.open(ba)
    with st.expander(":frame_with_picture: View an example of the LeafMachine2 collage image"):
        st.image(st.session_state["demo_collage"], caption='LeafMachine2 Collage', output_format="PNG")

@st.cache_data
def show_ocr():
    if "demo_overlay" not in st.session_state:
        # ocr = os.path.join(st.session_state.dir_home,'demo', 'ba','ocr.png')
        ocr = os.path.join(st.session_state.dir_home,'demo', 'ba','ocr2.png')
        st.session_state["demo_overlay"] = Image.open(ocr)
    
    with st.expander(":frame_with_picture: View an example of the OCR overlay image"):
        st.image(st.session_state["demo_overlay"], caption='OCR Overlay Images', output_format = "PNG")
        # st.image(st.session_state["demo_overlay"], caption='OCR Overlay Images', output_format = "JPEG")

def content_collage_overlay():
    col_collage, col_overlay = st.columns([4,4])   
    
    

    with col_collage:
        st.header('LeafMachine2 Label Collage')    
        st.info("NOTE: We strongly recommend enabling LeafMachine2 cropping if your images are full sized herbarium sheet. Often, the OCR algorithm struggles with full sheets, but works well with the collage images. We have disabled the collage by default for this Hugging Face Space because the Space lacks a GPU and the collage creation takes a bit longer.")
        default_crops = st.session_state.config['leafmachine']['cropped_components']['save_cropped_annotations']
        st.markdown("Prior to transcription, use LeafMachine2 to crop all labels from input images to create label collages for each specimen image. Showing just the text labels to the OCR algorithms significantly improves performance. This runs slowly on the free Hugging Face Space, but runs quickly with a fast CPU or any GPU.")
        st.markdown("Images that are mostly text (like a scanned notecard, or already cropped images) do not require LM2 collage.")

        # if st.session_state.is_hf:
        #     st.session_state.config['leafmachine']['use_RGB_label_images'] = st.checkbox(":rainbow[Use LeafMachine2 label collage for transcriptions]", st.session_state.config['leafmachine'].get('use_RGB_label_images', False), key='do make collage hf')
        # else:
        #     st.session_state.config['leafmachine']['use_RGB_label_images'] = st.checkbox(":rainbow[Use LeafMachine2 label collage for transcriptions]", st.session_state.config['leafmachine'].get('use_RGB_label_images', True), key='do make collage local')
        # Set the options for the radio button
        # Set the options for the radio button with corresponding indices
        # Set the options for the transcription method radio button
        options = {
            0: "Use original images for transcriptions",
            1: "Use LeafMachine2 label collage for transcriptions",
            2: "Use specimen collage for transcriptions"
        }

        # Determine the default index based on the current configuration
        default_index = st.session_state.config['leafmachine'].get('use_RGB_label_images', 1)

        # Create the radio button for transcription method selection
        selected_option = st.radio(
            "Select the transcription method:",
            options=list(options.values()),
            index=default_index
        )

        # Update the session state based on the selected option
        selected_index = list(options.values()).index(selected_option)
        st.session_state.config['leafmachine']['use_RGB_label_images'] = selected_index

        # If "Use specimen collage for transcriptions" is selected, show another radio button for rotation options
        if selected_index == 2:
            rotation_options = {
                True: "Rotate clockwise",
                False: "Rotate counterclockwise"
            }
            
            # Determine the default rotation direction
            default_rotation = st.session_state.config['leafmachine']['project'].get('specimen_rotate', True)
            
            # Create the radio button for rotation direction selection
            selected_rotation = st.radio(
                "Select the rotation direction:",
                options=list(rotation_options.values()),
                index=0 if default_rotation else 1
            )
            
            # Update the configuration based on the selected rotation direction
            st.session_state.config['leafmachine']['project']['specimen_rotate'] = selected_rotation == "Rotate clockwise"

        option_selected_crops = st.multiselect(label="Components to crop",  
                options=['ruler', 'barcode','label', 'colorcard','map','envelope','photo','attached_item','weights',
                'leaf_whole', 'leaf_partial', 'leaflet', 'seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many', 'bud','specimen','roots','wood'],default=default_crops)
        st.session_state.config['leafmachine']['cropped_components']['save_cropped_annotations'] = option_selected_crops
        show_collage()

    with col_overlay:
        st.header('OCR Overlay Image')    

        st.markdown('This will plot bounding boxes around all text that Google Vision was able to detect. If there are no boxes around text, then the OCR failed, so that missing text will not be seen by the LLM when it is creating the JSON object. The created image will be viewable in the VoucherVisionEditor.')
        
        do_create_OCR_helper_image = st.checkbox("Create image showing an overlay of the OCR detections",value=st.session_state.config['leafmachine']['do_create_OCR_helper_image'],disabled=True)
        st.session_state.config['leafmachine']['do_create_OCR_helper_image'] = do_create_OCR_helper_image
        show_ocr()
        



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
        st.session_state.config['leafmachine']['project']['pdf_conversion_dpi'] = st.number_input("PDF conversion DPI", value=st.session_state.config['leafmachine']['project'].get('pdf_conversion_dpi', 100), help='DPI of the JPG created from the page of a PDF. 100 should be fine for most cases, but 200 or 300 might be better for large images.')
    
    with col_processing_2:
        st.subheader('Filename Prefix Handling')
        st.session_state.config['leafmachine']['project']['prefix_removal'] = st.text_input("Remove prefix from catalog number", st.session_state.config['leafmachine']['project'].get('prefix_removal', ''),placeholder="e.g. MICH-V-")
        st.session_state.config['leafmachine']['project']['suffix_removal'] = st.text_input("Remove suffix from catalog number", st.session_state.config['leafmachine']['project'].get('suffix_removal', ''),placeholder="e.g. _B")
        st.session_state.config['leafmachine']['project']['catalog_numerical_only'] = st.checkbox("Require 'Catalog Number' to be numerical only", st.session_state.config['leafmachine']['project'].get('catalog_numerical_only', True))
    
    ### Logging and Image Validation - col_v1
    st.write("---")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.header('Logging and Image Validation')    
        option_check_illegal = st.checkbox("Check for illegal filenames", value=st.session_state.config['leafmachine']['do']['check_for_illegal_filenames'])
        st.session_state.config['leafmachine']['do']['check_for_illegal_filenames'] = option_check_illegal

        option_skip_vertical = st.checkbox("Skip vertical image requirement (e.g. horizontal PDFs)", value=st.session_state.config['leafmachine']['do']['skip_vertical'],help='LeafMachine2 label collage requires images to have vertical aspect ratios for stability. If your input images have a horizonatal aspect ratio, try skipping the vertical requirement first, look for strange behavior, and then reassess. If your image/PDFs are already closeups and you do not need the collage, then skipping the vertical requirement is the right choice.')
        st.session_state.config['leafmachine']['do']['skip_vertical'] = option_skip_vertical

        st.session_state.config['leafmachine']['do']['check_for_corrupt_images_make_vertical'] = st.checkbox("Check for corrupt images", st.session_state.config['leafmachine']['do'].get('check_for_corrupt_images_make_vertical', True),disabled=True)
        
        st.session_state.config['leafmachine']['print']['verbose'] = st.checkbox("Print verbose", st.session_state.config['leafmachine']['print'].get('verbose', True))
        st.session_state.config['leafmachine']['print']['optional_warnings'] = st.checkbox("Show optional warnings", st.session_state.config['leafmachine']['print'].get('optional_warnings', True))
        
        log_level = st.session_state.config['leafmachine']['logging'].get('log_level', None)
        log_level_display = log_level if log_level is not None else 'default'
        selected_log_level = st.selectbox("Logging Level", ['default', 'DEBUG', 'INFO', 'WARNING', 'ERROR'], index=['default', 'DEBUG', 'INFO', 'WARNING', 'ERROR'].index(log_level_display))
        
        if selected_log_level == 'default':
            st.session_state.config['leafmachine']['logging']['log_level'] = None
        else:
            st.session_state.config['leafmachine']['logging']['log_level'] = selected_log_level

    with col_v2:
        

        # print(f"Number of GPUs: {st.session_state.num_gpus}")
        # print(f"GPU Details: {st.session_state.gpu_dict}")
        # print(f"Total VRAM: {st.session_state.total_vram_gb} GB")
        # print(f"Capability Score: {st.session_state.capability_score}")

        st.header('System GPU Information')
        st.markdown(f"**Torch CUDA:** {torch.cuda.is_available()}")
        st.markdown(f"**Number of GPUs:** {st.session_state.num_gpus}")

        if st.session_state.num_gpus > 0:
            st.markdown("**GPU Details:**")
            for gpu_id, vram in st.session_state.gpu_dict.items():
                st.text(f"{gpu_id}: {vram}")
            
            st.markdown(f"**Total VRAM:** {st.session_state.total_vram_gb} GB")
            st.markdown(f"**Capability Score:** {st.session_state.capability_score}")
        else:
            st.warning("No GPUs detected in the system.")



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
@st.cache_data
def render_expense_report_summary():
    expense_summary = st.session_state.expense_summary
    expense_report = st.session_state.expense_report
    st.header('Expense Report Summary')

    if not expense_summary:
        st.warning('No expense report data available.')
    else:
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
    
    col_input, col_gallery = st.columns([4,8])
    content_project_settings(col_input)
    content_input_images(col_input, col_gallery)


    col3, col4 = st.columns([1,1])
    with col3:
        content_prompt_and_llm_version()
    with col4:
        content_api_check()

    content_ocr_method()

    content_collage_overlay()
    content_tools()
    content_llm_cost()
    content_processing_options()
    content_less_used()
    with st.expander("View additional settings"):
        content_archival_components()
        content_space_saver()


#################################################################################################################################################
# Main ##########################################################################################################################################
#################################################################################################################################################
do_print_profiler = False
if st.session_state['is_hf']:
    # if st.session_state.proceed_to_build_llm_prompt:
    #     build_LLM_prompt_config()
    if st.session_state.proceed_to_main:
        if do_print_profiler:
            profiler = cProfile.Profile()
            profiler.enable()

        main()
        
        if do_print_profiler:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            stats.print_stats(30)
    
else:
    if not st.session_state.private_file:
        create_private_file()
    # elif st.session_state.proceed_to_build_llm_prompt:
    #     build_LLM_prompt_config()
    elif st.session_state.proceed_to_private and not st.session_state['is_hf']:
        create_private_file()
    elif st.session_state.proceed_to_main:
        if do_print_profiler:
            profiler = cProfile.Profile()
            profiler.enable()

        main()

        if do_print_profiler:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            stats.print_stats(30)






