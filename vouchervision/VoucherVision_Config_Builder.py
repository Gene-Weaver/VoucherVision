import os #, yaml, platform, traceback
from vouchervision.LeafMachine2_Config_Builder import get_default_download_folder #, write_config_file
# from vouchervision.general_utils import validate_dir, print_main_fail
# from vouchervision.vouchervision_main import voucher_vision
from vouchervision.general_utils import get_cfg_from_full_path

def build_VV_config(loaded_cfg=None):
    if loaded_cfg is None:
        #############################################
        ############ Set common defaults ############
        #############################################
        # Changing the values below will set new 
        # default values each time you open the 
        # VoucherVision user interface
        #############################################
        #############################################
        #############################################

        dir_home = os.path.dirname(os.path.dirname(__file__))
        run_name = 'test'
        # dir_images_local = 'D:/Dropbox/LM2_Env/Image_Datasets/GBIF_BroadSample_3SppPerFamily1'
        dir_images_local = os.path.join(dir_home,'demo','demo_images')
        
        # The default output location is the computer's "Downloads" folder
        #    You can set dir_output directly by typing the folder path,
        # OR you can uncomment the line "dir_output = default_output_folder" 
        #    to have VoucherVision save to the Downloads folder by default
        default_output_folder = get_default_download_folder()
        dir_output = default_output_folder
        # dir_output = 'D:/D_Desktop/LM2'

        prefix_removal = '' #'MICH-V-'
        suffix_removal = ''
        catalog_numerical_only = False

        save_cropped_annotations = ['label','barcode']

        do_use_trOCR = False
        do_use_florence = False
        trOCR_model_path = "microsoft/trocr-large-handwritten"
        florence_model_path = "microsoft/Florence-2-large"
        OCR_option = 'hand'
        OCR_option_llava = 'llava-v1.6-mistral-7b' # "llava-v1.6-mistral-7b", "llava-v1.6-34b", "llava-v1.6-vicuna-13b", "llava-v1.6-vicuna-7b",
        OCR_option_llava_bit = 'full' # full or 4bit
        double_OCR = False

        tool_GEO = True
        tool_WFO = True
        tool_wikipedia = True

        check_for_illegal_filenames = False

        LLM_version_user = 'Gemini 1.5 Flash' # 'Azure GPT 4' #'Azure GPT 4 Turbo 1106-preview'
        prompt_version = 'SLTPvM_long.yaml' # from ["Version 1", "Version 1 No Domain Knowledge", "Version 2"]
        use_LeafMachine2_collage_images = True # Use LeafMachine2 collage images
        do_create_OCR_helper_image = True

        batch_size = 500
        num_workers = 8

        skip_vertical = False
        pdf_conversion_dpi = 100

        path_domain_knowledge = '' #os.path.join(dir_home,'domain_knowledge','SLTP_UM_AllAsiaMinimalInRegion.xlsx')
        embeddings_database_name = '' #os.path.splitext(os.path.basename(path_domain_knowledge))[0]

        #############################################
        #############################################
        ########## DO NOT EDIT BELOW HERE ###########
        #############################################
        #############################################
        return assemble_config(dir_home, run_name, dir_images_local,dir_output,
                        prefix_removal,suffix_removal,catalog_numerical_only,LLM_version_user,batch_size,num_workers,
                        path_domain_knowledge,embeddings_database_name,use_LeafMachine2_collage_images,
                        prompt_version, do_create_OCR_helper_image, do_use_trOCR, do_use_florence, trOCR_model_path, florence_model_path, OCR_option, OCR_option_llava, 
                        OCR_option_llava_bit, double_OCR, save_cropped_annotations, 
                        tool_GEO, tool_WFO, tool_wikipedia,
                        check_for_illegal_filenames, skip_vertical, pdf_conversion_dpi, use_domain_knowledge=False)
    else:
        dir_home = os.path.dirname(os.path.dirname(__file__))
        run_name = loaded_cfg['leafmachine']['project']['run_name']
        dir_images_local = loaded_cfg['leafmachine']['project']['dir_images_local']
        
        default_output_folder = loaded_cfg['leafmachine']['project']['dir_output']
        dir_output = loaded_cfg['leafmachine']['project']['dir_output']

        prefix_removal = loaded_cfg['leafmachine']['project']['prefix_removal']
        suffix_removal = loaded_cfg['leafmachine']['project']['suffix_removal']
        catalog_numerical_only = loaded_cfg['leafmachine']['project']['catalog_numerical_only']

        do_use_trOCR = loaded_cfg['leafmachine']['project']['do_use_trOCR']
        do_use_florence = loaded_cfg['leafmachine']['project']['do_use_florence']
        trOCR_model_path = loaded_cfg['leafmachine']['project']['trOCR_model_path']
        florence_model_path = loaded_cfg['leafmachine']['project']['florence_model_path']
        OCR_option = loaded_cfg['leafmachine']['project']['OCR_option']
        OCR_option_llava = loaded_cfg['leafmachine']['project']['OCR_option_llava']
        OCR_option_llava_bit  = loaded_cfg['leafmachine']['project']['OCR_option_llava_bit']
        double_OCR = loaded_cfg['leafmachine']['project']['double_OCR']

        tool_GEO = loaded_cfg['leafmachine']['project']['tool_GEO']
        tool_WFO = loaded_cfg['leafmachine']['project']['tool_WFO']
        tool_wikipedia = loaded_cfg['leafmachine']['project']['tool_wikipedia']

        pdf_conversion_dpi = loaded_cfg['leafmachine']['project']['pdf_conversion_dpi']

        LLM_version_user = loaded_cfg['leafmachine']['LLM_version']
        prompt_version = loaded_cfg['leafmachine']['project']['prompt_version']
        use_LeafMachine2_collage_images = loaded_cfg['leafmachine']['use_RGB_label_images']
        do_create_OCR_helper_image = loaded_cfg['leafmachine']['do_create_OCR_helper_image']

        batch_size = loaded_cfg['leafmachine']['project']['batch_size']
        num_workers = loaded_cfg['leafmachine']['project']['num_workers']

        path_domain_knowledge = loaded_cfg['leafmachine']['project']['path_to_domain_knowledge_xlsx']
        embeddings_database_name = os.path.splitext(os.path.basename(path_domain_knowledge))[0]

        save_cropped_annotations = loaded_cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        check_for_illegal_filenames = loaded_cfg['leafmachine']['do']['check_for_illegal_filenames']
        skip_vertical = loaded_cfg['leafmachine']['do']['skip_vertical']

        return assemble_config(dir_home, run_name, dir_images_local,dir_output,
                        prefix_removal,suffix_removal,catalog_numerical_only,LLM_version_user,batch_size,num_workers,
                        path_domain_knowledge,embeddings_database_name,use_LeafMachine2_collage_images,
                        prompt_version, do_create_OCR_helper_image, do_use_trOCR, do_use_florence, trOCR_model_path, florence_model_path, OCR_option, OCR_option_llava, 
                        OCR_option_llava_bit, double_OCR, save_cropped_annotations,
                        tool_GEO, tool_WFO, tool_wikipedia,
                        check_for_illegal_filenames, skip_vertical, pdf_conversion_dpi, use_domain_knowledge=False)


def assemble_config(dir_home, run_name, dir_images_local,dir_output,
                    prefix_removal,suffix_removal,catalog_numerical_only,LLM_version_user,batch_size,num_workers,
                    path_domain_knowledge,embeddings_database_name,use_LeafMachine2_collage_images,
                    prompt_version, do_create_OCR_helper_image_user, do_use_trOCR, do_use_florence, trOCR_model_path, florence_model_path, OCR_option, OCR_option_llava, 
                    OCR_option_llava_bit, double_OCR, save_cropped_annotations, 
                    tool_GEO, tool_WFO, tool_wikipedia,
                    check_for_illegal_filenames, skip_vertical, pdf_conversion_dpi, use_domain_knowledge=False):
    

    # Initialize the base structure
    config_data = {
        'leafmachine': {}
    }

    # Modular sections to be added to 'leafmachine'
    do_section = {
        'check_for_illegal_filenames': check_for_illegal_filenames,
        'check_for_corrupt_images_make_vertical': True,
        'skip_vertical': skip_vertical,
    }

    print_section = {
        'verbose': True,
        'optional_warnings': True
    }

    logging_section = {
        'log_level': None
    }


    project_section = {
        'dir_output': dir_output, 
        'run_name': run_name,
        'image_location': 'local',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'dir_images_local': dir_images_local,
        'continue_run_from_partial_xlsx': '',
        'prefix_removal': prefix_removal,
        'suffix_removal': suffix_removal,
        'catalog_numerical_only': catalog_numerical_only,
        'use_domain_knowledge': use_domain_knowledge,
        'embeddings_database_name': embeddings_database_name,
        'build_new_embeddings_database': False,
        'path_to_domain_knowledge_xlsx': path_domain_knowledge,
        'prompt_version': prompt_version,
        'delete_all_temps': False,
        'delete_temps_keep_VVE': False,
        'do_use_trOCR': do_use_trOCR,
        'do_use_florence': do_use_florence,
        'trOCR_model_path': trOCR_model_path,
        'florence_model_path': florence_model_path,
        'OCR_option': OCR_option,
        'OCR_option_llava': OCR_option_llava,
        'OCR_option_llava_bit': OCR_option_llava_bit,
        'double_OCR': double_OCR,
        'pdf_conversion_dpi': pdf_conversion_dpi,
        'tool_GEO': tool_GEO, 
        'tool_WFO': tool_WFO, 
        'tool_wikipedia': tool_wikipedia,
    }

    modules_section = {
        'specimen_crop': True
    }

    LLM_version = LLM_version_user 
    use_RGB_label_images = use_LeafMachine2_collage_images # Use LeafMachine2 collage images
    do_create_OCR_helper_image = do_create_OCR_helper_image_user
    
    cropped_components_section = {
        'do_save_cropped_annotations': True,
        'save_cropped_annotations': save_cropped_annotations,
        'save_per_image': False,
        'save_per_annotation_class': True,
        'binarize_labels': False,
        'binarize_labels_skeletonize': False
    }

    data_section = {
        'save_json_rulers': False,
        'save_json_measurements': False,
        'save_individual_csv_files_rulers': False,
        'save_individual_csv_files_measurements': False,
        'save_individual_csv_files_landmarks': False,
        'save_individual_efd_files': False,
        'include_darwin_core_data_from_combined_file': False,
        'do_apply_conversion_factor': False
    }

    overlay_section = {
        'save_overlay_to_pdf': False,
        'save_overlay_to_jpgs': True,
        'overlay_dpi': 300, # Between 100 to 300
        'overlay_background_color': 'black', # Either 'white' or 'black'

        'show_archival_detections': True,
        'show_plant_detections': True,
        'show_segmentations': True,
        'show_landmarks': True,
        'ignore_archival_detections_classes': [],
        'ignore_plant_detections_classes': ['leaf_whole', 'specimen'], # Could also include 'leaf_partial' and others if needed
        'ignore_landmark_classes': [],

        'line_width_archival': 12, # Previous value given was 2
        'line_width_plant': 12, # Previous value given was 6
        'line_width_seg': 12, # 12 is specified as "thick"
        'line_width_efd': 12, # 3 is specified as "thick" but 12 is given here
        'alpha_transparency_archival': 0.3,
        'alpha_transparency_plant': 0,
        'alpha_transparency_seg_whole_leaf': 0.4,
        'alpha_transparency_seg_partial_leaf': 0.3
    }

    archival_component_detector_section = {
        'detector_type': 'Archival_Detector',
        'detector_version': 'PREP_final',
        'detector_iteration': 'PREP_final',
        'detector_weights': 'best.pt',
        'minimum_confidence_threshold': 0.5, # Default is 0.5
        'do_save_prediction_overlay_images': True,
        'ignore_objects_for_overlay': []
    }

    # Add the sections to the 'leafmachine' key
    config_data['leafmachine']['do'] = do_section
    config_data['leafmachine']['print'] = print_section
    config_data['leafmachine']['logging'] = logging_section
    config_data['leafmachine']['project'] = project_section
    config_data['leafmachine']['LLM_version'] = LLM_version
    config_data['leafmachine']['use_RGB_label_images'] = use_RGB_label_images
    config_data['leafmachine']['do_create_OCR_helper_image'] = do_create_OCR_helper_image
    config_data['leafmachine']['cropped_components'] = cropped_components_section
    config_data['leafmachine']['modules'] = modules_section
    config_data['leafmachine']['data'] = data_section
    config_data['leafmachine']['overlay'] = overlay_section
    config_data['leafmachine']['archival_component_detector'] = archival_component_detector_section

    return config_data, dir_home

# def build_api_tests(api):
#     dir_home = os.path.dirname(os.path.dirname(__file__))
#     path_to_configs = os.path.join(dir_home,'demo','demo_configs')

#     dir_home = os.path.dirname(os.path.dirname(__file__))
#     dir_images_local = os.path.join(dir_home,'demo','demo_images')
#     validate_dir(os.path.join(dir_home,'demo','demo_configs'))
#     path_domain_knowledge = os.path.join(dir_home,'domain_knowledge','SLTP_UM_AllAsiaMinimalInRegion.xlsx')
#     embeddings_database_name = os.path.splitext(os.path.basename(path_domain_knowledge))[0]
#     prefix_removal = ''
#     suffix_removal = ''
#     catalog_numerical_only = False
#     batch_size = 500
#     do_create_OCR_helper_image = False


#     # ### Option 1: "GPT 4" of ["GPT 4", "GPT 3.5", "Azure GPT 4", "Azure GPT 3.5", "PaLM 2"]
#     # LLM_version_user = 'Azure GPT 4'
    
#     # ### Option 2: False of [False, True]
#     # use_LeafMachine2_collage_images = False
    
#     # ### Option 3: False of [False, True]
#     # use_domain_knowledge = True

#     test_results = {}
#     if api == 'openai':
#         OPT1, OPT2, OPT3 = TestOptionsAPI_openai.get_options()
#     elif api == 'palm':
#         OPT1, OPT2, OPT3 = TestOptionsAPI_palm.get_options()
#     elif api == 'azure_openai':
#         OPT1, OPT2, OPT3 = TestOptionsAPI_azure_openai.get_options()
#     else:
#         raise

#     ind = -1
#     ind_opt1 = -1
#     ind_opt2 = -1
#     ind_opt3 = -1

#     for opt1 in OPT1:
#         ind_opt1+= 1
#         for opt2 in OPT2:
#             ind_opt2 += 1
#             for opt3 in OPT3:
#                 ind += 1
#                 ind_opt3 += 1
                
#                 LLM_version_user = opt1
#                 use_LeafMachine2_collage_images = opt2
#                 prompt_version = opt3

#                 filename = f"{ind}__OPT1-{ind_opt1}__OPT2-{ind_opt2}__OPT3-{ind_opt3}.yaml"
#                 run_name = f"{ind}__OPT1-{ind_opt1}__OPT2-{ind_opt2}__OPT3-{ind_opt3}"

#                 dir_output = os.path.join(dir_home,'demo','demo_output','run_name')
#                 validate_dir(dir_output)
                
#                 config_data, dir_home = assemble_config(dir_home, run_name, dir_images_local,dir_output,
#                     prefix_removal,suffix_removal,catalog_numerical_only,LLM_version_user,batch_size,
#                     path_domain_knowledge,embeddings_database_name,use_LeafMachine2_collage_images,
#                     prompt_version,do_create_OCR_helper_image)
                
#                 write_config_file(config_data, os.path.join(dir_home,'demo','demo_configs'),filename=filename)

#                 test_results[run_name] = False
#             ind_opt3 = -1
#         ind_opt2 = -1
#     ind_opt1 = -1
        
#     return dir_home, path_to_configs, test_results

# def build_demo_tests(llm_version):
#     dir_home = os.path.dirname(os.path.dirname(__file__))
#     path_to_configs = os.path.join(dir_home,'demo','demo_configs')

#     dir_home = os.path.dirname(os.path.dirname(__file__))
#     dir_images_local = os.path.join(dir_home,'demo','demo_images')
#     validate_dir(os.path.join(dir_home,'demo','demo_configs'))
#     path_domain_knowledge = os.path.join(dir_home,'domain_knowledge','SLTP_UM_AllAsiaMinimalInRegion.xlsx')
#     embeddings_database_name = os.path.splitext(os.path.basename(path_domain_knowledge))[0]
#     prefix_removal = ''
#     suffix_removal = ''
#     catalog_numerical_only = False
#     batch_size = 500
#     do_create_OCR_helper_image = False

#     # ### Option 1: "GPT 4" of ["GPT 4", "GPT 3.5", "Azure GPT 4", "Azure GPT 3.5", "PaLM 2"]
#     # LLM_version_user = 'Azure GPT 4'
    
#     # ### Option 2: False of [False, True]
#     # use_LeafMachine2_collage_images = False
    
#     # ### Option 3: False of [False, True]
#     # use_domain_knowledge = True

#     test_results = {}
#     if llm_version == 'gpt':
#         OPT1, OPT2, OPT3 = TestOptionsGPT.get_options()
#     elif llm_version == 'palm':
#         OPT1, OPT2, OPT3 = TestOptionsPalm.get_options()
#     else:
#         raise

#     ind = -1
#     ind_opt1 = -1
#     ind_opt2 = -1
#     ind_opt3 = -1

#     for opt1 in OPT1:
#         ind_opt1+= 1
#         for opt2 in OPT2:
#             ind_opt2 += 1
#             for opt3 in OPT3:
#                 ind += 1
#                 ind_opt3 += 1
                
#                 LLM_version_user = opt1
#                 use_LeafMachine2_collage_images = opt2
#                 prompt_version = opt3

#                 filename = f"{ind}__OPT1-{ind_opt1}__OPT2-{ind_opt2}__OPT3-{ind_opt3}.yaml"
#                 run_name = f"{ind}__OPT1-{ind_opt1}__OPT2-{ind_opt2}__OPT3-{ind_opt3}"

#                 dir_output = os.path.join(dir_home,'demo','demo_output','run_name')
#                 validate_dir(dir_output)
                

#                 if llm_version == 'gpt':
#                     if prompt_version in ['Version 1']:
#                         config_data, dir_home = assemble_config(dir_home, run_name, dir_images_local,dir_output,
#                             prefix_removal,suffix_removal,catalog_numerical_only,LLM_version_user,batch_size,
#                             path_domain_knowledge,embeddings_database_name,use_LeafMachine2_collage_images,
#                             prompt_version, do_create_OCR_helper_image, use_domain_knowledge=True)
#                     else:
#                         config_data, dir_home = assemble_config(dir_home, run_name, dir_images_local,dir_output,
#                             prefix_removal,suffix_removal,catalog_numerical_only,LLM_version_user,batch_size,
#                             path_domain_knowledge,embeddings_database_name,use_LeafMachine2_collage_images,
#                             prompt_version, do_create_OCR_helper_image)
#                 elif llm_version == 'palm':
#                     if prompt_version in ['Version 1 PaLM 2']:
#                         config_data, dir_home = assemble_config(dir_home, run_name, dir_images_local,dir_output,
#                             prefix_removal,suffix_removal,catalog_numerical_only,LLM_version_user,batch_size,
#                             path_domain_knowledge,embeddings_database_name,use_LeafMachine2_collage_images,
#                             prompt_version, do_create_OCR_helper_image, use_domain_knowledge=True)
#                     else:
#                         config_data, dir_home = assemble_config(dir_home, run_name, dir_images_local,dir_output,
#                             prefix_removal,suffix_removal,catalog_numerical_only,LLM_version_user,batch_size,
#                             path_domain_knowledge,embeddings_database_name,use_LeafMachine2_collage_images,
#                             prompt_version, do_create_OCR_helper_image)
                
                
#                 write_config_file(config_data, os.path.join(dir_home,'demo','demo_configs'),filename=filename)

#                 test_results[run_name] = False
#             ind_opt3 = -1
#         ind_opt2 = -1
#     ind_opt1 = -1
        
#     return dir_home, path_to_configs, test_results

class TestOptionsGPT:
    OPT1 = ["gpt-4-1106-preview","GPT 4", "GPT 3.5", "Azure GPT 4", "Azure GPT 3.5"]
    OPT2 = [False, True]
    OPT3 = ["Version 1", "Version 1 No Domain Knowledge", "Version 2"]

    @classmethod
    def get_options(cls):
        return cls.OPT1, cls.OPT2, cls.OPT3
    @classmethod
    def get_length(cls):
        return 24
    
class TestOptionsPalm:
    OPT1 = ["PaLM 2"]
    OPT2 = [False, True]
    OPT3 = ["Version 1 PaLM 2", "Version 1 PaLM 2 No Domain Knowledge", "Version 2 PaLM 2"]

    @classmethod
    def get_options(cls):
        return cls.OPT1, cls.OPT2, cls.OPT3
    @classmethod
    def get_length(cls):
        return 6
    
class TestOptionsAPI_openai:
    OPT1 = ["GPT 3.5"]
    OPT2 = [False]
    OPT3 = ["Version 2"]

    @classmethod
    def get_options(cls):
        return cls.OPT1, cls.OPT2, cls.OPT3
    @classmethod
    def get_length(cls):
        return 24
    
class TestOptionsAPI_azure_openai:
    OPT1 = ["Azure GPT 3.5"]
    OPT2 = [False]
    OPT3 = ["Version 2"]

    @classmethod
    def get_options(cls):
        return cls.OPT1, cls.OPT2, cls.OPT3
    @classmethod
    def get_length(cls):
        return 24
    
class TestOptionsAPI_palm:
    OPT1 = ["PaLM 2"]
    OPT2 = [False]
    OPT3 = ["Version 2 PaLM 2"]

    @classmethod
    def get_options(cls):
        return cls.OPT1, cls.OPT2, cls.OPT3
    @classmethod
    def get_length(cls):
        return 6
    
# def run_demo_tests_GPT(progress_report):
#     dir_home, path_to_configs, test_results = build_demo_tests('gpt')
#     progress_report.set_n_overall(len(test_results.items()))

#     JSON_results = {}

#     for ind, (cfg, result) in enumerate(test_results.items()):
#         OPT1, OPT2, OPT3 = TestOptionsGPT.get_options()
        
#         test_ind, ind_opt1, ind_opt2, ind_opt3 = cfg.split('__')
#         opt1_readable = OPT1[int(ind_opt1.split('-')[1])]

#         if opt1_readable in ["Azure GPT 4", "Azure GPT 3.5"]:
#             api_version = 'gpt-azure'
#         elif opt1_readable in ["GPT 4", "GPT 3.5"]:
#             api_version = 'gpt'
#         else:
#             raise

#         opt2_readable = "Use LeafMachine2 for Collage Images" if OPT2[int(ind_opt2.split('-')[1])] else "Don't use LeafMachine2 for Collage Images"
#         opt3_readable = f"Prompt {OPT3[int(ind_opt3.split('-')[1])]}"
#         # Construct the human-readable test name
#         human_readable_name = f"{opt1_readable}, {opt2_readable}, {opt3_readable}"
#         get_n_overall = progress_report.get_n_overall()
#         progress_report.update_overall(f"Test {int(test_ind)+1} of {get_n_overall} --- Validating {human_readable_name}")
#         print_main_fail(f"Starting validation test: {human_readable_name}")
#         cfg_file_path = os.path.join(path_to_configs,'.'.join([cfg,'yaml']))
        
#         if check_API_key(dir_home, api_version) and check_API_key(dir_home, 'google-vision-ocr'):
#             try:
#                 last_JSON_response, total_cost = voucher_vision(cfg_file_path, dir_home, cfg_test=None, progress_report=progress_report, test_ind=int(test_ind))
#                 test_results[cfg] = True
#                 JSON_results[ind] = last_JSON_response
#             except Exception as e:
#                 JSON_results[ind] = None
#                 test_results[cfg] = False
#                 print(f"An exception occurred: {e}")
#                 traceback.print_exc()  # This will print the full traceback
#         else:
#             fail_response = ''
#             if not check_API_key(dir_home, 'google-vision-ocr'):
#                 fail_response += "No API key found for Google Vision OCR"
#             if not check_API_key(dir_home, api_version):
#                 fail_response += f"  +  No API key found for {api_version}"
#             test_results[cfg] = False
#             JSON_results[ind] = fail_response
#             print(f"No API key found for {fail_response}")
            
#     return test_results, JSON_results

# def run_demo_tests_Palm(progress_report):
#     api_version = 'palm'

#     dir_home, path_to_configs, test_results = build_demo_tests('palm')
#     progress_report.set_n_overall(len(test_results.items()))

#     JSON_results = {}

#     for ind, (cfg, result) in enumerate(test_results.items()):
#         OPT1, OPT2, OPT3 = TestOptionsPalm.get_options()
#         test_ind, ind_opt1, ind_opt2, ind_opt3 = cfg.split('__')
#         opt1_readable = OPT1[int(ind_opt1.split('-')[1])]
#         opt2_readable = "Use LeafMachine2 for Collage Images" if OPT2[int(ind_opt2.split('-')[1])] else "Don't use LeafMachine2 for Collage Images"
#         opt3_readable = f"Prompt {OPT3[int(ind_opt3.split('-')[1])]}"
#         # opt3_readable = "Use Domain Knowledge" if OPT3[int(ind_opt3.split('-')[1])] else "Don't use Domain Knowledge"
#         # Construct the human-readable test name
#         human_readable_name = f"{opt1_readable}, {opt2_readable}, {opt3_readable}"
#         get_n_overall = progress_report.get_n_overall()
#         progress_report.update_overall(f"Test {int(test_ind)+1} of {get_n_overall} --- Validating {human_readable_name}")
#         print_main_fail(f"Starting validation test: {human_readable_name}")
#         cfg_file_path = os.path.join(path_to_configs,'.'.join([cfg,'yaml']))
        
#         if check_API_key(dir_home, api_version) and check_API_key(dir_home, 'google-vision-ocr') :
#             try:
#                 last_JSON_response, total_cost = voucher_vision(cfg_file_path, dir_home, cfg_test=None, path_custom_prompts=None, progress_report=progress_report, test_ind=int(test_ind))
#                 test_results[cfg] = True
#                 JSON_results[ind] = last_JSON_response
#             except Exception as e:
#                 test_results[cfg] = False
#                 JSON_results[ind] = None
#                 print(f"An exception occurred: {e}")
#                 traceback.print_exc()  # This will print the full traceback
#         else:
#             fail_response = ''
#             if not check_API_key(dir_home, 'google-vision-ocr'):
#                 fail_response += "No API key found for Google Vision OCR"
#             if not check_API_key(dir_home, api_version):
#                 fail_response += f"  +  No API key found for {api_version}"
#             test_results[cfg] = False
#             JSON_results[ind] = fail_response
#             print(f"No API key found for {fail_response}")

#     return test_results, JSON_results

# def run_api_tests(api):
#     try:
#         dir_home, path_to_configs, test_results = build_api_tests(api)

#         JSON_results = {}

#         for ind, (cfg, result) in enumerate(test_results.items()):
#             if api == 'openai':
#                 OPT1, OPT2, OPT3 = TestOptionsAPI_openai.get_options()
#             elif 'azure_openai':
#                 OPT1, OPT2, OPT3 = TestOptionsAPI_azure_openai.get_options()
#             elif 'palm':
#                 OPT1, OPT2, OPT3 = TestOptionsAPI_palm.get_options()
#             test_ind, ind_opt1, ind_opt2, ind_opt3 = cfg.split('__')
#             opt1_readable = OPT1[int(ind_opt1.split('-')[1])]
#             opt2_readable = "Use LeafMachine2 for Collage Images" if OPT2[int(ind_opt2.split('-')[1])] else "Don't use LeafMachine2 for Collage Images"
#             opt3_readable = f"Prompt {OPT3[int(ind_opt3.split('-')[1])]}"
#             # opt3_readable = "Use Domain Knowledge" if OPT3[int(ind_opt3.split('-')[1])] else "Don't use Domain Knowledge"
#             # Construct the human-readable test name
#             human_readable_name = f"{opt1_readable}, {opt2_readable}, {opt3_readable}"
#             print_main_fail(f"Starting validation test: {human_readable_name}")
#             cfg_file_path = os.path.join(path_to_configs,'.'.join([cfg,'yaml']))
            
#             if check_API_key(dir_home, api) and check_API_key(dir_home, 'google-vision-ocr') :
#                 try:
#                     last_JSON_response, total_cost = voucher_vision(cfg_file_path, dir_home, None,path_custom_prompts=None , cfg_test=None, progress_report=None, test_ind=int(test_ind))
#                     test_results[cfg] = True
#                     JSON_results[ind] = last_JSON_response
#                     return True

#                 except Exception as e:
#                     print(e)
#                     return False
#             else:
#                 return False
#     except Exception as e:
#         print(e)
#         return False

def has_API_key(val):
        if val != '':
            return True
        else:
            return False
        
def check_if_usable(is_hf): ############################################################################################################## TODO fix
    if is_hf:
            return True ########### needs actual logic. borrow from another function to not repeat this
    else:
        dir_home = os.path.dirname(os.path.dirname(__file__))
        path_cfg_private = os.path.join(dir_home, 'PRIVATE_DATA.yaml')
        cfg_private = get_cfg_from_full_path(path_cfg_private)

        has_key_openai = has_API_key(cfg_private['openai']['OPENAI_API_KEY'])

        has_key_azure_openai = has_API_key(cfg_private['openai_azure']['OPENAI_API_VERSION']) 
        
        has_key_google_OCR = has_API_key(cfg_private['google']['GOOGLE_APPLICATION_CREDENTIALS'])

        has_key_MISTRAL = has_API_key(cfg_private['mistral']['MISTRAL_API_KEY'])


        if has_key_google_OCR and (has_key_azure_openai or has_key_openai or has_key_MISTRAL):
            return True
        else:
            return False

# def check_API_key(dir_home, api_version):
#     dir_home = os.path.dirname(os.path.dirname(__file__))
#     path_cfg_private = os.path.join(dir_home, 'PRIVATE_DATA.yaml')
#     cfg_private = get_cfg_from_full_path(path_cfg_private)

#     has_key_openai = has_API_key(cfg_private['openai']['OPENAI_API_KEY'])

#     has_key_azure_openai = has_API_key(cfg_private['openai_azure']['api_version']) 

#     # has_key_palm2 = has_API_key(cfg_private['google_palm']['google_palm_api'])

#     has_key_google_OCR = has_API_key(cfg_private['google']['GOOGLE_APPLICATION_CREDENTIALS'])

#     if api_version in ['gpt','openai'] and has_key_openai:
#         return True
#     elif api_version in ['gpt-azure', 'azure_openai'] and has_key_azure_openai:
#         return True
#     elif api_version == 'google-vision-ocr' and has_key_google_OCR:
#         return True
#     else:
#         return False
