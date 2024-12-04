'''
VoucherVision - based on LeafMachine2 Processes
'''
import os, inspect, sys, shutil, yaml
from time import perf_counter
# currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)
# sys.path.append(currentdir)
from vouchervision.component_detector.component_detector import detect_plant_components, detect_archival_components
from vouchervision.general_utils import create_specimen_collage, save_token_info_as_csv, print_main_start, check_for_subdirs_VV, load_config_file, load_config_file_testing, report_config, save_config_file, crop_detections_from_images_VV
from vouchervision.directory_structure_VV import Dir_Structure
from vouchervision.data_project import Project_Info
from vouchervision.LM2_logger import start_logging
from vouchervision.fetch_data import fetch_data
from vouchervision.utils_VoucherVision import VoucherVision, space_saver
# from vouchervision.utils_VoucherVision_parallel import VoucherVision, space_saver
from vouchervision.utils_hf import upload_to_drive

def voucher_vision(cfg_file_path, dir_home, path_custom_prompts, cfg_test, progress_report, json_report, path_api_cost=None, test_ind = None, is_hf = True, is_real_run=False):
    t_overall = perf_counter()

    # Load config file
    report_config(dir_home, cfg_file_path, system='VoucherVision')

    if isinstance(cfg_file_path, dict):
        cfg = cfg_file_path
    elif cfg_file_path is not None: # Using custom config from CLI
        cfg = load_custom_cfg(cfg_file_path)
    elif cfg_test is None:
        cfg = load_config_file(dir_home, cfg_file_path, system='VoucherVision')  # For VoucherVision
    else:
        cfg = cfg_test 

    # Check to see if there are subdirs
    # Yes --> use the names of the subsirs as run_name
    run_name, dirs_list, has_subdirs = check_for_subdirs_VV(cfg)
    print(f"run_name {run_name} dirs_list{dirs_list} has_subdirs{has_subdirs}")

    # Dir structure
    if is_real_run:
        progress_report.update_overall(f"Creating Output Directory Structure")
    print_main_start("Creating Directory Structure")
    Dirs = Dir_Structure(cfg)

    # logging.info("Hi")
    logger = start_logging(Dirs, cfg)

    # Check to see if required ML files are ready to use
    if is_real_run:
        progress_report.update_overall(f"Fetching LeafMachine2 Files")
    ready_to_use = fetch_data(logger, dir_home, cfg_file_path)
    assert ready_to_use, "Required ML files are not ready to use!\nThe download may have failed,\nor\nthe directory structure of LM2 has been altered"

    # Wrangle images and preprocess
    print_main_start("Gathering Images and Image Metadata")
    Project = Project_Info(cfg, logger, dir_home, Dirs) # Where file names are modified

    # Save config file
    save_config_file(cfg, logger, Dirs)

    # Detect Archival Components
    print_main_start("Locating Archival Components")
    Project = detect_archival_components(cfg, logger, dir_home, Project, Dirs, is_real_run, progress_report)

    # Save cropped detections
    crop_detections_from_images_VV(cfg, logger, dir_home, Project, Dirs)

    
    create_specimen_collage(cfg, logger, dir_home, Project, Dirs)


    # Process labels
    Voucher_Vision = VoucherVision(cfg, logger, dir_home, path_custom_prompts, Project, Dirs, is_hf)
    n_images = len(Voucher_Vision.img_paths)
    last_JSON_response, final_WFO_record, final_GEO_record, total_tokens_in, total_tokens_out, OCR_cost, OCR_tokens_in, OCR_tokens_out, OCR_cost_in, OCR_cost_out, OCR_method = Voucher_Vision.process_specimen_batch(progress_report, json_report, is_real_run)

    costs = save_token_info_as_csv(Dirs, cfg['leafmachine']['LLM_version'], path_api_cost, total_tokens_in, total_tokens_out, OCR_cost, OCR_tokens_in, OCR_tokens_out, OCR_cost_in, OCR_cost_out, OCR_method, n_images, dir_home, logger)
    total_cost = costs[0]
    parsing_cost = costs[1]
    ocr_cost = costs[2]

    t_overall_s = perf_counter()
    logger.name = 'Run Complete! :)'
    logger.info(f"[Total elapsed time] {round((t_overall_s - t_overall)/60)} minutes")
    space_saver(cfg, Dirs, logger)

    if is_real_run:
        progress_report.update_overall(f"Run Complete!")

    Voucher_Vision.close_logger_handlers()

    zip_filepath = None
    # Create Higging Face zip file
    dir_to_zip = os.path.join(Dirs.dir_home, Dirs.run_name)  
    zip_filename = Dirs.run_name

    # Creating a zip file
    zip_filepath = make_zipfile(dir_to_zip, zip_filename) ####################################################################################################### TODO Make this configurable
    if is_hf:
        upload_to_drive(zip_filepath, zip_filename, is_hf, cfg_private=Voucher_Vision.cfg_private, do_upload=True) ###################################### TODO Make this configurable
    else:
        upload_to_drive(zip_filepath, zip_filename, is_hf, cfg_private=Voucher_Vision.cfg_private, do_upload=False) ##################################### TODO Make this configurable

    return {'last_JSON_response': last_JSON_response, 
            'final_WFO_record': final_WFO_record, 
            'final_GEO_record': final_GEO_record, 
            'total_cost': total_cost, 
            'n_failed_OCR': Voucher_Vision.n_failed_OCR, 
            'n_failed_LLM_calls': Voucher_Vision.n_failed_LLM_calls, 
            'zip_filepath': zip_filepath,
            'parsing_cost': parsing_cost,
            'ocr_cost': ocr_cost,
            }

def make_zipfile(base_dir, output_filename):
    # Determine the directory where the zip file should be saved
    # Construct the full path for the zip file
    full_output_path = os.path.join(base_dir, output_filename)
    # Create the zip archive
    shutil.make_archive(full_output_path, 'zip', base_dir)
    # Return the full path of the created zip file
    return os.path.join(base_dir, output_filename + '.zip')

def load_custom_cfg(full_path_to_cfg):
    if not os.path.isabs(full_path_to_cfg):
        raise ValueError("The configuration path must be an absolute path.")

    try:
        with open(full_path_to_cfg, "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {full_path_to_cfg}")
    return cfg

def voucher_vision_OCR_test(cfg_file_path, dir_home, cfg_test, path_to_crop):
    # get_n_overall = progress_report.get_n_overall()
    # progress_report.update_overall(f"Working on {test_ind+1} of {get_n_overall}")

    # Load config file
    report_config(dir_home, cfg_file_path, system='VoucherVision')

    if cfg_test is None:
        cfg = load_config_file(dir_home, cfg_file_path, system='VoucherVision')  # For VoucherVision
    else:
        cfg = cfg_test 
    # user_cfg = load_config_file(dir_home, cfg_file_path)
    # cfg = Config(user_cfg)

    # Check to see if there are subdirs
    # Yes --> use the names of the subsirs as run_name
    run_name, dirs_list, has_subdirs = check_for_subdirs_VV(cfg)
    print(f"run_name {run_name} dirs_list{dirs_list} has_subdirs{has_subdirs}")

    # for dir_ind, dir_in in enumerate(dirs_list):
    #     if has_subdirs:
    #         cfg['leafmachine']['project']['dir_images_local'] = dir_in
    #         cfg['leafmachine']['project']['run_name'] = run_name[dir_ind]

    # Dir structure
    print_main_start("Creating Directory Structure")
    Dirs = Dir_Structure(cfg)

    # logging.info("Hi")
    logger = start_logging(Dirs, cfg)

    # Check to see if required ML files are ready to use
    ready_to_use = fetch_data(logger, dir_home, cfg_file_path)
    assert ready_to_use, "Required ML files are not ready to use!\nThe download may have failed,\nor\nthe directory structure of LM2 has been altered"

    # Wrangle images and preprocess
    print_main_start("Gathering Images and Image Metadata")
    Project = Project_Info(cfg, logger, dir_home, Dirs) # Where file names are modified

    # Save config file
    save_config_file(cfg, logger, Dirs)

    # Detect Archival Components
    print_main_start("Locating Archival Components")
    Project = detect_archival_components(cfg, logger, dir_home, Project, Dirs)

    # Save cropped detections
    crop_detections_from_images_VV(cfg, logger, dir_home, Project, Dirs)

    # Process labels
    Voucher_Vision = VoucherVision(cfg, logger, dir_home, None, Project, Dirs)
    last_JSON_response = Voucher_Vision.process_specimen_batch_OCR_test(path_to_crop)

if __name__ == '__main__':    
    is_test = False

    # Set LeafMachine2 dir 
    dir_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if is_test:
        cfg_file_path = os.path.join(dir_home, 'demo','demo.yaml') #'D:\Dropbox\LeafMachine2\LeafMachine2.yaml'
        # cfg_file_path = 'test_installation'

        cfg_testing = load_config_file_testing(dir_home, cfg_file_path)
        cfg_testing['leafmachine']['project']['dir_images_local'] = os.path.join(dir_home, cfg_testing['leafmachine']['project']['dir_images_local'][0], cfg_testing['leafmachine']['project']['dir_images_local'][1])
        cfg_testing['leafmachine']['project']['dir_output'] = os.path.join(dir_home, cfg_testing['leafmachine']['project']['dir_output'][0], cfg_testing['leafmachine']['project']['dir_output'][1])

        last_JSON_response = voucher_vision(cfg_file_path, dir_home, cfg_testing, None)
    else:
        cfg_file_path = None
        cfg_testing = None
        last_JSON_response = voucher_vision(cfg_file_path, dir_home, cfg_testing, None)