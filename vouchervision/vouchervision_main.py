'''
VoucherVision - based on LeafMachine2 Processes
'''
import os, inspect, sys, logging, subprocess
from time import perf_counter
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from vouchervision.component_detector.component_detector import detect_plant_components, detect_archival_components
from general_utils import check_for_subdirs_VV, load_config_file, load_config_file_testing, report_config, save_config_file, subset_dir_images, crop_detections_from_images_VV
from general_utils import print_main_start
from directory_structure_VV import Dir_Structure
from data_project import Project_Info
from LM2_logger import start_logging
from fetch_data import fetch_data
from utils_VoucherVision import VoucherVision, space_saver


def voucher_vision(cfg_file_path, dir_home, cfg_test, progress_report, test_ind = None):
    # get_n_overall = progress_report.get_n_overall()
    # progress_report.update_overall(f"Working on {test_ind+1} of {get_n_overall}")

    t_overall = perf_counter()

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
    Voucher_Vision = VoucherVision(cfg, logger, dir_home, Project, Dirs)
    last_JSON_response = Voucher_Vision.process_specimen_batch(progress_report)

    t_overall_s = perf_counter()
    logger.name = 'Run Complete! :)'
    logger.info(f"[Total elapsed time] {round((t_overall_s - t_overall)/60)} minutes")
    space_saver(cfg, Dirs, logger)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    return last_JSON_response

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