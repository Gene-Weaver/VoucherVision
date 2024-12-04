import os
import sys
import yaml
from vouchervision.vouchervision_main import voucher_vision, load_custom_cfg

def main(path_to_dir, redo_completed_projects):
    # Validate the base directory
    if not os.path.isdir(path_to_dir):
        print(f"Error: {path_to_dir} is not a valid directory")
        sys.exit(1)
    
    # Statically set config file path
    config_file = os.path.join(os.path.dirname(__file__), 'custom_VV_config.yaml')
    
    # Validate config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found at {config_file}")
        sys.exit(1)
    
    # Load configuration
    try:
        cfg = load_custom_cfg(config_file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Find subdirectories in the input directory
    subdirs = [os.path.join(path_to_dir, d) for d in os.listdir(path_to_dir) if os.path.isdir(os.path.join(path_to_dir, d))]
    
    if not subdirs:
        print(f"No subdirectories found in: {path_to_dir}")
        sys.exit(1)
    
    # Determine home directory and custom prompts path
    dir_home = os.path.dirname(os.path.abspath(config_file))
    path_custom_prompts = os.path.join(dir_home, 'custom_prompts', cfg['leafmachine']['project']['prompt_version'])
    
    # Process each subdirectory
    for subdir_path in subdirs:
        do_run_VV = False
        subdir_name = os.path.basename(subdir_path)
        
        # Update configuration for the current subdirectory
        cfg['leafmachine']['project']['dir_images_local'] = subdir_path
        cfg['leafmachine']['project']['run_name'] = subdir_name
        dir_output = cfg['leafmachine']['project']['dir_output']

        expected_output_dir = os.path.join(dir_output, subdir_name)

        if not redo_completed_projects:
            if not os.path.exists(expected_output_dir):
                do_run_VV = True # Dir needs to be processed
            else:
                do_run_VV = False # Dir already processed
                print(f"Project [{subdir_name}] has already been completed and can be found here [{expected_output_dir}]")
        else:
            do_run_VV = True

        if do_run_VV:
            # Save updated configuration to a temporary file
            try:
                # Run VoucherVision for the current subdirectory
                print(f"Processing subdirectory: {subdir_path}")
                result = voucher_vision(
                    cfg_file_path=cfg,
                    dir_home=dir_home,
                    path_custom_prompts=path_custom_prompts,
                    cfg_test=None,
                    progress_report=None,
                    json_report=False,
                    path_api_cost=os.path.join(dir_home, 'api_cost', 'api_cost.yaml'),
                    test_ind=None,
                    is_hf=False,
                    is_real_run=False # True is for streamlit GUI
                )
                print(f"Finished processing {subdir_path}")
                print(result)
            
            except Exception as e:
                print(f"Error processing {subdir_path}: {e}")

if __name__ == "__main__":
    # Hardcoded input directory (modify as needed)
    path_to_dir = 'S:/Imagers_CR2_only/1-Incomplete/AllAsia_Workbench/_NotesFromNature'
    redo_completed_projects = False # False will skip projects that have already been completed, can be used like "pick up where I left off"
    main(path_to_dir, redo_completed_projects)
