import os, sys, yaml, argparse
from vouchervision.vouchervision_main import voucher_vision, load_custom_cfg

def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Run VoucherVision with the given configuration file.")
    parser.add_argument(
        "config_file",
        nargs="?",
        default="custom_VV_config.yaml",
        help="Path to the configuration YAML file. Use your_config_filename.yaml for local files in the same directory as this script, or provide a full path. Defaults to 'custom_VV_config.yaml' in the home directory."
    )
    args = parser.parse_args()

    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve the configuration file path
    if not os.path.isabs(args.config_file):
        config_file_path = os.path.join(script_dir, args.config_file)
    else:
        config_file_path = args.config_file

    # Ensure the configuration file exists
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found: {config_file_path}")
        sys.exit(1)

    cfg = load_custom_cfg(config_file_path)
    # Set directory home and other default values
    dir_home = os.path.dirname(__file__)
    path_custom_prompts = os.path.join(dir_home,'custom_prompts',cfg['leafmachine']['project']['prompt_version'])
    cfg_test = None
    progress_report = None  # Replace with an appropriate progress report object if necessary
    json_report = False 
    path_api_cost = os.path.join(dir_home,'api_cost','api_cost.yaml')
    test_ind = None
    is_hf = False
    is_real_run = False

    # Run the VoucherVision process
    result = voucher_vision(
        cfg_file_path=config_file_path,
        dir_home=dir_home,
        path_custom_prompts=path_custom_prompts,
        cfg_test=cfg_test,
        progress_report=progress_report,
        json_report=json_report,
        path_api_cost=path_api_cost,
        test_ind=test_ind,
        is_hf=is_hf,
        is_real_run=is_real_run
    )

    # Print result summary
    print("VoucherVision completed successfully!")
    print(result)

if __name__ == "__main__":
    main()
