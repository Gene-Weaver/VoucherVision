"""
Lightweight subset of vouchervision_main for cold-start-sensitive callers
(e.g. the VoucherVisionGO Flask app).

Importing the full vouchervision_main pulls component_detector (torch,
matplotlib, cv2), general_utils (streamlit, torch, cv2), utils_VoucherVision,
utils_hf, fetch_data, etc. — tens of seconds of import time on Cloud Run
cold starts. Mirror only the helpers that hot-path callers actually need.
"""
import os
import yaml


def load_custom_cfg(full_path_to_cfg):
    if not os.path.isabs(full_path_to_cfg):
        raise ValueError("The configuration path must be an absolute path.")

    try:
        with open(full_path_to_cfg, "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {full_path_to_cfg}")
    return cfg
