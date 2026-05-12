"""
Lightweight subset of general_utils for cold-start-sensitive callers
(e.g. the VoucherVisionGO Flask app).

Importing the full general_utils module pulls in torch, cv2, streamlit,
tiktoken, pandas, matplotlib, etc. — tens of seconds of import time on
Cloud Run cold starts. Re-export only the functions that callers on the
hot serving path actually need, with no heavy top-level imports.
"""
import yaml


def calculate_cost(LLM_version, path_api_cost, total_tokens_in, total_tokens_out):
    with open(path_api_cost, 'r') as file:
        cost_data = yaml.safe_load(file)

    if LLM_version in cost_data:
        rates = cost_data[LLM_version]
        cost_in = rates['in'] * (total_tokens_in / 1000000)
        cost_out = rates['out'] * (total_tokens_out / 1000000)
        total_cost = cost_in + cost_out
    else:
        raise ValueError(f"LLM version {LLM_version} not found in the cost data")

    return cost_in, cost_out, total_cost, rates['in'], rates['out']


def get_cfg_from_full_path(path_cfg):
    with open(path_cfg, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)
    return cfg
