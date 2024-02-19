import streamlit.web.cli as stcli
import os, sys, socket

# pip install protobuf==3.20.0
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 nope
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# pip install protobuf==3.20.0

def find_available_port(start_port, max_attempts=1000):
    port = start_port
    attempts = 0
    while attempts < max_attempts:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                # If successful, return the current port
                return port
            except socket.error:
                # If the port is in use, increment the port number and try again
                port += 1
                attempts += 1
    # Optional: Return None or raise an exception if no port is found within the attempts limit
    raise ValueError(f"Could not find an available port within {max_attempts} attempts starting from port {start_port}.")


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    dir_home = os.path.dirname(__file__)

    start_port = 8528
    try:
        free_port = find_available_port(start_port)
        sys.argv = [
            "streamlit",
            "run",
            resolve_path(os.path.join(os.path.dirname(__file__),"app.py")),
            # resolve_path(os.path.join(dir_home,"vouchervision", "VoucherVision_GUI.py")),
            "--global.developmentMode=false",
            # "--server.port=8545",
            f"--server.port={free_port}",
            # Toggle below for HF vs Local
            # "--is_hf=1",
            # "--is_hf=0",
        ]
        sys.exit(stcli.main())
    except ValueError as e:
        print(e)