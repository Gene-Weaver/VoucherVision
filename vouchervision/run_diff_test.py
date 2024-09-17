import streamlit.web.cli as stcli
import os
import sys
import socket
import subprocess

# git clone https://github.com/Gene-Weaver/VoucherVision.git
# git submodule update --init --recursive

def uppercase_drive_letter(path):
    if os.name == 'nt' and len(path) >= 2 and path[1] == ':':
        path = path[0].upper() + path[1:]
    return path

# Function to find an available port
def find_available_port(start_port, max_attempts=1000):
    port = start_port
    attempts = 0
    while attempts < max_attempts:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port  # Return the current port if successful
            except socket.error:
                port += 1
                attempts += 1
    raise ValueError(f"Could not find an available port within {max_attempts} attempts starting from port {start_port}.")


# Function to resolve the app path
def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


# Function to start the Yarn dev server
def start_yarn_dev(dir_home):
    sub_command = ["git", "submodule", "update", "--init", "--recursive"]
    yarn_command1 = ["yarn", "install"]
    yarn_command2 = ["yarn", "cache", "clean"]
    yarn_command3 = ["yarn", "dev"]

    quick_diff_path = os.path.join(dir_home, "VoucherVision-quick-diff")
    quick_diff_path = uppercase_drive_letter(quick_diff_path)
    print(quick_diff_path)
    
    subprocess.Popen(
        sub_command,
        cwd=quick_diff_path,
        shell=True
    )
    subprocess.Popen(
        yarn_command1,
        cwd=quick_diff_path,
        shell=True
    )
    subprocess.Popen(
        yarn_command2,
        cwd=quick_diff_path,
        shell=True
    )
    subprocess.Popen(
        yarn_command3,
        cwd=quick_diff_path,
        shell=True
    )



# Launcher logic to run both Yarn and Streamlit app
if __name__ == "__main__":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    dir_home = uppercase_drive_letter(os.path.dirname(__file__))

    # Start Yarn server
    start_yarn_dev(dir_home)

    start_port = 8532
    try:
        free_port = find_available_port(start_port)
        sys.argv = [
            "streamlit",
            "run",
            resolve_path(os.path.join(dir_home, "diff_test.py")),
            f"--server.port={free_port}",
            "--server.maxUploadSize=51200",
        ]
        sys.exit(stcli.main())
    except ValueError as e:
        print(e)
