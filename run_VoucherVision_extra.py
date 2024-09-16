import os
import sys
import socket
import time
import subprocess
import random

# pip install protobuf==3.20.0
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 nope
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 nope
# pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# pip install protobuf==3.20.0

def find_available_port(start_port, end_port):
    ports = list(range(start_port, end_port + 1))
    random.shuffle(ports)
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except socket.error:
                print(f"Port {port} is in use, trying another port...")
    raise ValueError(f"Could not find an available port in the range {start_port}-{end_port}.")

def find_app_path():
    # Check if we're running from within a PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # If we are running in a PyInstaller bundle
        bundle_dir = sys._MEIPASS
        app_path = os.path.join(bundle_dir, 'app.py')
    else:
        # If we're not frozen, use the original app.py path
        app_path = os.path.join(os.getcwd(), 'app.py')

    if not os.path.exists(app_path):
        raise FileNotFoundError(f"app.py not found at {app_path}")

    return app_path

def run_streamlit(port):
    app_path = "app.py"  # Assuming app.py is in the root directory

    # Check if we're running from within a PyInstaller bundle
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS
        python_executable = os.path.join(bundle_dir, "python.exe")
    else:
        python_executable = sys.executable

    cmd = [
        python_executable,
        "-m", "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        f"--server.port={port}",
        "--server.maxUploadSize=51200",
    ]
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    return process

if __name__ == "__main__":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    start_port = 8530
    end_port = 8599
    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            free_port = find_available_port(start_port, end_port)
            print(f"Found available port: {free_port}")
            
            process = run_streamlit(free_port)
            
            # Wait and capture output
            start_time = time.time()
            while time.time() - start_time < 30:  # Wait for up to 30 seconds
                output = process.stdout.readline()
                if output:
                    print(output.strip())
                    if "You can now view your Streamlit app in your browser." in output:
                        print("Streamlit started successfully.")
                        break
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            
            if process.poll() is None:
                print(f"Streamlit process is running. Please open http://localhost:{free_port} in your browser.")
                while process.poll() is None:
                    time.sleep(1)
                print("Streamlit process ended.")
                break
            else:
                stdout, stderr = process.communicate()
                print(f"Streamlit failed to start. Error:\n{stderr}")
                
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        retry_count += 1
        if retry_count < max_retries:
            print(f"Retrying... (Attempt {retry_count} of {max_retries})")
            time.sleep(2)
        else:
            print("Failed to start the application after multiple attempts.")

    print("Press Enter to exit...")
    input()