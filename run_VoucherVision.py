import streamlit.web.cli as stcli
import os, sys, socket, subprocess, git
from pathlib import Path
from importlib.metadata import distributions
from packaging.requirements import Requirement
from packaging import version
# pip install protobuf==3.20.0
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 nope
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 nope
# pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# pip install protobuf==3.20.0

def update_setuptools():
    """Update the setuptools package using pip."""
    print("Updating setuptools to avoid compatibility issues...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"], check=True)
        print("setuptools updated successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to update setuptools:", e)
        sys.exit(1)  # Exit if setuptools can't be updated
        

def normalize_package_name(name):
    """Normalize package names to match the naming convention used in distributions."""
    return name.lower().replace('-', '_').replace(' ', '')

def get_installed_distributions():
    """Retrieve installed distributions as a dict with normalized package names as keys."""
    return {normalize_package_name(dist.metadata['Name']): dist.version for dist in distributions()}

def check_and_fix_requirements(requirements_file):
    """
    Checks if installed packages in the virtual environment satisfy the requirements specified
    in the requirements.txt file and fixes them if they do not.
    """
    installed_distributions = get_installed_distributions()
    missing_or_incompatible = []
    
    with open(requirements_file, 'r', encoding='utf-16') as req_file:
        requirements = [Requirement(line.strip()) for line in req_file if line.strip() and not line.startswith('#')]

    for req in requirements:
        pkg_name = normalize_package_name(req.name)
        if pkg_name not in installed_distributions:
            missing_or_incompatible.append(f"{req} is not installed")
        elif req.specifier:
            installed_ver = version.parse(installed_distributions[pkg_name])
            if installed_ver not in req.specifier:
                missing_or_incompatible.append(f"{pkg_name}=={installed_distributions[pkg_name]} does not satisfy {req}")

    if missing_or_incompatible:
        print("The following packages are missing or incompatible:")
        for issue in missing_or_incompatible:
            print(f"  - {issue}")
        
        print("Attempting to fix the package issues by running 'pip install -r requirements.txt'...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
            print("Packages have been successfully updated.")
        except subprocess.CalledProcessError as e:
            print("Failed to install packages:", e)
    else:
        print("All requirements are satisfied.")


def find_github_desktop_git():
    """Search for the most recent GitHub Desktop Git installation."""
    # Base path where GitHub Desktop versions are located
    base_path = Path(f"C:/Users/{os.getlogin()}/AppData/Local/GitHubDesktop/")
    print(f"base_path: {base_path}")

    # Searching recursively for git.exe within any directories under the base path
    versions = sorted(base_path.rglob('git.exe'), key=lambda x: x.parent, reverse=True)
    for git_path in versions:
        print(f"git_path: {git_path}")
        if "app-" in str(git_path.parent):  # Ensuring it's in an 'app-' directory if that's still relevant
            print(f"git_path_exists: TRUE")
            return str(git_path)

    print(f"git_path_exists: FALSE")
    return None

def update_repository(repo_path):
    print(f"changing path to: {repo_path}")
    os.chdir(repo_path)
    print(f"changed to: {repo_path}")

    # """Attempts to update the repository using the system's git or GitHub Desktop's git."""
    # try:
    #     # Try using the system's git command
    #     result = subprocess.run(["git", "pull"], capture_output=True, text=True, check=True)
    #     print(result.stdout)
    #     if result.returncode == 0:
    #         print("Repository updated successfully.")
    # except Exception as e:
    #     print(f"Error updating repository with system Git: {e}")
        # Fallback: use GitHub Desktop's Git executable
    try:
        # Open the existing repository at the specified path
        repo = git.Repo(repo_path)
        # Check for the current working branch
        current_branch = repo.active_branch
        print(f"Updating repository on branch: {current_branch.name}")

        # Pulls updates for the current branch
        origin = repo.remotes.origin
        result = origin.pull()

        # Check if the pull was successful
        if result[0].flags > 0:
            print("Repository updated successfully.")
        else:
            print("No updates were available.")
                
    except Exception as e:
        print(f"Error while updating repository: {e}")

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
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    start_port = 8501
    end_port = 8599
    retry_count = 0

    # repo_path = resolve_path(os.path.dirname(__file__))
    # print(f"repo_path: {repo_path}")

    # update_setuptools()
    # requirements_file = 'requirements.txt'
    # check_and_fix_requirements(resolve_path(os.path.join(os.path.dirname(__file__),'requirements.txt')))

    # try:
    #     update_repository(repo_path)
    # except:
    #     print(f"Could not update VVE using git pull.")
    #     print(f"Make sure that 'Git' is installed and can be accessed by this user account.")

    # # Update again in case the pull introduced a new package
    # check_and_fix_requirements(resolve_path(os.path.join(os.path.dirname(__file__),'requirements.txt')))

    try:
        free_port = find_available_port(start_port, end_port)
        sys.argv = [
            'streamlit',
            'run',
            resolve_path(os.path.join(os.path.dirname(__file__),'app.py')),
            '--global.developmentMode=false',
            f'--server.maxUploadSize=51200',
            f'--server.enableStaticServing=true',
            f'--server.runOnSave=true',
            f'--server.port={free_port}',
            f'--theme.primaryColor=#16a616',
            f'--theme.backgroundColor=#1a1a1a',
            f'--theme.secondaryBackgroundColor=#303030',
            f'--theme.textColor=cccccc',
        ]
        sys.exit(stcli.main())

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    else:
        print("Failed to start the application after multiple attempts.")