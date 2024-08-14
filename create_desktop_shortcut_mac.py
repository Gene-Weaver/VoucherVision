import os

def create_mac_shortcut(script_path, venv_path, app_name, app_dir, icon_path):
    print(f"Script Path: {script_path}")
    print(f"Venv Path: {venv_path}")
    print(f"App Name: {app_name}")
    print(f"App Dir: {app_dir}")
    print(f"Icon Path: {icon_path}")

    # Construct the command to activate the venv and run the script
    venv_path_full = os.path.join(venv_path,'bin','activate')
    command = f"source '{venv_path_full}' && python '{script_path}'"
    print(f"Command: {command}")

    # AppleScript to run the command in Terminal
    apple_script = f'''
    tell application "Terminal"
        do script "{command}"
    end tell
    '''

    # Path to save the AppleScript
    apple_script_path = os.path.join(app_dir, f"{app_name}.applescript")
    print(f"AppleScript Path: {apple_script_path}")

    # Save the AppleScript
    with open(apple_script_path, 'w') as file:
        file.write(apple_script)

    # Convert the AppleScript to an application
    os.system(f'osacompile -o "{os.path.join(app_dir, app_name)}.app" "{apple_script_path}"')

    # Define the path for the app's icon
    app_icon_path = os.path.join(app_dir, f"{app_name}.app/Contents/Resources/applet.icns")

    # Replace the default icon with the custom icon
    if os.path.exists(icon_path):
        os.system(f'cp "{icon_path}" "{app_icon_path}"')
        print(f"Icon replaced at {app_icon_path}")
    else:
        print(f"Icon file not found at {icon_path}")

    print(f"Application created at {os.path.join(app_dir, app_name)}.app")

if __name__ == "__main__":
    # Example usage
    create_mac_shortcut(
        script_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_VoucherVision.py'),
        venv_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), '.venv_VV'),
        app_name='VoucherVision',
        app_dir=os.path.dirname(os.path.realpath(__file__)),
        icon_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'img', 'VoucherVision.icns')
    )
