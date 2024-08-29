import streamlit as st
import os, json, re, datetime, tempfile, yaml
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
import base64
from PIL import Image
from PIL import Image
from io import BytesIO
from shutil import copyfileobj, copyfile

# from vouchervision.general_utils import get_cfg_from_full_path


def setup_streamlit_config(dir_home):
    # Define the directory path and filename
    dir_path = os.path.join(dir_home, ".streamlit")
    file_path = os.path.join(dir_path, "config.toml")

    # Check if directory exists, if not create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Create or modify the file with the provided content
    config_content = f"""
    [theme]
    base = "dark"
    primaryColor = "#00ff00"

    [server]
    enableStaticServing = false
    runOnSave = true
    port = 8524
    maxUploadSize = 5000
    """

    with open(file_path, "w") as f:
        f.write(config_content.strip())


def save_uploaded_file_local(directory_in, directory_out, img_file_name, image=None):
    if not os.path.exists(directory_out):
        os.makedirs(directory_out)

    # Assuming img_file_name includes the extension
    img_file_base, img_file_ext = os.path.splitext(img_file_name)
    
    full_path_out = os.path.join(directory_out, img_file_name)
    full_path_in = os.path.join(directory_in, img_file_name)

    # Check if the file extension is .pdf (or add other conditions for different file types)
    if img_file_ext.lower() == '.pdf':
        # Copy the file from the input directory to the output directory
        copyfile(full_path_in, full_path_out)
        return full_path_out
    else:
        if image is None:
            try:
                with Image.open(full_path_in) as image:
                    image.save(full_path_out, "JPEG")
                # Return the full path of the saved image
                return full_path_out
            except:
                pass
        else:
            try:
                image.save(full_path_out, "JPEG")
                return full_path_out
            except:
                pass
            
def save_uploaded_file(directory, uploaded_file, image=None):
    if not os.path.exists(directory):
        os.makedirs(directory)

    full_path = os.path.join(directory, uploaded_file.name)

    # Handle PDF and Image files differently
    if uploaded_file.name.lower().endswith('.pdf'):
        # Save PDF file
        try:
            with open(full_path, 'wb') as out_file:
                if hasattr(uploaded_file, 'read'):
                    # This is a file-like object
                    out_file.write(uploaded_file.read())
                else:
                    # If uploaded_file is a path string
                    with open(uploaded_file, 'rb') as fd:
                        out_file.write(fd.read())
            if os.path.getsize(full_path) == 0:
                raise ValueError(f"The file {uploaded_file.name} is empty.")
            return full_path
        except Exception as e:
            st.error(f"Failed to save PDF file {uploaded_file.name}. Error: {e}")
            return None
    else:
        # Handle image files
        if image is None:
            try:
                with Image.open(uploaded_file) as image:
                    image.save(full_path, "JPEG")
            except Exception as e:
                st.error(f"Failed to save image file {uploaded_file.name}. Error: {e}")
                return None
        else:
            try:
                image.save(full_path, "JPEG")
            except Exception as e:
                st.error(f"Failed to save processed image file {uploaded_file.name}. Error: {e}")
                return None

        if os.path.getsize(full_path) == 0:
            st.error(f"The image file {uploaded_file.name} is empty.")
            return None

    return full_path


# def save_uploaded_file(directory, img_file, image=None): # not working with pdfs
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     full_path = os.path.join(directory, img_file.name) ########## TODO THIS MUST BE MOVED TO conditional specific location

#     # Assuming the uploaded file is an image
#     if img_file.name.lower().endswith('.pdf'):
#         with open(full_path, 'wb') as out_file:
#             # If img_file is a file-like object (e.g., Django's UploadedFile),
#             # you can use copyfileobj or read chunks.
#             # If it's a path, you'd need to open and then save it.
#             if hasattr(img_file, 'read'):
#                 # This is a file-like object
#                 copyfileobj(img_file, out_file)
#             else:
#                 # If img_file is a path string
#                 with open(img_file, 'rb') as fd:
#                     copyfileobj(fd, out_file)    
#             return full_path
#     else:
#         if image is None:
#             try:
#                 with Image.open(img_file) as image:
#                     full_path = os.path.join(directory, img_file.name)
#                     image.save(full_path, "JPEG")
#                 # Return the full path of the saved image
#                 return full_path
#             except:
#                 try:
#                     with Image.open(os.path.join(directory,img_file)) as image:
#                         full_path = os.path.join(directory, img_file)
#                         image.save(full_path, "JPEG")
#                     # Return the full path of the saved image
#                     return full_path
#                 except:
#                     with Image.open(img_file.name) as image:
#                         full_path = os.path.join(directory, img_file.name)
#                         image.save(full_path, "JPEG")
#                     # Return the full path of the saved image
#                     return full_path
#         else:
#             try:
#                 full_path = os.path.join(directory, img_file.name)
#                 image.save(full_path, "JPEG")
#                 return full_path
#             except:
#                 full_path = os.path.join(directory, img_file)
#                 image.save(full_path, "JPEG")
#                 return full_path
            


# def save_uploaded_file(directory, uploaded_file, image=None):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     full_path = os.path.join(directory, uploaded_file.name)

#     # Handle PDF files
#     if uploaded_file.name.lower().endswith('.pdf'):
#         with open(full_path, 'wb') as out_file:
#             if hasattr(uploaded_file, 'read'):
#                 copyfileobj(uploaded_file, out_file)
#             else:
#                 with open(uploaded_file, 'rb') as fd:
#                     copyfileobj(fd, out_file)
#         return full_path
#     else:
#         if image is None:
#             try:
#                 with Image.open(uploaded_file) as img:
#                     img.save(full_path, "JPEG")
#             except:
#                 with Image.open(full_path) as img:
#                     img.save(full_path, "JPEG")
#         else:
#             try:
#                 image.save(full_path, "JPEG")
#             except:
#                 image.save(os.path.join(directory, uploaded_file.name), "JPEG")
#         return full_path
    
def save_uploaded_local(directory, img_file, image=None):
    name = img_file.split(os.path.sep)[-1]
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Assuming the uploaded file is an image
    if image is None:
        with Image.open(img_file) as image:
            full_path = os.path.join(directory, name)
            image.save(full_path, "JPEG")
        # Return the full path of the saved image
        return os.path.join('uploads_small',name)
    else:
        full_path = os.path.join(directory, name)
        image.save(full_path, "JPEG")
        return os.path.join('.','uploads_small',name)
    
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def check_prompt_yaml_filename(fname):
    # Check if the filename only contains letters, numbers, underscores, and dashes
    pattern = r'^[\w-]+$'
    
    # The \w matches any alphanumeric character and is equivalent to the character class [a-zA-Z0-9_].
    # The hyphen - is literally matched.

    if re.match(pattern, fname):
        return True
    else:
        return False

def report_violation(file_name, is_hf=True):
    # Format the current date and time
    current_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    violation_file_name = f"violation_{current_time}.yaml"  # Updated variable name to avoid confusion
    
    # Create a temporary YAML file in text mode
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.yaml') as temp_file:
        # Example content - customize as needed
        content = {
            'violation_time': current_time,
            'notes': 'This is an autogenerated violation report.',
            'name_of_file': file_name,
        }
        # Write the content to the temporary YAML file in text mode
        yaml.dump(content, temp_file, default_flow_style=False)
        temp_filepath = temp_file.name

    # Now upload the temporary file
    upload_to_drive(temp_filepath, violation_file_name, is_hf=is_hf)

    # Optionally, delete the temporary file if you don't want it to remain on disk after uploading
    os.remove(temp_filepath)

# Function to upload files to Google Drive
def upload_to_drive(filepath, filename, is_hf=True, cfg_private=None, do_upload = True):
    if do_upload:
        creds = get_google_credentials(is_hf=is_hf, cfg_private=cfg_private)
        if creds:
            service = build('drive', 'v3', credentials=creds)

            # Get the folder ID from the environment variable
            if is_hf:
                folder_id = os.environ.get('GDRIVE_FOLDER_ID')  # Renamed for clarity
            else:
                folder_id = cfg_private['google']['GDRIVE_FOLDER_ID']  # Renamed for clarity


            if folder_id:
                file_metadata = {
                    'name': filename,
                    'parents': [folder_id]
                }

                # Determine the mimetype based on the file extension
                if filename.endswith('.yaml') or filename.endswith('.yml') or filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    mimetype = 'application/x-yaml'
                elif filepath.endswith('.zip'):
                    mimetype = 'application/zip'
                else:
                    # Set a default mimetype if desired or handle the unsupported file type
                    print("Unsupported file type")
                    return None

                # Upload the file
                try:
                    media = MediaFileUpload(filepath, mimetype=mimetype)
                    file = service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id'
                    ).execute()
                    print(f"Uploaded file with ID: {file.get('id')}")
                except Exception as e:
                    msg = f"If the following error is '404 cannot find file...' then you need to share the GDRIVE folder with your Google API service account's email address. Open your Google API JSON file, find the email account that ends with '@developer.gserviceaccount.com', go to your Google Drive, share the folder with this email account. {e}"
                    print(msg)
                    raise Exception(msg)
        else:
            print("GDRIVE_API environment variable not set.")

def get_google_credentials(is_hf=True, cfg_private=None): # Also used for google drive
    if is_hf:
        creds_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json_str))
        return credentials
    else:
        with open(cfg_private['google']['GOOGLE_APPLICATION_CREDENTIALS'], 'r') as file:
            data = json.load(file)
            creds_json_str = json.dumps(data)
            credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json_str))
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_json_str
            return credentials