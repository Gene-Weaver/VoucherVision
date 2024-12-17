###                                 Instructions
### To run this file: 
###      1. Edit the paths below
###      2. In the terminal, change directories and move into the directory that contains this file, using 'cd'
###      3. Type `python split_finished_batch_by_geography_Africa.py`
###      4. The code will run
### 
### This code is designed to split the VoucherVision output based on geopgraphy to find certain locations, (e.g. Africa).
### 
### 
### 
### THIS CODE SHOULD ONLY BE RUN AFTER SPLITTING FOR GEOGRAPHY WITH THE REGULAR SCRIPT. i.e. split for _exAA first, then look for something else with this script
### ALSO this version of the script will not alter the original folder, whereas split_finished_batch_by_geography.py does alter the original folder's contents
### ALSO this script looks for the "transcribed_prior_to_subsetting.xlsx" file and uses it to do the sorting
### 
### 
### 
### This code will 
### 1. Make a full copy of the original project and all its files, saving it to the same location with the new suffix. 
###    If the project with suffix already exists, skip it, do nothing else
###     e.g. ORIGINAL S:/VoucherVision/Unassigned/2023_10_13_I4_kathia_AllAsia_Api
###          NEW      S:/VoucherVision/Unassigned/2023_10_13_I4_kathia_AllAsia_Api_Africa
###     where the _Africa is the suffix you provide
###     - NOTE: ALL images will exist in BOTH projects so we don't break any links. The transcribed.xlsx file dictates which specimens the person sees in the VVEditor
### 2. Delete the .zip folders in BOTH projects
###     - remember that we drag the .zip file into editor, so we need to create a new verison with the new transcribed.xlsx files
### 3. Load the continent and country parameters. THese define the geographies that you want to keep in the ORIGINAL folder. All 
###    exclusions will be moved to the NEW folder.
###     - It will exclude the continents that you provide
###     - It will include the countries that you provide
### 4. It will save a fallback copy of the original transcribed.xlsx file in ORIGINAL and NEW called transcribed_prior_to_geography_split.xlsx
###     - you can use this as a reference
### 5. Filter based on geographies. All unkowns will remain in the ORIGINAL
###     - e.g. If working on the All Asia grant, a specimen no geo information will be sent to the person for review along with the Asian specimens
###            If there are African specimens, they will be moved to the _Africa project's transcribed.xlsx
### 6. Save the two transcribed.xlsx versions, overwriting the original transcribed.xlsx version in both folders.
### 7 Create a new .zip file in both ORIGINAL and NEW, which will be used in the VVEditor
### 
### Provide the following:
###     Input:
###         path_input_project
###             path to project that you want to split
###             e.g. "C:/Users/willwe/Downloads/two_batches/2023_11_01_I4_mikedmac_AllAsia_Ole"  # a single project
###             e.g. "C:/Users/willwe/Downloads/two_batches"                                     # a folder that contains multiple projects
path_input_project = "C:/Users/willwe/Downloads/two_batches"

###         suffix_new_project (should not change unless the scope of the project changes)
###             suffix for new project (original_project_name{suffix_new_project})
###             e.g. "_Africa"
suffix_new_project = "_Africa"

###         suffix_to_skip (should not change unless the scope of the project changes)
###             suffix(s) that we can ignore. for example, if we split the VoucherVision output to create an _exAA set, then to find 
###             specimens from Africa we don't want to look inside those folders. Add the suffix to the list.
###             e.g. ["_exAA",]
suffix_to_skip = ["_exAA",]

###         path_countries_keep (should not change unless the scope of the project changes)
###             path to countries_keep.csv
###             e.g. "S:/VoucherVision/Tools/Country_Include_Africa.csv"
path_countries_keep = "S:/VoucherVision/Tools/Country_Include_Africa.csv"

###         path_continents_exclude (should not change unless the scope of the project changes)
###             path to continents_exclude.csv
###             e.g. "S:/VoucherVision/Tools/Continent_Exclude_Africa.csv"
path_continents_exclude = "S:/VoucherVision/Tools/Continent_Exclude_Africa.csv"






### Do not edit below this text
import os, glob, shutil
import pandas as pd
def copy_project_folder(path_input_project, suffix_new_project):
    """Copies the input project folder to a new project folder with a suffix."""
    parent_dir = os.path.dirname(path_input_project)
    project_name = os.path.basename(path_input_project)
    new_project_name = f"{project_name}{suffix_new_project}"
    new_project_path = os.path.join(parent_dir, new_project_name)

    if os.path.exists(new_project_path):
        print(f"    Skipping project [{project_name}] since [{new_project_path}] already exists")
        return new_project_path, False
    
    shutil.copytree(path_input_project, new_project_path)
    return new_project_path, True

def delete_old_zip_files(directory):
    """Recursively deletes all .zip files in a given directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"    Deleted ZIP file: {file_path}")
                except Exception as e:
                    print(f"    Failed to delete {file_path}: {e}")

def load_keep_files(path_countries_keep, path_continents_keep):
    """Loads the country and continent keep files."""
    countries_keep = set(pd.read_csv(path_countries_keep, header=None).iloc[:, 0].dropna().str.strip().tolist())
    continents_keep = set(pd.read_csv(path_continents_keep, header=None).iloc[:, 0].dropna().str.strip().tolist())
    return countries_keep, continents_keep

def split_transcription_file(path_transcription, countries_keep, continents_exclude):
    """Splits the transcription file into two parts based on country and continent filters."""
    # Read the transcription file
    df = pd.read_excel(path_transcription, engine='openpyxl', dtype=str)
    df_pre = pd.read_excel(path_transcription, engine='openpyxl', dtype=str)

    # Ensure 'country' and 'continent' columns exist
    if 'country' not in df.columns or 'continent' not in df.columns:
        raise ValueError("The transcription file must contain 'country' and 'continent' columns.")
    
    # Create a mask to identify which rows should be part of the original transcription
    keep_mask = (
        df['country'].str.lower().isin({country.lower() for country in countries_keep}) | 
        ~df['continent'].str.lower().isin({continent.lower() for continent in continents_exclude}) | 
        (df['country'].isna() & df['continent'].isna())
    )
    
    # Filter for the new project transcription (all other rows)
    df_new = df[~keep_mask].copy()  # Copy to avoid issues with SettingWithCopyWarning
    
    # Replace 'country' with 'X' for rows not in the original transcription
    df.loc[~keep_mask, 'country'] = 'X'
    
    # Convert empty strings back to native empty values for non-object dtypes in df and df_pre
    filename_index = df.columns.get_loc('filename') if 'filename' in df.columns else None
    if filename_index is not None:
        columns_to_clear = [col for i, col in enumerate(df.columns[:filename_index]) if col not in ['catalogNumber', 'country']]
        for col in columns_to_clear:
            df.loc[df['country'] == 'X', col] = ""
    
    # Filter for the original transcription with updated 'country' values
    df = df.fillna("")
    df_new = df_new.fillna("")
    df_pre = df_pre.fillna("")
    
    df_original = df.copy()
    
    return df_pre, df_original, df_new


def save_transcription_files(path_transcription, path_transcription_new, df_original, df_new, df_pre, fallback_suffix, is_already_processed=False):
    """Saves the original and new transcription files."""
    if is_already_processed:
        # Save the original transcription back to its original location
        df_original.to_excel(path_transcription_new, index=False, engine='openpyxl')
        
        # Save the new transcription to the copied project's transcription file
        # df_new.to_excel(path_transcription_new, index=False, engine='openpyxl')

        print(f"    Number of images in project: {path_transcription}")
        print(f"        {df_pre.shape[0]}")

        path_transcription_fallback_new = path_transcription_new.replace(".xlsx", f"{fallback_suffix}.xlsx")
        df_pre.to_excel(path_transcription_fallback_new, index=False, engine='openpyxl')
    else:
        # Save the original transcription back to its original location
        df_original.to_excel(path_transcription, index=False, engine='openpyxl')
        
        # Save the new transcription to the copied project's transcription file
        df_new.to_excel(path_transcription_new, index=False, engine='openpyxl')

        print(f"    Number of images in original project: {path_transcription}")
        print(f"        {df_pre.shape[0]}")

        path_transcription_fallback = path_transcription.replace(".xlsx", f"{fallback_suffix}.xlsx")
        df_pre.to_excel(path_transcription_fallback, index=False, engine='openpyxl')

        path_transcription_fallback_new = path_transcription_new.replace(".xlsx", f"{fallback_suffix}.xlsx")
        df_pre.to_excel(path_transcription_fallback_new, index=False, engine='openpyxl')

def make_zipfile(base_dir, output_filename):
    # Determine the directory where the zip file should be saved
    # Construct the full path for the zip file
    full_output_path = os.path.join(base_dir, output_filename)
    # Create the zip archive
    shutil.make_archive(full_output_path, 'zip', base_dir)
    # Return the full path of the created zip file
    return os.path.join(base_dir, output_filename + '.zip')

def check_project_validity(path_input_project, version):
    if version == 'single':
        path_transcription = os.path.join(path_input_project, 'Transcription', 'transcribed.xlsx')
        return os.path.isfile(path_transcription)
    elif version == 'already_processed_once':
        is_single = check_project_validity(path_input_project, 'single')
        is_batch = check_project_validity(path_input_project, 'batch')

        if is_single:
            transcription_folder = os.path.join(path_input_project, 'Transcription')
            # Look for any file that matches 'transcription_*.xlsx' but is not 'transcription.xlsx'
            transcription_files = glob.glob(os.path.join(transcription_folder, 'transcribed_*.xlsx'))
            # Remove 'transcription.xlsx' from the list if it exists
            transcription_files = [f for f in transcription_files if not f.endswith('transcribed.xlsx')]
            
            if transcription_files:
                print(f"Found additional transcription files: {transcription_files}")
                return True
            else:
                return False

        elif is_batch:
            try:
                for project in os.listdir(path_input_project):
                    transcription_folder = os.path.join(path_input_project, project, 'Transcription')
                    # Look for any file that matches 'transcription_*.xlsx' but is not 'transcription.xlsx'
                    transcription_files = glob.glob(os.path.join(transcription_folder, 'transcribed_*.xlsx'))
                    # Remove 'transcription.xlsx' from the list if it exists
                    transcription_files = [f for f in transcription_files if not f.endswith('transcribed.xlsx')]
                    
                    if transcription_files:
                        print(f"Found additional transcription files: {transcription_files}")
                        return True
                    else:
                        return False
            except:
                return False
        else:
            raise ValueError(f"The input project path {path_input_project} does not contain the expected file structure.")
 
                
    else:
        try:
            for project in os.listdir(path_input_project):
                path_input_project_sub = os.path.join(path_input_project, project)
                is_valid_sub = check_project_validity(path_input_project_sub, 'single')
                return is_valid_sub
        except:
            return False
        
def start_splitting(path_input_project, path_continents_exclude, path_countries_keep, suffix_new_project, is_already_processed=False, fallback_suffix="_prior_to_subsetting"):

    # If the project has been processed before, we need to use the '_prior_to_subsetting' version of transcribed.xlsx
    if is_already_processed:
        transcription_folder = os.path.join(path_input_project, 'Transcription')
        # Look for any file that matches 'transcription_*.xlsx' but is not 'transcription.xlsx'
        transcription_files = glob.glob(os.path.join(transcription_folder, 'transcribed_prior_to_subsetting.xlsx'))
        # Remove 'transcription.xlsx' from the list if it exists
        transcription_file = [f for f in transcription_files if not f.endswith('transcribed.xlsx')][0]

        # Step 1: Copy the project folder
        new_project_path, did_create_new_project = copy_project_folder(path_input_project, suffix_new_project)

        if did_create_new_project:
            delete_old_zip_files(new_project_path)

            base_name_original_project = os.path.basename(os.path.normpath(path_input_project))
            base_name_new_project = os.path.basename(os.path.normpath(new_project_path))

            # Step 2: Load the keep files
            countries_keep, continents_exclude = load_keep_files(path_countries_keep, path_continents_exclude)
            
            # Step 3: Split the transcription file into two parts
            path_transcription = os.path.join(path_input_project, 'Transcription', transcription_file)
            path_transcription_new = os.path.join(new_project_path, 'Transcription', 'transcribed.xlsx')
            df_pre, df_original, df_new = split_transcription_file(path_transcription, countries_keep, continents_exclude)
            
            # Step 4: Save the two versions of the transcription file
            save_transcription_files(path_transcription, path_transcription_new, df_original, df_new, df_pre, fallback_suffix, is_already_processed)

            zip_filepath_new = make_zipfile(new_project_path, base_name_new_project)

            print(f"    Number of images moved to new location: {path_transcription_new}")
            print(f"        {df_original.shape[0]}")
            print(f"    Number of images skipped due to criteria: {path_transcription}")
            print(f"        {df_new.shape[0]}")


    else:
        # Step 1: Copy the project folder
        new_project_path, did_create_new_project = copy_project_folder(path_input_project, suffix_new_project)

        if did_create_new_project:
            delete_old_zip_files(path_input_project)
            delete_old_zip_files(new_project_path)

            base_name_original_project = os.path.basename(os.path.normpath(path_input_project))
            base_name_new_project = os.path.basename(os.path.normpath(new_project_path))

            # Step 2: Load the keep files
            countries_keep, continents_exclude = load_keep_files(path_countries_keep, path_continents_exclude)
            
            # Step 3: Split the transcription file into two parts
            path_transcription = os.path.join(path_input_project, 'Transcription', 'transcribed.xlsx')
            path_transcription_new = os.path.join(new_project_path, 'Transcription', 'transcribed.xlsx')
            df_pre, df_original, df_new = split_transcription_file(path_transcription, countries_keep, continents_exclude)
            
            # Step 4: Save the two versions of the transcription file
            save_transcription_files(path_transcription, path_transcription_new, df_original, df_new, df_pre, fallback_suffix=fallback_suffix)

            zip_filepath = make_zipfile(path_input_project, base_name_original_project)
            zip_filepath_new = make_zipfile(new_project_path, base_name_new_project)

            print(f"    Number of images kept in original location: {path_transcription}")
            print(f"        {df_original.shape[0]}")
            print(f"    Number of images moved to new location: {path_transcription_new}")
            print(f"        {df_new.shape[0]}")

if __name__ == "__main__":




    # Determine if we are going to be working on a single project, or a folder containing multiple projects

    is_single_project = check_project_validity(path_input_project, 'single')
    is_batch_project = check_project_validity(path_input_project, 'batch')

    if not is_single_project and not is_batch_project:
        print(f"    The provided path_input_project [{path_input_project}] is not a valid VoucherVision project.")
        print(f"    You must provide the path to a single VoucherVision project's output like this:")
        print(f"            C:/Users/willwe/Downloads/2023_10_09_I5_bamaral_AllAsia_Onagr")
        print(f"        OR to a folder that contains multiple VoucherVision projects like this:")
        print(f"            S:/VoucherVision/Unassigned         where Unassigned contains 2023_11_13_I4_kathia_AllAsia_Verben, 2023_10_09_I5_bamaral_AllAsia_Onagr, etc...")
    
    elif is_single_project and not is_batch_project:
        is_already_processed = check_project_validity(path_input_project, 'already_processed_once')
        start_splitting(path_input_project, path_continents_exclude, path_countries_keep, suffix_new_project, is_already_processed)
    
    elif not is_single_project and is_batch_project:
        len_dir = len(os.listdir(path_input_project))
        for i, project in enumerate(os.listdir(path_input_project)):
            print(f"\nWorking on {i+1} / {len_dir}")

            # Check if the directory ends with the suffix and skip it
            if project.endswith(suffix_new_project):
                print(f"    Skipping {project} as it already ends with the suffix '{suffix_new_project}'")
                continue

            if any(suffix in project for suffix in suffix_to_skip):
                print(f"    Skipping {project} as it contains a skipped suffix from the list [{suffix_to_skip}]")
                continue
            
            is_already_processed = check_project_validity(path_input_project, 'already_processed_once')

            path_input_project_sub = os.path.join(path_input_project, project)
            start_splitting(path_input_project_sub, path_continents_exclude, path_countries_keep, suffix_new_project, is_already_processed)

    else:
        print(f"    Plese double check your provided path. The automated test showed that [{path_input_project}] is a single project AND a batch of projects, which is not acceptable.\n")
        print(f"    The provided path_input_project [{path_input_project}] is not a valid VoucherVision project.")
        print(f"    You must provide the path to a single VoucherVision project's output like this:")
        print(f"            C:/Users/willwe/Downloads/2023_10_09_I5_bamaral_AllAsia_Onagr")
        print(f"        OR to a folder that contains multiple VoucherVision projects like this:")
        print(f"            S:/VoucherVision/Unassigned         where Unassigned contains 2023_11_13_I4_kathia_AllAsia_Verben, 2023_10_09_I5_bamaral_AllAsia_Onagr, etc...")
    print("\nFinished")
