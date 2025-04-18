###                                 Instructions
### To run this file: 
###      1. Edit the paths below
###      2. In the terminal, change directories and move into the directory that contains this file, using 'cd'
###      3. Two run the code, we need 2 packages. You can either activate the VoucherVision virtual environment (https://github.com/Gene-Weaver/VoucherVision)
###         OR you can just run `pip install pandas openpyxl` to install them locally (these are basic packages that should not cause any problems)
###      4. Type `python split_finished_batch_by_geography.py`
###      5. The code will run
### 
### This code is designed to split the VoucherVision output based on geopgraphy, usually to split out the exAA specimens.
### 
### This code will 
### 1. Make a full copy of the original project and all its files, saving it to the same location with the new suffix. 
###    If the project with suffix already exists, skip it, do nothing else
###     e.g. ORIGINAL S:/VoucherVision/Unassigned/2023_10_13_I4_kathia_AllAsia_Api
###          NEW      S:/VoucherVision/Unassigned/2023_10_13_I4_kathia_AllAsia_Api_exAA
###     where the _exAA is the suffix you provide
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
###            If there are African specimens, they will be moved to the _exAA project's transcribed.xlsx
### 6. Save the two transcribed.xlsx versions, overwriting the original transcribed.xlsx version in both folders.
### 7 Create a new .zip file in both ORIGINAL and NEW, which will be used in the VVEditor
### 
### NOTE: In the event of an error, you will see a folder with the suffix "__BACKUP". This retains a copy of the full original project.
###       If the code has finished running and you see a "__BACKUP" folder, then the code did not actually run successfully. The most
###       likely error is that the S Drive lost connection for a bit or there is a typo in a file path name.
###       TO RESOLVE THIS ERROR:
###             Delete the folder that you were tring to split and delete the folder that you were trying to create, then rename the "__BACKUP"
###             folder by simply delete the "__BACKUP" from the file name. Then rerun the code.
###             e.g. your project is called "2023_10_13_I4_kathia_AllAsia_Api", you run the code and it starts to create "2023_10_13_I4_kathia_AllAsia_Api_exAA"
###                  the code finishes/stops running, but when you look in the output folder you still see "2023_10_13_I4_kathia_AllAsia_Api__BACKUP". 
###                  Double check that the file paths you provided are all correct. Then delete "2023_10_13_I4_kathia_AllAsia_Api" and "2023_10_13_I4_kathia_AllAsia_Api_exAA"
###                  Now rename "2023_10_13_I4_kathia_AllAsia_Api__BACKUP" to "2023_10_13_I4_kathia_AllAsia_Api" and rerun the code.
### 
### Provide the following:
###     Input:
###         path_input_project
###             NOTE: This path can either be to a single project, or a folder that contains many projects
###             path to project that you want to split
###             e.g. "S:/VoucherVision/Unassigned_AA/2023_11_01_I4_mikedmac_AllAsia_Ole"  # a single project
###             e.g. "S:/VoucherVision/Unassigned_AA"                                     # a folder that contains multiple projects
path_input_project = "S:/VoucherVision/Unassigned_AA"

###         suffix_new_project (should not change unless the scope of the project changes)
###             suffix for new project (original_project_name{suffix_new_project})
###             e.g. "_exAA"
suffix_new_project = "_exAA"

###         path_countries_keep (should not change unless the scope of the project changes)
###             path to countries_keep.csv
###             e.g. "S:/Curatorial Projects/VoucherVision/Tools/Country_Include_AA.csv"
path_countries_keep = "S:/VoucherVision/Tools/Country_Include_AA.csv"

###         path_continents_exclude (should not change unless the scope of the project changes)
###             path to continents_exclude.csv
###             e.g. "S:/Curatorial Projects/VoucherVision/Tools/Continent_Exclude_AA.csv"
path_continents_exclude = "S:/VoucherVision/Tools/Continent_Exclude_AA.csv"






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

def delete_directory(directory):
    """Recursively deletes the specified directory and all its contents."""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"    Successfully deleted __BACKUP directory: {directory}")
        else:
            print(f"Directory does not exist: {directory}")
    except Exception as e:
        print(f"Failed to delete directory {directory}: {e}")

def load_keep_files(path_countries_keep, path_continents_keep):
    """Loads the country and continent keep files."""
    countries_keep = set(pd.read_csv(path_countries_keep, header=None).iloc[:, 0].dropna().str.strip().tolist())
    continents_keep = set(pd.read_csv(path_continents_keep, header=None).iloc[:, 0].dropna().str.strip().tolist())
    return countries_keep, continents_keep

# def split_transcription_file(path_transcription, countries_keep, continents_exclude):
#     """Splits the transcription file into two parts based on country and continent filters."""
#     # Read the transcription file
#     df = pd.read_excel(path_transcription, engine='openpyxl')
    
#     # Ensure 'country' and 'continent' columns exist
#     if 'country' not in df.columns or 'continent' not in df.columns:
#         raise ValueError("The transcription file must contain 'country' and 'continent' columns.")
    
#     # Filter for the original transcription (keep these rows)
#     df_original = df[
#         df['country'].str.lower().isin({country.lower() for country in countries_keep}) | 
#         ~df['continent'].str.lower().isin({continent.lower() for continent in continents_exclude}) | 
#         (df['country'].isna() & df['continent'].isna())
#     ]
    
#     # Filter for the new project transcription (all other rows)
#     df_new = df[~df.index.isin(df_original.index)]
    
#     return df, df_original, df_new
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
        print(f"    Copying files from {path_input_project} --> {suffix_new_project}")
        new_project_path, did_create_new_project = copy_project_folder(path_input_project, suffix_new_project)
        print(f"    Copying files from {path_input_project} --> __BACKUP")
        backup_project_path, did_create_backup_project = copy_project_folder(path_input_project, "__BACKUP")

        if did_create_new_project and did_create_backup_project:
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
            save_transcription_files(path_transcription, path_transcription_new, df_original, df_new, df_pre, is_already_processed, fallback_suffix=fallback_suffix)

            zip_filepath_new = make_zipfile(new_project_path, base_name_new_project)

            print(f"    Number of images moved to new location: {path_transcription_new}")
            print(f"        {df_original.shape[0]}")
            print(f"    Number of images skipped due to criteria: {path_transcription}")
            print(f"        {df_new.shape[0]}")

            delete_directory(backup_project_path)
        else:
            print(f"Skipping {path_input_project}")
            print(f"    Created {suffix_new_project} folder version of project: {did_create_new_project}")
            print(f"        {new_project_path}")
            print(f"    Created __BACKUP folder version of project: {did_create_backup_project}")
            print(f"        {backup_project_path}")


    else:
        # Step 1: Copy the project folder
        print(f"    Copying files from {path_input_project} --> {suffix_new_project}")
        new_project_path, did_create_new_project = copy_project_folder(path_input_project, suffix_new_project)
        print(f"    Copying files from {path_input_project} --> __BACKUP")
        backup_project_path, did_create_backup_project = copy_project_folder(path_input_project, "__BACKUP")

        if did_create_new_project and did_create_backup_project:
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

            delete_directory(backup_project_path)

        else:
            print(f"Skipping {path_input_project}")
            print(f"    Created {suffix_new_project} folder version of project: {did_create_new_project}")
            print(f"        {new_project_path}")
            print(f"    Created __BACKUP folder version of project: {did_create_backup_project}")
            print(f"        {backup_project_path}")


if __name__ == "__main__":
    """
    These commented out things can be used for debugging but you should use the global 
    variables at the top of the script for normal use
    """
    # Define input paths and suffix
    # path_input_project = "/path/to/input/project"
    # path_countries_keep = "/path/to/countries_keep.csv"
    # path_continents_keep = "/path/to/continents_keep.csv"
    # suffix_new_project = "_new"
    
    # path_input_project = "C:/Users/willwe/Downloads/2023_10_09_I5_bamaral_AllAsia_Onagr"
    # path_input_project = "S:/VoucherVision/Unassigned"
    # path_input_project = "C:/Users/willwe/Downloads/two_batches"

    # path_continents_exclude = "S:/VoucherVision/Tools/Continent_Exclude_AA.csv"
    # path_countries_keep = "S:/VoucherVision/Tools/Country_Include_AA.csv"
    # suffix_new_project = "_exAA"

    # path_input_project = "S:/VoucherVision/Unassigned_AA"
    # path_continents_exclude = "S:/VoucherVision/Tools/Continent_Exclude_Africa.csv"
    # path_countries_keep = "S:/VoucherVision/Tools/Country_Include_Africa.csv"
    # suffix_new_project = "_Africa"

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
