### Provide the following:
###     Input:
###         path_input_project
###             path to project that you want to split
###             e.g. "S:/VoucherVision/Unassigned_AA/2023_11_01_I4_mikedmac_AllAsia_Ole"  # a single project
###             e.g. "S:/VoucherVision/Unassigned_AA"                                    # a folder that contains multiple projects
# path_input_project = "S:/VoucherVision/Unassigned_AA"
path_input_project = "S:/VoucherVision/Unassigned_AA"

###         suffix_to_skip (should not change unless the scope of the project changes)
###             suffix(s) that we can ignore. for example, if we split the VoucherVision output to create an _exAA set, then to find 
###             specimens from Africa we don't want to look inside those folders, we want to look in the original folders. Add the suffix to the list.
###             e.g. ["_exAA",]
suffix_to_skip = ["_exAA",]

import os, glob
import pandas as pd

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

def concatenate_transcribed_prior_to_subsetting(path_input_project):
    """
    Finds all files named transcribed_prior_to_subsetting.xlsx in non-skipped folders and 
    writes their contents directly to a single output file transcribed_prior_to_subsetting_ALL.xlsx.

    Parameters:
        path_input_project (str): Path to the project or folder containing multiple projects.
    """
    output_file = os.path.join(os.path.dirname(path_input_project), "transcribed_prior_to_subsetting_ALL.xlsx")
    # Attempt to locate the transcription file
    transcription_file_prior = os.path.join(path_input_project, 'Transcription', "transcribed_prior_to_subsetting.xlsx")
    transcription_file = os.path.join(path_input_project, 'Transcription', "transcribed.xlsx")
    
    if os.path.isfile(transcription_file_prior):
        transcription_file = transcription_file_prior
    elif not os.path.isfile(transcription_file):
        raise FileNotFoundError(
            f"Neither 'transcribed_prior_to_subsetting.xlsx' nor 'transcribed.xlsx' were found in the expected directory: {os.path.join(path_input_project, 'Transcription')}."
        )
    
    if os.path.isfile(transcription_file):
        try:
            df = pd.read_excel(transcription_file)

            if os.path.exists(output_file):
                # Load existing file and append to it
                with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                    # Append data to the same sheet
                    df.to_excel(writer, index=False, header=False, startrow=writer.sheets["Transcription"].max_row, sheet_name="Transcription")
            else:
                # Create a new file and write the data
                with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Transcription")
            
            print(f"Added {transcription_file} to the output file.")
        except Exception as e:
            print(f"Error reading file {transcription_file}: {e}")


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
        # concatenate_transcribed_prior_to_subsetting(path_input_project, suffix_to_skip)
        # already concatenated.... bc it's just a file
    elif not is_single_project and is_batch_project:
        len_dir = len(os.listdir(path_input_project))
        for i, project in enumerate(os.listdir(path_input_project)):
            print(f"\nWorking on {i+1} / {len_dir}")

            if any(suffix in project for suffix in suffix_to_skip):
                print(f"    Skipping {project} as it contains a skipped suffix from the list [{suffix_to_skip}]")
                continue
            
            is_already_processed = check_project_validity(path_input_project, 'already_processed_once')

            path_input_project_sub = os.path.join(path_input_project, project)
            concatenate_transcribed_prior_to_subsetting(path_input_project_sub)

    else:
        print(f"    Plese double check your provided path. The automated test showed that [{path_input_project}] is a single project AND a batch of projects, which is not acceptable.\n")
        print(f"    The provided path_input_project [{path_input_project}] is not a valid VoucherVision project.")
        print(f"    You must provide the path to a single VoucherVision project's output like this:")
        print(f"            C:/Users/willwe/Downloads/2023_10_09_I5_bamaral_AllAsia_Onagr")
        print(f"        OR to a folder that contains multiple VoucherVision projects like this:")
        print(f"            S:/VoucherVision/Unassigned         where Unassigned contains 2023_11_13_I4_kathia_AllAsia_Verben, 2023_10_09_I5_bamaral_AllAsia_Onagr, etc...")
    print("\nFinished")