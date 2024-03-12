import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def SUMMARY_permute_llms_to_sweep_temperature_and_topP_for_GPT4_SHORT():
    #####################
    # Load the Excel file
    file_path = 'D:/Dropbox/VoucherVision/demo/validation_output/summary/SUMMARY_permute_llms_to_sweep_temperature_and_topP_for_GPT4_SHORT.xlsx'
    save_path = 'D:/Dropbox/VoucherVision/demo/validation_output/figures/avg_L_score_analysis_SUMMARY_permute_llms_to_sweep_temperature_and_topP_for_GPT4_SHORT.png'

    df = pd.read_excel(file_path)

    # Display the first few rows of the dataframe to understand its structure
    df.head()

    # Grouping by the parameters and calculating the mean of avg_L_score for each group
    grouped = df.groupby(['v_prompt_version', 'v_double_ocr', 'temperature', 'top_p'])['avg_L_score'].mean().reset_index()

    # Finding the group with the highest average L score
    max_avg_L_score = grouped['avg_L_score'].max()
    best_group = grouped[grouped['avg_L_score'] == max_avg_L_score]

    print(best_group)


    ### Viz
    # Filtering the dataset for the conditions mentioned
    filtered_df = df[df['v_prompt_version'] == 'SLTPvB_long.yaml'][df['v_double_ocr'] == True]

    # Setting up the plotting
    plt.figure(figsize=(14, 6))

    # Plot 1: avg_L_score as a function of temperature for each top_p value
    plt.subplot(1, 2, 1)
    sns.lineplot(data=filtered_df, x='temperature', y='avg_L_score', hue='top_p', marker='o')
    plt.title('Average L Score by Temperature for each Top P')
    plt.xlabel('Temperature')
    plt.ylabel('Average L Score')
    plt.legend(title='Top P', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: avg_L_score as a function of top_p for each temperature value
    plt.subplot(1, 2, 2)
    sns.lineplot(data=filtered_df, x='top_p', y='avg_L_score', hue='temperature', marker='o')
    plt.title('Average L Score by Top P for each Temperature')
    plt.xlabel('Top P')
    plt.ylabel('Average L Score')
    plt.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)

def SUMMARY_permute_llms_to_sweep_temperature_and_topP_for_google_SHORT():
    #####################
    # Load the Excel file
    file_path = 'D:/Dropbox/VoucherVision/demo/validation_output/summary/SUMMARY_permute_llms_to_sweep_temperature_and_topP_for_google_SHORT.xlsx'
    save_path = 'D:/Dropbox/VoucherVision/demo/validation_output/figures/avg_L_score_analysis_SUMMARY_permute_llms_to_sweep_temperature_and_topP_for_google_SHORT.png'

    df = pd.read_excel(file_path)

    # Display the first few rows of the dataframe to understand its structure
    df.head()

    # Grouping by the parameters and calculating the mean of avg_L_score for each group
    grouped = df.groupby(['v_prompt_version', 'v_double_ocr', 'temperature', 'top_p'])['avg_L_score'].mean().reset_index()

    # Finding the group with the highest average L score
    max_avg_L_score = grouped['avg_L_score'].max()
    best_group = grouped[grouped['avg_L_score'] == max_avg_L_score]

    print(best_group)


    ### Viz
    # Filtering the dataset for the conditions mentioned
    filtered_df = df[df['v_prompt_version'] == 'SLTPvB_long.yaml'][df['v_double_ocr'] == True]

    # Setting up the plotting
    plt.figure(figsize=(14, 6))

    # Plot 1: avg_L_score as a function of temperature for each top_p value
    plt.subplot(1, 2, 1)
    sns.lineplot(data=filtered_df, x='temperature', y='avg_L_score', hue='top_p', marker='o')
    plt.title('Average L Score by Temperature for each Top P')
    plt.xlabel('Temperature')
    plt.ylabel('Average L Score')
    plt.legend(title='Top P', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: avg_L_score as a function of top_p for each temperature value
    plt.subplot(1, 2, 2)
    sns.lineplot(data=filtered_df, x='top_p', y='avg_L_score', hue='temperature', marker='o')
    plt.title('Average L Score by Top P for each Temperature')
    plt.xlabel('Top P')
    plt.ylabel('Average L Score')
    plt.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)

if __name__ == '__main__':
    # SUMMARY_permute_llms_to_sweep_temperature_and_topP_for_GPT4_SHORT()
    SUMMARY_permute_llms_to_sweep_temperature_and_topP_for_google_SHORT()
    

