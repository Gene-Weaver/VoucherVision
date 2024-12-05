import pandas as pd
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
import numpy as np
import seaborn as sns
import re

def fix_and_plot():
    # Load data into a dataframe
    data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Gemini_OCR_parameter_sweep_results_wSpecies.csv"
    data_out = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Gemini_OCR_parameter_sweep_results_wSpecies_fixed.csv"
    plot_out = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Gemini_OCR_parameter_sweep_results_wSpecies_fixed_Plot.png"
    
    df = pd.read_csv(data, usecols=lambda column: column != "Response", encoding="ISO-8859-1")

    # Aggregate by unique combinations of Temperature, Top K, Top P
    grouped = df.groupby(["Temperature", "Top K", "Top P"]).sum().reset_index()

    # Ensure that each group has Success Count + Fail Count equal to 3
    def rebin(row):
        total = row["Success Count"] + row["Fail Count"]
        if total < 3:
            row["Success Count"] += (3 - total)
        elif total > 3:
            excess = total - 3
            if row["Fail Count"] >= excess:
                row["Fail Count"] -= excess
            else:
                row["Success Count"] -= (excess - row["Fail Count"])
                row["Fail Count"] = 0
        return row

    # Apply rebinding logic
    result = grouped.apply(rebin, axis=1)
    result.to_csv(data_out, index=False)

    # Calculate success rates
    result['Total'] = result['Success Count'] + result['Fail Count']
    result['Success Rate (%)'] = (result['Success Count'] / result['Total']) * 100

    # Plot success rates for each combination
    plt.figure(figsize=(12, 6))
    combinations = result[['Temperature', 'Top K', 'Top P']].astype(str).agg(', '.join, axis=1)
    plt.bar(combinations, result['Success Rate (%)'], color='skyblue')

    # Customize the plot
    plt.title("Success Rate by Temperature, Top K, and Top P", fontsize=14)
    plt.xlabel("Combination (Temperature, Top K, Top P)", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(plot_out)  # Save the plot as a PNG file
    plt.show()

    print(f"Plot saved to: {plot_out}")



def calc_L_scores():
    # Load data into a dataframe
    data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Gemini_OCR_parameter_sweep_results_wSpecies.csv"
    plot_out = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Gemini_OCR_parameter_sweep_results_wSpecies_fixed_Plot_L.png"
    
    df = pd.read_csv(data, encoding="ISO-8859-1")

    # Sort by Temperature, Top K, and Top P
    df = df.sort_values(by=["Temperature", "Top K", "Top P"]).reset_index(drop=True)

    # Calculate pairwise Levenshtein distances for the "Response" column
    responses = df["Response"].astype(str)
    n = len(responses)
    distances = np.zeros((n, n))

    # Fill the distance matrix
    for i in range(n):
        for j in range(i, n):
            dist = levenshtein_distance(responses[i], responses[j])
            distances[i, j] = dist
            distances[j, i] = dist

    # Calculate average distance for each response
    avg_distances = distances.mean(axis=1)
    df["Avg Levenshtein Distance"] = avg_distances

    # Identify the response with the most agreement and most dissimilarity
    most_agreement_index = avg_distances.argmin()
    most_dissimilar_index = avg_distances.argmax()

    most_agreement = responses[most_agreement_index]
    most_dissimilar = responses[most_dissimilar_index]

    print(f"Response with most agreement: {most_agreement}")
    print(f"Response with most dissimilarity: {most_dissimilar}")

    # Identify indices where Temperature changes
    temperature_changes = df["Temperature"].ne(df["Temperature"].shift()).cumsum()
    temp_boundaries = df.groupby(temperature_changes).size().cumsum().to_list()

    # Plot the confusion matrix for Levenshtein distances
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        distances,
        cmap="viridis",
        annot=False,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Levenshtein Distance"}
    )
    
    # Add white lines for temperature changes
    for boundary in temp_boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.5)
        ax.axvline(boundary - 0.5, color="white", linewidth=1.5)

    # Customize the plot
    plt.title("Confusion Matrix of Levenshtein Distances", fontsize=14)
    plt.xlabel("Responses", fontsize=12)
    plt.ylabel("Responses", fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_out)
    plt.show()
    print(f"Levenshtein distance heatmap saved to: {plot_out}")

def calc_L_scores_sorted(path_data, path_L_plot, plot_title, do_print=True, label_size=4, compare_ignoring_newlines_and_case=True):
    # Load data into a dataframe    
    df = pd.read_csv(path_data, encoding="ISO-8859-1")

    # Calculate pairwise Levenshtein distances for the "Response" column
    if compare_ignoring_newlines_and_case:
        responses = (
            df["Response"]
            .astype(str)
            .str.replace('\n', ' ')        # Replace newlines with spaces
            .apply(lambda x: re.sub(r'\s+', ' ', x))  # Normalize spaces to a single space
            .str.strip()                   # Remove leading/trailing spaces
            .str.lower()                   # Convert to lowercase
        )
        # responses2 = df["Response"].astype(str)
    else:
        responses = df["Response"].astype(str)
    n = len(responses)
    distances = np.zeros((n, n))

    # Fill the distance matrix
    for i in range(n):
        for j in range(i, n):
            dist = levenshtein_distance(responses[i], responses[j])
            distances[i, j] = dist
            distances[j, i] = dist

    # Calculate average distance for each response
    avg_distances = distances.mean(axis=1)
    df["Avg Levenshtein Distance"] = avg_distances

    # Identify the response with the most agreement and most dissimilarity
    most_agreement_index = avg_distances.argmin()
    most_dissimilar_index = avg_distances.argmax()

    if do_print:
        print(f"Response with most agreement: {responses[most_agreement_index]}")
        print(f"Response with most dissimilarity: {responses[most_dissimilar_index]}")

    # Sort by Avg Levenshtein Distance
    sorted_indices = np.argsort(avg_distances)
    sorted_distances = distances[sorted_indices][:, sorted_indices]
    try:
        sorted_temp_k_p = (
            df.iloc[sorted_indices]
            .apply(lambda row: f"T={row['Temperature']}, K={row['Top K']}, P={row['Top P']}", axis=1)
        )
    except:
        sorted_temp_k_p = (
            df.iloc[sorted_indices]
            .apply(lambda row: f"T={row['Temperature']}, P={row['Top P']}", axis=1)
        )

    # Define label colors based on the new logic
    label_colors = []
    for i in sorted_indices:
        if i == most_agreement_index:
            label_colors.append("green")  # Most similar
        elif i == most_dissimilar_index:
            label_colors.append("cyan")  # Most dissimilar
        elif df.iloc[i]["Fail Count"] > 0:
            label_colors.append("red")  # Fail Count > 0
        else:
            label_colors.append("black")  # Default

    # Plot the confusion matrix for sorted Levenshtein distances
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(
        sorted_distances,
        cmap="viridis",
        annot=False,
        xticklabels=sorted_temp_k_p,
        yticklabels=sorted_temp_k_p,
        cbar_kws={"label": "Levenshtein Distance"}
    )

    # Apply label colors to tick labels
    for tick_label, color in zip(ax.get_xticklabels(), label_colors):
        tick_label.set_color(color)
        tick_label.set_fontsize(label_size)
        tick_label.set_rotation(90)

    for tick_label, color in zip(ax.get_yticklabels(), label_colors):
        tick_label.set_color(color)
        tick_label.set_fontsize(label_size)

    # Customize the plot
    if compare_ignoring_newlines_and_case:
        plot_title = plot_title + " ignoring newlines and case"
        path_L_plot = path_L_plot.replace(".png", "_ignoring_newlines_and_case.png")
    plt.title(plot_title, fontsize=14)
    plt.xlabel("Temperature, Top K, Top P (Sorted)", fontsize=12)
    plt.ylabel("Temperature, Top K, Top P (Sorted)", fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig(path_L_plot, dpi=600)
    # plt.show()
    if do_print:
        print(f"Levenshtein distance heatmap (sorted by L-score) saved to: {path_L_plot}")


if __name__ == "__main__":
    # fix_and_plot()
    # calc_L_scores()

    ### TODO use the setdiff implementation to add the cyan and green json to the plots as a right andcenter panel
    
    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_72B_Instruct_parameter_sweep_results_QwenVersion_wSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_72B_Instruct_parameter_sweep_results_QwenVersion_wSpecies_fixed_Plot_L_sorted.png",
                        plot_title="Qwen2-VL-72B-Instruct OCR QwenVersion wSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10, compare_ignoring_newlines_and_case=True)

    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_72B_Instruct_parameter_sweep_results_QwenVersion_woSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_72B_Instruct_parameter_sweep_results_QwenVersion_woSpecies_fixed_Plot_L_sorted.png",
                        plot_title="Qwen2-VL-72B-Instruct OCR QwenVersion woSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)
    
    # BEST of Qwen
    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_7B_Instruct_parameter_sweep_results_QwenVersion_wSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_7B_Instruct_parameter_sweep_results_QwenVersion_wSpecies_fixed_Plot_L_sorted.png",
                        plot_title="Qwen2-VL-7B-Instruct OCR QwenVersion wSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)

    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_7B_Instruct_parameter_sweep_results_QwenVersion_woSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_7B_Instruct_parameter_sweep_results_QwenVersion_woSpecies_fixed_Plot_L_sorted.png",
                        plot_title="Qwen2-VL-7B-Instruct OCR QwenVersion woSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)
    
    # BEST of Qwen T=1, P=0.8 has jouvea in main body, extra species are wrong | T=0.5,P=0.8 is reverse
    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_7B_Instruct_parameter_sweep_results_wSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_7B_Instruct_parameter_sweep_results_wSpecies_fixed_Plot_L_sorted.png",
                        plot_title="Qwen2-VL-7B-Instruct OCR wSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)
    
    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_7B_Instruct_parameter_sweep_results_woSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_7B_Instruct_parameter_sweep_results_woSpecies_fixed_Plot_L_sorted.png",
                        plot_title="Qwen2-VL-7B-Instruct OCR woSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)
    
    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_72B_Instruct_parameter_sweep_results_wSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_72B_Instruct_parameter_sweep_results_wSpecies_fixed_Plot_L_sorted.png",
                        plot_title="Qwen2-VL-72B-Instruct OCR wSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)
    
    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_72B_Instruct_parameter_sweep_results_woSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Qwen2_VL_72B_Instruct_parameter_sweep_results_woSpecies_fixed_Plot_L_sorted.png",
                        plot_title="Qwen2-VL-72B-Instruct OCR woSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)
    
    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/GPT4o_OCR_parameter_sweep_results_GPT4version.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/GPT4o_OCR_parameter_sweep_results_GPT4version_fixed_Plot_L_sorted.png",
                        plot_title="GPT-4o OCR GPT4version - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)

    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/GPT4o_OCR_parameter_sweep_results_GPT4version_wSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/GPT4o_OCR_parameter_sweep_results_GPT4version_wSpecies_fixed_Plot_L_sorted.png",
                        plot_title="GPT-4o OCR GPT4version wSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)

    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/GPT4o_OCR_parameter_sweep_results_notGPT4version.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/GPT4o_OCR_parameter_sweep_results_notGPT4version_fixed_Plot_L_sorted.png",
                        plot_title="GPT-4o OCR notGPT4version - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)

    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/GPT4o_OCR_parameter_sweep_results_notGPT4version_wSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/GPT4o_OCR_parameter_sweep_results_notGPT4version_wSpecies_fixed_Plot_L_sorted.png",
                        plot_title="GPT-4o OCR notGPT4version wSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=10)
    
    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Gemini_OCR_parameter_sweep_results_wSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Gemini_OCR_parameter_sweep_results_wSpecies_fixed_Plot_L_sorted.png",
                        plot_title="Gemini-1.5-Pro OCR wSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=4)
    
    calc_L_scores_sorted(path_data = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Gemini_OCR_parameter_sweep_results_woSpecies.csv",
                        path_L_plot = "C:/Users/willwe/Documents/VoucherVision/OCR_vLM_Parameter_Sweep/Gemini_OCR_parameter_sweep_results_woSpecies_fixed_Plot_L_sorted.png",
                        plot_title="Gemini-1.5-Pro OCR woSpecies - Confusion Matrix of Levenshtein Distances (Sorted by L-Distance)",
                        label_size=4)
