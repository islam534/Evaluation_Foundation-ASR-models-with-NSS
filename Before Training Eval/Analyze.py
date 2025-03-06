import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Create directories to store reports and plots
REPORT_DIR = "analysis_reports"
PLOT_DIR = os.path.join(REPORT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def generate_plots(df, model_name, file_identifier):
    """
    Generates and saves plots:
      - Correlation heatmap for numeric columns
      - Histograms for each numeric column
    Returns a list of plot file paths.
    """
    plot_files = []
    
    # Only numeric columns for correlation and histograms
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Correlation heatmap
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        heatmap_file = os.path.join(PLOT_DIR, f"{model_name}_{file_identifier}_correlation_heatmap.png")
        plt.title(f"{model_name} Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(heatmap_file)
        plt.close()
        plot_files.append(heatmap_file)
        
        # Histograms for each numeric column
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), kde=True, bins=20)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            hist_file = os.path.join(PLOT_DIR, f"{model_name}_{file_identifier}_{col}_histogram.png")
            plt.tight_layout()
            plt.savefig(hist_file)
            plt.close()
            plot_files.append(hist_file)
    
    return plot_files

def analyze_file(file_path, model_name):
    """
    Reads and analyzes a CSV file.
    Returns a detailed text report as a string.
    Also generates and saves plots.
    """
    report_lines = []
    report_lines.append(f"Model: {model_name}")
    report_lines.append(f"File: {file_path}")
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    
    try:
        # Read CSV
        df = pd.read_csv(file_path)
        report_lines.append(f"\nOverall Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    except Exception as e:
        report_lines.append(f"Error reading file: {e}")
        return "\n".join(report_lines)
    
    # Basic Info: Column types
    report_lines.append("Column Data Types:")
    report_lines.append(df.dtypes.to_string())
    report_lines.append("\n" + "-" * 80)
    
    # Missing values and unique counts
    report_lines.append("Missing Values and Unique Counts per Column:")
    missing_counts = df.isnull().sum()
    unique_counts = df.nunique()
    for col in df.columns:
        report_lines.append(f"Column '{col}': Missing = {missing_counts[col]}, Unique = {unique_counts[col]}")
    report_lines.append("\n" + "-" * 80)
    
    # Descriptive statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        report_lines.append("Descriptive Statistics (Numeric Columns):")
        report_lines.append(df[numeric_cols].describe().to_string())
    else:
        report_lines.append("No numeric columns to describe.")
    report_lines.append("\n" + "-" * 80)
    
    # Frequency counts for categorical columns (if any)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        report_lines.append("Top 5 Frequent Values for Categorical Columns:")
        for col in categorical_cols:
            report_lines.append(f"\nColumn '{col}':")
            report_lines.append(df[col].value_counts().head(5).to_string())
    else:
        report_lines.append("No categorical columns found.")
    report_lines.append("\n" + "-" * 80)
    
    # Correlation matrix for numeric columns
    if numeric_cols:
        report_lines.append("Correlation Matrix (Numeric Columns):")
        corr_matrix = df[numeric_cols].corr()
        report_lines.append(corr_matrix.to_string())
    else:
        report_lines.append("No numeric columns available for correlation analysis.")
    report_lines.append("\n" + "=" * 80)
    
    # Generate and save plots
    file_identifier = os.path.basename(file_path).split('.')[0]
    plot_files = generate_plots(df, model_name, file_identifier)
    if plot_files:
        report_lines.append("Generated Plots:")
        for plot_file in plot_files:
            report_lines.append(f" - {plot_file}")
    else:
        report_lines.append("No plots were generated.")
    
    return "\n".join(report_lines)

def main():
    # List of CSV files and their corresponding model names
    file_details = [
        ("C:\\Users\\A-plus store\\OneDrive - MOE Stem Schools\\Desktop\\Work Done\\Before Training Eval\\Conformer_Evaluation_results.csv", "Conformer"),
        ("C:\\Users\\A-plus store\\OneDrive - MOE Stem Schools\\Desktop\\Work Done\\Before Training Eval\\Hubert_Evaluation_results.csv", "Hubert"),
        ("C:\\Users\\A-plus store\\OneDrive - MOE Stem Schools\\Desktop\\Work Done\\Before Training Eval\\wav2vec2_asr_evaluation_results.csv", "wav2vec2"),
        ("C:\\Users\\A-plus store\\OneDrive - MOE Stem Schools\\Desktop\\Work Done\\Before Training Eval\\whisper_asr_evaluation_results.csv", "Whisper")
    ]
    
    # Process each file, generate report and save to a text file
    for file_path, model_name in file_details:
        report = analyze_file(file_path, model_name)
        report_filename = os.path.join(REPORT_DIR, f"{model_name}_analysis_report.txt")
        try:
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report saved to: {report_filename}")
        except Exception as e:
            print(f"Failed to write report for {model_name}: {e}")

if __name__ == "__main__":
    main()
