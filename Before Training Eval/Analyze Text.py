import os
import re
import difflib
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud

# Uncomment the next two lines if you plan to generate word clouds
# from wordcloud import WordCloud
# import matplotlib.colors as mcolors

# Create directories to store reports and plots
REPORT_DIR = "text_analysis_reports"
PLOT_DIR = os.path.join(REPORT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def preprocess_text(text):
    """
    Normalize text by lowering case and removing punctuation.
    """
    if pd.isna(text):
        return ""
    # Lowercase the text
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def jaccard_similarity(text1, text2):
    """
    Computes the Jaccard similarity between two texts based on word tokens.
    """
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())
    if not tokens1 and not tokens2:
        return 1.0  # Both empty
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

def compute_text_metrics(actual, predicted):
    """
    Given an actual and predicted string, compute several text-based similarity metrics.
    Returns a dictionary with:
      - sequence_ratio: difflib SequenceMatcher similarity ratio
      - jaccard_sim: Jaccard similarity of token sets
      - exact_match: 1 if texts match exactly, else 0
      - word_count_actual, word_count_predicted, word_count_diff
    """
    actual_clean = preprocess_text(actual)
    predicted_clean = preprocess_text(predicted)
    
    # Use difflib to compute a similarity ratio
    seq_ratio = difflib.SequenceMatcher(None, actual_clean, predicted_clean).ratio()
    
    # Compute Jaccard similarity based on word tokens
    jaccard_sim = jaccard_similarity(actual_clean, predicted_clean)
    
    # Exact match check
    exact_match = 1 if actual_clean == predicted_clean else 0
    
    # Word counts and difference
    wc_actual = len(actual_clean.split())
    wc_predicted = len(predicted_clean.split())
    wc_diff = abs(wc_actual - wc_predicted)
    
    return {
        "sequence_ratio": seq_ratio,
        "jaccard_sim": jaccard_sim,
        "exact_match": exact_match,
        "word_count_actual": wc_actual,
        "word_count_predicted": wc_predicted,
        "word_count_diff": wc_diff
    }

def add_text_metrics(df, actual_col='Actual', predicted_col='Predicted'):
    """
    Apply compute_text_metrics for each row and add the results as new columns.
    """
    metrics = df.apply(lambda row: compute_text_metrics(row[actual_col], row[predicted_col]), axis=1)
    metrics_df = pd.DataFrame(metrics.tolist())
    df = pd.concat([df, metrics_df], axis=1)
    return df

def generate_text_plots(df, model_name, file_identifier):
    """
    Generate and save various plots based on text similarity metrics.
    Returns a list of generated plot file paths.
    """
    plot_files = []
    
    # Histogram of Sequence Similarity Ratio
    plt.figure(figsize=(8, 6))
    sns.histplot(df['sequence_ratio'], bins=30, kde=True)
    plt.title(f"{model_name}: Distribution of Sequence Similarity Ratio")
    plt.xlabel("Sequence Similarity Ratio")
    plt.ylabel("Frequency")
    seq_plot_file = os.path.join(PLOT_DIR, f"{model_name}_{file_identifier}_sequence_ratio_hist.png")
    plt.tight_layout()
    plt.savefig(seq_plot_file)
    plt.close()
    plot_files.append(seq_plot_file)
    
    # Histogram of Jaccard Similarity
    plt.figure(figsize=(8, 6))
    sns.histplot(df['jaccard_sim'], bins=30, kde=True, color='orange')
    plt.title(f"{model_name}: Distribution of Jaccard Similarity")
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Frequency")
    jaccard_plot_file = os.path.join(PLOT_DIR, f"{model_name}_{file_identifier}_jaccard_similarity_hist.png")
    plt.tight_layout()
    plt.savefig(jaccard_plot_file)
    plt.close()
    plot_files.append(jaccard_plot_file)
    
    # Scatter plot: Word count difference vs. Sequence Similarity
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='word_count_diff', y='sequence_ratio', data=df, alpha=0.5)
    plt.title(f"{model_name}: Word Count Difference vs. Sequence Similarity")
    plt.xlabel("Absolute Difference in Word Count")
    plt.ylabel("Sequence Similarity Ratio")
    scatter_plot_file = os.path.join(PLOT_DIR, f"{model_name}_{file_identifier}_wordcount_vs_seqratio.png")
    plt.tight_layout()
    plt.savefig(scatter_plot_file)
    plt.close()
    plot_files.append(scatter_plot_file)
    
    # Boxplot of similarity metrics
    plt.figure(figsize=(10, 6))
    data_to_plot = df[['sequence_ratio', 'jaccard_sim']]
    data_to_plot = pd.melt(data_to_plot)
    sns.boxplot(x='variable', y='value', data=data_to_plot)
    plt.title(f"{model_name}: Boxplot of Similarity Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    boxplot_file = os.path.join(PLOT_DIR, f"{model_name}_{file_identifier}_similarity_boxplot.png")
    plt.tight_layout()
    plt.savefig(boxplot_file)
    plt.close()
    plot_files.append(boxplot_file)
    
    # Optional: Generate word clouds for Actual and Predicted texts if wordcloud is installed
    # Uncomment the following section if you wish to generate word clouds.
    
    text_actual = ' '.join(df['Actual'].dropna().astype(str).tolist()).lower()
    text_predicted = ' '.join(df['Predicted'].dropna().astype(str).tolist()).lower()
    wc_actual = WordCloud(width=800, height=400, background_color='white').generate(text_actual)
    wc_predicted = WordCloud(width=800, height=400, background_color='white').generate(text_predicted)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_actual, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{model_name}: Word Cloud for Actual Texts")
    wordcloud_actual_file = os.path.join(PLOT_DIR, f"{model_name}_{file_identifier}_actual_wordcloud.png")
    plt.tight_layout()
    plt.savefig(wordcloud_actual_file)
    plt.close()
    plot_files.append(wordcloud_actual_file)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_predicted, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{model_name}: Word Cloud for Predicted Texts")
    wordcloud_predicted_file = os.path.join(PLOT_DIR, f"{model_name}_{file_identifier}_predicted_wordcloud.png")
    plt.tight_layout()
    plt.savefig(wordcloud_predicted_file)
    plt.close()
    plot_files.append(wordcloud_predicted_file)
   
    
    return plot_files

def analyze_text_predictions(file_path, model_name, actual_col='Actual', predicted_col='Predicted'):
    """
    Analyze the text data by comparing the actual and predicted columns.
    Returns a detailed analysis report as a string.
    """
    report_lines = []
    report_lines.append(f"Model: {model_name}")
    report_lines.append(f"File: {file_path}")
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    
    try:
        df = pd.read_csv(file_path)
        report_lines.append(f"\nOverall Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    except Exception as e:
        report_lines.append(f"Error reading file: {e}")
        return "\n".join(report_lines)
    
    # Report basic info for text columns
    for col in [actual_col, predicted_col]:
        report_lines.append(f"Column '{col}' Data Type: {df[col].dtype}")
        report_lines.append(f"Unique values in '{col}': {df[col].nunique()}")
    report_lines.append("-" * 80)
    
    # Compute additional text metrics and add them to the dataframe
    df = add_text_metrics(df, actual_col, predicted_col)
    
    # Descriptive statistics for the new text metrics
    metrics_cols = ['sequence_ratio', 'jaccard_sim', 'exact_match', 
                    'word_count_actual', 'word_count_predicted', 'word_count_diff']
    report_lines.append("Descriptive Statistics for Text Similarity Metrics:")
    report_lines.append(df[metrics_cols].describe().to_string())
    report_lines.append("-" * 80)
    
    # Overall exact match percentage
    exact_match_pct = df['exact_match'].mean() * 100
    report_lines.append(f"Overall Exact Match Percentage: {exact_match_pct:.2f}%")
    
    # Report mean and median similarity ratios
    mean_seq = df['sequence_ratio'].mean()
    median_seq = df['sequence_ratio'].median()
    mean_jaccard = df['jaccard_sim'].mean()
    median_jaccard = df['jaccard_sim'].median()
    report_lines.append(f"Mean Sequence Similarity Ratio: {mean_seq:.4f}")
    report_lines.append(f"Median Sequence Similarity Ratio: {median_seq:.4f}")
    report_lines.append(f"Mean Jaccard Similarity: {mean_jaccard:.4f}")
    report_lines.append(f"Median Jaccard Similarity: {median_jaccard:.4f}")
    report_lines.append("-" * 80)
    
    # Frequency analysis for common words (basic version)
    def get_top_words(text_series, top_n=10):
        all_text = ' '.join(text_series.dropna().astype(str))
        tokens = preprocess_text(all_text).split()
        freq_dist = pd.Series(tokens).value_counts()
        return freq_dist.head(top_n)
    
    report_lines.append("Top 10 Frequent Words in Actual Texts:")
    report_lines.append(get_top_words(df[actual_col]).to_string())
    report_lines.append("-" * 40)
    report_lines.append("Top 10 Frequent Words in Predicted Texts:")
    report_lines.append(get_top_words(df[predicted_col]).to_string())
    report_lines.append("-" * 80)
    
    # Generate and save text-based plots
    file_identifier = os.path.basename(file_path).split('.')[0]
    plot_files = generate_text_plots(df, model_name, file_identifier)
    if plot_files:
        report_lines.append("Generated Plots:")
        for plot_file in plot_files:
            report_lines.append(f" - {plot_file}")
    else:
        report_lines.append("No plots were generated.")
    
    report_lines.append("=" * 80)
    return "\n".join(report_lines)

def main():
    # List of files and corresponding model names (customize these paths as needed)
    file_details = [
        (r"C:\Users\A-plus store\OneDrive - MOE Stem Schools\Desktop\Work Done\Before Training Eval\Conformer_Evaluation_results.csv", "Conformer"),
        (r"C:\Users\A-plus store\OneDrive - MOE Stem Schools\Desktop\Work Done\Before Training Eval\Hubert_Evaluation_results.csv", "Hubert"),
        (r"C:\Users\A-plus store\OneDrive - MOE Stem Schools\Desktop\Work Done\Before Training Eval\wav2vec2_asr_evaluation_results.csv", "wav2vec2"),
        (r"C:\Users\A-plus store\OneDrive - MOE Stem Schools\Desktop\Work Done\Before Training Eval\whisper_asr_evaluation_results.csv", "Whisper")
    ]
    
    # Process each file and generate a detailed text analysis report
    for file_path, model_name in file_details:
        report = analyze_text_predictions(file_path, model_name, actual_col='Actual', predicted_col='Predicted')
        report_filename = os.path.join(REPORT_DIR, f"{model_name}_text_analysis_report.txt")
        try:
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report for {model_name} saved to: {report_filename}")
        except Exception as e:
            print(f"Failed to write report for {model_name}: {e}")

if __name__ == "__main__":
    main()
