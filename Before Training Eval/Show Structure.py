import pandas as pd

def analyze_csv_structure(file_paths):
    for file in file_paths:
        print(f"\nAnalyzing: {file}")
        try:
            df = pd.read_csv(file)
            print("\nFirst 5 rows:")
            print(df.head())
            print("\nColumn Data Types:")
            print(df.dtypes)
            print("\nSummary Statistics:")
            print(df.describe())
        except Exception as e:
            print(f"Error reading {file}: {e}")

file_paths = [
    r"C:\Users\A-plus store\OneDrive - MOE Stem Schools\Desktop\Work Done\Before Training Eval\Conformer_Evaluation_results.csv",
    r"C:\Users\A-plus store\OneDrive - MOE Stem Schools\Desktop\Work Done\Before Training Eval\Hubert_Evaluation_results.csv",
    r"C:\Users\A-plus store\OneDrive - MOE Stem Schools\Desktop\Work Done\Before Training Eval\wav2vec2_asr_evaluation_results.csv",
    r"C:\Users\A-plus store\OneDrive - MOE Stem Schools\Desktop\Work Done\Before Training Eval\whisper_asr_evaluation_results.csv"
]

analyze_csv_structure(file_paths)
