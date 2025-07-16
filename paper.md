---
title: 'Evaluation Scripts for Comparative Analysis of Foundation ASR Models'
tags:
  - Python
  - Automatic Speech Recognition
  - ASR
  - dysarthric speech
  - accented speech
  - Conformer
  - Hubert
  - wav2vec2
  - Whisper
authors:
  - name: Islam Farrag Alsohby
    orcid: 0000-0000-0000-0000 # Replace with your ORCID if available, or leave as placeholder
    affiliation: "1"
affiliations:
  - name: Dakahlia STEM High School, Dakahlia Governorate, Egypt
    index: 1
date: 16 July 2025
bibliography: paper.bib
---

# Summary

This software package provides Python scripts for evaluating the performance of four state-of-the-art Automatic Speech Recognition (ASR) models—Conformer, Hubert, wav2vec2, and Whisper—on the TORGO dataset, which includes dysarthric, accented, and general speech. The scripts implement preprocessing (e.g., audio normalization to 16 kHz, text normalization to uppercase), batch processing with GPU acceleration, and evaluation using metrics such as Word Error Rate (WER), Character Error Rate (CER), Exact Match Percentage, Sequence Similarity Ratio, and Jaccard Similarity. Hosted on GitHub at `https://github.com/isiann534/Evaluation_Foundation-ASR-models-with-NSS.git`, the package enables reproducible analysis of ASR model performance, particularly for non-standard speech varieties where data is limited. The software supports research into improving ASR systems for specialized applications, such as assisting individuals with speech disabilities.

# Statement of Need

The evaluation of ASR models on diverse speech types, especially dysarthric and accented speech, requires robust, reusable software tools due to the scarcity of labeled data and the complexity of performance metrics. Existing frameworks like Hugging Face Transformers provide model implementations, but lack standardized scripts for comparative analysis across multiple metrics and datasets like TORGO. This package fills that gap by offering:

- **Flexible Preprocessing**: Handles audio and text normalization, ensuring consistency across evaluations.
- **Comprehensive Metrics**: Computes WER, CER, and similarity measures using the `jiwer` library, providing a holistic view of transcription quality.
- **Reproducibility**: Includes detailed documentation and scripts for batch processing, making it accessible to researchers and students.

The software has been used to generate results for a comparative study published as a preprint [@alsohby2025]. It is particularly valuable for students and researchers exploring ASR adaptations for low-resource or disordered speech.

# State of the Field

The field of ASR has advanced with foundation models like Conformer [@gulati2020], Hubert [@hsu2021], wav2vec2 [@baevski2020], and Whisper [@radford2023], which rely on self-supervised learning and large-scale training. However, evaluating these models on non-standard speech requires tailored tools. Studies like Shor et al. [@shor2019] highlight the need for personalized ASR with limited data, while Hu et al. [@hu2024] emphasize self-supervised approaches for dysarthric speech. This software complements existing tools like Kaldi or ESPnet with a Python-based, open-source solution.

# Design

The package is designed for ease of use and extensibility:
- **Input Handling**: Processes TORGO audio files and transcriptions, with exclusions for noisy data (e.g., marked "xxx").
- **Metric Computation**: Implements WER and CER via `jiwer`, with custom functions for Exact Match, Sequence Similarity Ratio, and Jaccard Similarity.
- **Optimization**: Uses GPU acceleration for batch processing, reducing computation time on large datasets.
- **Documentation**: Includes a README with installation instructions, usage examples, and links to the TORGO dataset documentation.

The code is written in Python, leveraging libraries like `numpy`, `torch`, and `jiwer`, and is optimized with Cython where applicable.

# Functionality

Key features include:
- **Preprocessing**: Trims silence, normalizes audio to 16 kHz, and converts text to uppercase without punctuation.
- **Evaluation**: Computes performance metrics across 2,356–3,290 samples (depending on model), as detailed in the GitHub repository.
- **Output**: Generates tables and statistics (e.g., mean WER of 1.1090 for Hubert, 1.7962 for Whisper) for analysis.

Example usage:
```python
from evaluate_asr import evaluate_model
results = evaluate_model(model="whisper", dataset="torgo", metrics=["wer", "cer"])
print(results)
