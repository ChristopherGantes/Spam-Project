# Spam Detection Using Machine Learning Models

## Overview

This project focuses on spam detection in emails, utilizing machine learning models and data mining techniques. Conducted as part of a research project at the University of Georgia (January 2023 – May 2023), it involves comparing the performance of various scikit-learn models to determine the most effective approach for spam detection.

## Features

- **Model Comparison**: Evaluated and compared five different scikit-learn models.
- **Data Processing**: Preprocessed a labeled dataset of emails to extract relevant features.
- **Model Implementation**: Implemented models using Python and scikit-learn.
- **Performance Evaluation**: Assessed models based on accuracy, precision, recall, and F1 score.
- **Research Documentation**: Compiled findings into a detailed research paper and presented results.

## Models Compared

1. **Decision Trees**
2. **Logistic Regression**
3. **Naive Bayes**
4. **Random Forest**
5. **Support Vector Machines (SVM)**

## File Structure

- `main.py`: The main script for preprocessing data, training models, and evaluating results.
- `data/`: Directory containing CSV files for training and testing models.
  - `spam.csv`
  - `email_classification.csv`
- `research_paper.pdf`: The research paper detailing the study, methodology, and findings.

## Installation

To set up the project locally using Anaconda:

1. Clone the repository:
   ```bash
   git clone https://github.com/christophergantes/spam-project.git
   cd spam-project
   ```

2. Create a new Anaconda environment:
   ```bash
   conda create --name spam-project python=3.8
   conda activate spam-project
   ```

3. Install the required packages:
   ```bash
   conda install scikit-learn pandas scipy
   ```

## Usage

1. Ensure your environment is activated:
   ```bash
   conda activate spam-project
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

## Research Paper

The research paper detailing the study, methodology, and findings can be found at `research_paper.pdf` in the project directory. It includes a comprehensive analysis of the models, their performance metrics, and conclusions drawn from the research.