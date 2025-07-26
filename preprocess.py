import pandas as pd
import re
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


def clean_text(text):
    """
    Applies basic text cleaning to a string.
    - Converts to lowercase.
    - Removes extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space
    return text

def normalize_math_expressions(text):
    """
    Normalizes common mathematical expressions to make them more consistent.
    This is a basic example; more complex normalization might be needed.
    - Standardizes decimal point notation (e.g., '0.50' to '0.5').
    - Adds spaces around operators for better tokenization later.
    - Handles common fraction notations.
    """
    if not isinstance(text, str):
        return ""

    # Replace common math symbols with space-separated versions
    # This helps in tokenization later where '2+2' becomes '2 + 2'
    text = re.sub(r'([+\-*/^=<>%])', r' \1 ', text)

    # Standardize decimal representations (e.g., 0.50 -> 0.5)
    text = re.sub(r'\b(\d+)\.0+\b', r'\1', text) # Remove trailing zeros after decimal if only zeros follow
    text = re.sub(r'\.(\d*?)0+\b', r'.\1', text) # Remove trailing zeros in decimals
    text = re.sub(r'\. $', '', text) # Remove lone decimal points at end

    # Replace common fraction text with a standardized symbol or word
    text = text.replace('one half', '1/2').replace('a half', '1/2')
    text = text.replace('one third', '1/3').replace('a third', '1/3')
    text = text.replace('one fourth', '1/4').replace('a fourth', '1/4')

    # Add spaces around numbers adjacent to non-word characters (e.g., $10 -> $ 10)
    text = re.sub(r'(\d)([^\w\s.])', r'\1 \2', text)
    text = re.sub(r'([^\w\s.])(\d)', r'\1 \2', text)

    # Remove extra spaces introduced by replacements
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_data(df, tokenizer_name="tbs17/MathBERT", max_length=512, return_labels=True):
    """
    Applies full preprocessing pipeline to a DataFrame.
    Combines QuestionText, MC_Answer, and StudentExplanation into a single input string.

    Args:
        df (pd.DataFrame): The input DataFrame (train or test).
        tokenizer_name (str): Name of the Hugging Face tokenizer to use.
        max_length (int): Maximum sequence length for the tokenizer.
        return_labels (bool): Whether to process and return labels (for training data).

    Returns:
        tuple: (tokenized_inputs, y_ohe) if return_labels=True, else just tokenized_inputs
    """
    # Category mapping for labels
    category_mapping = {
        "True_Correct": 0,
        "False_Misconception": 1,
        "False_Neither": 2,
        "True_Neither": 3,
        "True_Misconception": 4,
        "False_Correct": 5
    }
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Process text inputs
    processed_inputs = []
    for index, row in df.iterrows():
        # Apply cleaning and normalization to individual components
        question_text = normalize_math_expressions(clean_text(row['QuestionText']))
        mc_answer = clean_text(row['MC_Answer'])
        student_explanation = normalize_math_expressions(clean_text(row['StudentExplanation']))

        # Combine text components
        combined_text = (
            f"question : {question_text} "
            f"answer : {mc_answer} "
            f"explanation : {student_explanation}"
        )
        processed_inputs.append(combined_text)

    # Add processed inputs to dataframe
    df = df.copy()  # Avoid modifying original dataframe
    df['preprocessed_input'] = processed_inputs

    # Tokenize inputs
    tokenized_inputs = tokenizer(
        df['preprocessed_input'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='np'
    )
    
    # Process labels if requested and available
    if return_labels and 'Category' in df.columns:
        # Map categories to integers
        df['Category_int'] = df['Category'].map(category_mapping)
        
        # Check for any unmapped categories
        if df['Category_int'].isna().any():
            unmapped = df[df['Category_int'].isna()]['Category'].unique()
            raise ValueError(f"Unmapped categories found: {unmapped}")
        
        # Convert to numpy array and reshape for one-hot encoding
        y_int = df['Category_int'].values.reshape(-1, 1)
        
        # One-hot encode
        ohe = OneHotEncoder(sparse_output=False)
        y_ohe = ohe.fit_transform(y_int)
        
        return tokenized_inputs, y_ohe
    
    return tokenized_inputs


# Alternative: Return integer labels instead of one-hot
def preprocess_data_with_int_labels(df, tokenizer_name="tbs17/MathBERT", max_length=512):
    """
    Similar to preprocess_data but returns integer labels instead of one-hot encoded.
    This is often more convenient for PyTorch CrossEntropyLoss.
    """
    category_mapping = {
        "True_Correct": 0,
        "False_Misconception": 1,
        "False_Neither": 2,
        "True_Neither": 3,
        "True_Misconception": 4,
        "False_Correct": 5
    }
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    processed_inputs = []
    for index, row in df.iterrows():
        question_text = normalize_math_expressions(clean_text(row['QuestionText']))
        mc_answer = clean_text(row['MC_Answer'])
        student_explanation = normalize_math_expressions(clean_text(row['StudentExplanation']))

        combined_text = (
            f"question : {question_text} "
            f"answer : {mc_answer} "
            f"explanation : {student_explanation}"
        )
        processed_inputs.append(combined_text)

    df = df.copy()
    df['preprocessed_input'] = processed_inputs

    tokenized_inputs = tokenizer(
        df['preprocessed_input'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    if 'Category' in df.columns:
        df['Category_int'] = df['Category'].map(category_mapping)
        if df['Category_int'].isna().any():
            unmapped = df[df['Category_int'].isna()]['Category'].unique()
            raise ValueError(f"Unmapped categories found: {unmapped}")
        
        return tokenized_inputs, df['Category_int'].values
    
    return tokenized_inputs