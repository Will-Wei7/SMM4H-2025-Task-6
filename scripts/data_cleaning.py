import pandas as pd
import re
import tempfile
import os
import shutil

def clean_text(text):
    """
    Clean text for BERT classification of vaccine adverse events:
    1. Extract comment part (after first colon) if present
    2. Remove URLs, user mentions, subreddit references, and markdown elements
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text with minimal modifications
    """
    if not isinstance(text, str):
        return ""
    
    # Step 1: Split title and comment at first colon
    # If colon exists, take the part after it (the comment)
    # If no colon, keep the entire text
    if ':' in text:
        parts = text.split(':', 1)  # Split only at the first colon
        if len(parts) > 1 and parts[1].strip():  # Ensure there's content after the colon
            text = parts[1].strip()
    
    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Step 3: Remove Reddit-specific elements
    # Remove user mentions (u/username)
    text = re.sub(r'u/\w+', '', text)
    
    # Remove subreddit mentions (r/subreddit)
    text = re.sub(r'r/\w+', '', text)
    
    # Remove markdown elements (*, _, >, etc.)
    # Bold/italic
    text = re.sub(r'\*\*|\*|__|\^|~~', '', text)
    
    # Remove blockquotes
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    
    # Remove special tags like [gif], [url], [video], etc.
    text = re.sub(r'\[\w+\]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_process_data(train_file, valid_file, test_file):
    """
    Load datasets and apply cleaning
    
    Args:
        train_file (str): Path to the training data CSV
        valid_file (str): Path to the validation data CSV
        test_file (str): Path to the test data CSV
        
    Returns:
        tuple: Processed training, validation and test dataframes
    """
    # Load data
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)
    print(f"Training data shape: {train_df.shape}")
    print(f"Validation data shape: {valid_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Apply cleaning
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    valid_df['cleaned_text'] = valid_df['text'].apply(clean_text)
    test_df['cleaned_text'] = test_df['text'].apply(clean_text)

    # Also keep the original title (part before the colon) if present
    def extract_title(text):
        if not isinstance(text, str):
            return ""
        if ':' in text:
            return text.split(':', 1)[0].strip()
        return ""
    
    train_df['title'] = train_df['text'].apply(extract_title)
    valid_df['title'] = valid_df['text'].apply(extract_title)
    test_df['title'] = test_df['text'].apply(extract_title)
    
    return train_df, valid_df, test_df

def save_processed_data(df, output_file):
    """
    Save the processed dataframe to a CSV file
    
    Args:
        df (DataFrame): Processed dataframe
        output_file (str): Path to save the processed data
    """
    try:
        # First try direct save
        df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    except PermissionError:
        print(f"Permission denied for {output_file}. Trying alternative method...")
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            temp_path = tmp_file.name
            try:
                # Save to temporary file
                df.to_csv(temp_path, index=False)
                # Try to replace the original file
                try:
                    shutil.move(temp_path, output_file)
                    print(f"Processed data saved to {output_file}")
                except Exception as e:
                    # If move fails, save with a different name
                    new_file = f"{output_file}.new"
                    shutil.move(temp_path, new_file)
                    print(f"Could not save to {output_file}. Data saved to {new_file}")
                    print(f"Please close any programs that might have {output_file} open and try again.")
            except Exception as e:
                print(f"Error saving data: {str(e)}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

def main():
    """
    Main function to run the preprocessing pipeline
    """
    # Load and process data
    train_df, valid_df, test_df = load_and_process_data('train.csv', 'valid.csv', 'test.csv')
    
    # Save processed data
    save_processed_data(train_df, 'processed_train.csv')
    save_processed_data(valid_df, 'processed_valid.csv')
    save_processed_data(test_df, 'processed_test.csv')

    # Print example of original vs cleaned text
    print("\nExample of original vs cleaned text:")
    example = train_df.iloc[0]
    print(f"Original: {example['text'][:100]}...")
    print(f"Cleaned: {example['cleaned_text'][:100]}...")
    
    # Count posts with title/comment structure
    train_with_colon = sum(1 for text in train_df['text'] if ':' in str(text))
    valid_with_colon = sum(1 for text in valid_df['text'] if ':' in str(text))
    
    print(f"\nPosts with title/comment structure (contains colon):")
    print(f"Training: {train_with_colon} out of {len(train_df)} ({train_with_colon/len(train_df)*100:.1f}%)")
    print(f"Validation: {valid_with_colon} out of {len(valid_df)} ({valid_with_colon/len(valid_df)*100:.1f}%)")

if __name__ == "__main__":
    main() 