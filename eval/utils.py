import pandas as pd
from datasets import Dataset, DatasetDict
import requests
import io

def load_hinglish_dataset(**kwargs):
    """
    Load the Hinglish TOP dataset from GitHub and prepare for translation evaluation
    """
    # URLs for both validation and test sets
    test_url = "https://raw.githubusercontent.com/google-research-datasets/Hinglish-TOP-Dataset/main/Dataset/Human%20Annotated%20Data/test.tsv"
    val_url = "https://raw.githubusercontent.com/google-research-datasets/Hinglish-TOP-Dataset/main/Dataset/Human%20Annotated%20Data/validation.tsv"
    train_url = "https://raw.githubusercontent.com/google-research-datasets/Hinglish-TOP-Dataset/main/Dataset/Human%20Annotated%20Data/train.tsv"
    
    def load_split(url):
        response = requests.get(url)
        response.raise_for_status()
        
        # Read TSV data
        df = pd.read_csv(io.StringIO(response.text), sep='\t')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Keep only English and Hinglish columns, rename for clarity
        df_clean = df[['en_query', 'cs_query']].copy()
        df_clean = df_clean.rename(columns={
            'en_query': 'english',
            'cs_query': 'hinglish'
        })
        
        # Remove any rows with missing data
        df_clean = df_clean.dropna()
        
        return Dataset.from_pandas(df_clean)
    
    # Load both splits
    test_dataset = load_split(test_url)
    val_dataset = load_split(val_url)
    train_dataset = load_split(train_url)
    
    return DatasetDict({
        'validation': val_dataset,
        'test': test_dataset,
        'train': train_dataset
    })
