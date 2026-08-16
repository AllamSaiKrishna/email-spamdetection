import re
import html
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Programmatically check and download required NLTK components
def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    for r in resources:
        try:
            # Check if resource is available locally
            if r == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif r == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            else:
                nltk.data.find(f'corpora/{r}')
        except LookupError:
            print(f"Downloading NLTK resource: {r}...")
            nltk.download(r, quiet=True)

# Run download on import
download_nltk_resources()

class TextPreprocessor:
    def __init__(self, use_lemmatizer=True):
        self.use_lemmatizer = use_lemmatizer
        self.lemmatizer = WordNetLemmatizer() if use_lemmatizer else None
        
        # Load standard English stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Explicitly protect important spam identifiers from being removed
        protected_words = {
            'free', 'winner', 'offer', 'money', 'urgent', 'congratulations', 
            'won', 'cash', 'prize', 'claim', 'award', 'now', 'guaranteed'
        }
        self.stop_words = self.stop_words - protected_words
        
    def clean_text(self, text):
        """
        Cleans a single string of email text.
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Convert to HTML entities unescaping first
        text = html.unescape(text)
        
        # 2. Convert to lowercase
        text = text.lower()
        
        # 3. Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 4. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # 5. Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # 6. Remove special characters and punctuation, but keep numbers/words
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 7. Normalize whitespace (remove multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 8. Tokenization
        tokens = word_tokenize(text)
        
        # 9. Stop-word removal
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # 10. Lemmatization
        if self.use_lemmatizer and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
        return " ".join(tokens)
        
    def preprocess_df(self, df, text_col='text', label_col='label'):
        """
        Takes a dataframe and returns a preprocessed copy after removing duplicates,
        handling missing values, cleaning, and filtering empty strings.
        """
        # Copy to avoid modifying the original dataframe
        df = df.copy()
        
        # 1. Handle missing values
        initial_count = len(df)
        df = df.dropna(subset=[text_col, label_col])
        print(f"Dropped {initial_count - len(df)} rows with missing values.")
        
        # 2. Remove duplicate emails
        initial_count = len(df)
        df = df.drop_duplicates(subset=[text_col])
        print(f"Removed {initial_count - len(df)} duplicate rows.")
        
        # 3. Clean email content
        print("Cleaning text column...")
        df['cleaned_text'] = df[text_col].apply(self.clean_text)
        
        # 4. Remove empty strings after cleaning
        initial_count = len(df)
        df = df[df['cleaned_text'].str.strip() != ""]
        print(f"Dropped {initial_count - len(df)} empty rows after text cleaning.")
        
        return df

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    sample = "<b>CONGRATULATIONS!</b> You have won a free $1000 prize. Claim now at http://win.com!"
    print(f"Original: {sample}")
    print(f"Cleaned : {preprocessor.clean_text(sample)}")
