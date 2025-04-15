import os
import re
import nltk
import random
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


def download_nltk_resources():
    # Download required NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Try a different approach with direct word tokenization
    # instead of relying on punkt_tab
    try:
        # Test if tokenization works
        word_tokenize("Testing tokenization")
    except LookupError:
        # If there's an issue, let's use a more basic tokenizer
        print("Warning: Default tokenizer not available. Using basic whitespace tokenizer.")


def basic_tokenize(text):
    """Simple tokenizer that splits on whitespace and punctuation"""
    # First replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Then split on whitespace
    return text.split()


def preprocess_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()

    text = re.sub(r'[^\w\s]', ' ', text)  # Removes punctuations, replace with space
    text = re.sub(r'\d+', ' ', text)      # Removes numbers, replace with space

    # Try to use NLTK tokenizer, fall back to basic if it fails
    try:
        tokens = word_tokenize(text)
    except:
        # If NLTK tokenizer fails, use the basic one
        tokens = basic_tokenize(text)

    try:
        # Try to use NLTK stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    except:
        # If NLTK stopwords fail, use a minimal set of common English stopwords
        minimal_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 
                            'when', 'at', 'from', 'by', 'for', 'with', 'about', 'to', 'in',
                            'on', 'it', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        tokens = [token for token in tokens if token not in minimal_stopwords]

    try:
        # Try to use Porter Stemmer
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    except:
        # If stemming fails, just skip it
        pass

    processed_text = ' '.join(tokens)
    return processed_text


def xml_metadata_parser(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        metadata = {}
        
        # Extract information from feature elements
        for feature in root.findall('.//feature'):
            if feature.attrib.get('name') == 'about':
                # Extract basic information
                metadata['authors'] = feature.attrib.get('authors', 'unknown')
                metadata['title'] = feature.attrib.get('title', 'unknown')
                metadata['language'] = feature.attrib.get('lang', 'unknown')
            elif feature.attrib.get('name') == 'plagiarism':
                # If we don't have a plagiarism list yet, create one
                if 'plagiarism' not in metadata:
                    metadata['plagiarism'] = []
                
                # Extract plagiarism information
                plagiarism_info = {
                    'source_reference': feature.attrib.get('source_reference', ''),
                    'offset': int(feature.attrib.get('this_offset', -1)),
                    'length': int(feature.attrib.get('this_length', 0)),
                    'type': feature.attrib.get('type', 'unknown'),
                    'obfuscation': feature.attrib.get('obfuscation', 'none')
                }
                metadata['plagiarism'].append(plagiarism_info)
        
        return metadata

    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return {'language': 'unknown'}


def load_pan_sample(corpus_path, task_type='extrinsic', max_src=100, max_sus=100):
    documents = []
    if task_type == 'extrinsic':
        # Updated folder paths to match your corpus structure
        base_dir = os.path.join(corpus_path, 'external-detection-corpus')
        src_base_dir = os.path.join(base_dir, 'source-document')  # Singular form
        src_count = 0
        
        # Find part directories...
        part_dirs = [d for d in os.listdir(src_base_dir) if d.startswith('part')]

        for part_dir in part_dirs:
            part_path = os.path.join(src_base_dir, part_dir)
            file_list = [f for f in os.listdir(part_path) if f.endswith('.txt')]

            # randomly sample if too many files...
            if len(file_list) > max_src - src_count:
                file_list = random.sample(file_list, max_src - src_count)

            for filename in file_list:
                doc_id = filename.replace('.txt', '')
                txt_path = os.path.join(part_path, filename)
                xml_path = os.path.join(part_path, doc_id + '.xml')
           
                # read text content
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                processed_text = preprocess_text(text)

                metadata = {}
                if os.path.exists(xml_path):
                    metadata = xml_metadata_parser(xml_path)

                doc_entry = {
                    'id': doc_id,
                    'type': 'source',
                    'raw_text': text,
                    'processed_text': processed_text,
                    'metadata': metadata
                }

                documents.append(doc_entry)
                src_count += 1

                if src_count >= max_src:
                    break
            if src_count >= max_src:
                break
    else:
        base_dir = os.path.join(corpus_path, 'intrinsic-detection-corpus')
        # For intrinsic detection, we'd add different logic here

    # Process suspicious documents
    suspicious_base_dir = os.path.join(base_dir, 'suspicious-document')  # Singular form
    sus_count = 0

    part_dirs = [d for d in os.listdir(suspicious_base_dir) if d.startswith('part')]

    for part_dir in part_dirs:
        part_path = os.path.join(suspicious_base_dir, part_dir)
        file_list = [f for f in os.listdir(part_path) if f.endswith('.txt')]

        if len(file_list) > max_sus - sus_count:
            file_list = random.sample(file_list, max_sus - sus_count)

        for filename in file_list:
            doc_id = filename.replace('.txt', '')
            txt_path = os.path.join(part_path, filename)
            xml_path = os.path.join(part_path, doc_id + '.xml')
        
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            processed_text = preprocess_text(text)

            metadata = {}
            if os.path.exists(xml_path):
                metadata = xml_metadata_parser(xml_path)

            doc_entry = {
                'id': doc_id,
                'type': 'suspicious',
                'raw_text': text,
                'processed_text': processed_text,
                'metadata': metadata,
                'has_plagiarism': 'plagiarism' in metadata
            }

            documents.append(doc_entry)
            sus_count += 1

            if sus_count >= max_sus:
                break
        if sus_count >= max_sus:
            break

    df = pd.DataFrame(documents)
    
    return df


if __name__ == "__main__":
    download_nltk_resources()
