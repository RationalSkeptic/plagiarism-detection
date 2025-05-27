import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
import re

try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK punkt: {e}")
    print("Will use basic sentence splitting as fallback")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BERT-based Plagiarism Detection for Paraphrased Content')
    
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the source document')
    
    parser.add_argument('--suspicious', type=str, required=True,
                        help='Path to the suspicious document')
    
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Similarity threshold for plagiarism detection (higher for BERT)')
    
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='BERT model to use for embeddings')
    
    parser.add_argument('--chunk_size', type=int, default=1,
                        help='Number of sentences per chunk')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to cache BERT model')
    
    return parser.parse_args()

def basic_sentence_tokenize(text):
    """Basic sentence tokenization as fallback if NLTK fails."""
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    return sentences

def chunk_text(text, chunk_size=1):
    """
    Divide text into chunks of specified number of sentences.
    
    Args:
        text: Text to be chunked
        chunk_size: Number of sentences per chunk
        
    Returns:
        List of text chunks
    """
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        print(f"Warning: NLTK sentence tokenization failed: {e}")
        sentences = basic_sentence_tokenize(text)
    
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    if not chunks and text.strip():
        chunks = [text]
    
    return chunks

def get_bert_embeddings(texts, model_name, cache_dir=None):
    """
    Get BERT embeddings for a list of texts.
    
    Args:
        texts: List of text chunks to embed
        model_name: Name of the BERT model to use
        cache_dir: Directory to cache the model
        
    Returns:
        Array of embeddings
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        print("Trying alternative model: distilbert-base-uncased")
        try:
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception as e2:
            raise RuntimeError(f"Failed to load alternative model: {e2}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    batch_size = 8
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        sentence_embeddings = model_output.last_hidden_state.mean(dim=1)
        
        embeddings.append(sentence_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)

def detect_paraphrased_plagiarism(source_text, suspicious_text, threshold=0.8, model_name='sentence-transformers/all-MiniLM-L6-v2', chunk_size=1, cache_dir=None):
    """
    Detect paraphrased plagiarism using BERT embeddings.
    
    Args:
        source_text: Source document text
        suspicious_text: Suspicious document text
        threshold: Similarity threshold for plagiarism detection
        model_name: BERT model to use
        chunk_size: Number of sentences per chunk
        cache_dir: Directory to cache BERT model
        
    Returns:
        Dictionary with detection results
    """
    print("Dividing texts into chunks...")
    source_chunks = chunk_text(source_text, chunk_size)
    suspicious_chunks = chunk_text(suspicious_text, chunk_size)
    
    print(f"Processing {len(source_chunks)} source chunks and {len(suspicious_chunks)} suspicious chunks")
    
    if not source_chunks or not suspicious_chunks:
        print("Warning: One or both documents have no valid chunks")
        return {
            'overall_similarity': 0.0,
            'max_chunk_similarity': 0.0,
            'plagiarized_chunks': [],
            'all_chunk_similarities': []
        }
    
    print(f"Generating BERT embeddings using model: {model_name}...")
    print("This may take a while for longer documents...")
    
    try:
        all_chunks = source_chunks + suspicious_chunks
        all_embeddings = get_bert_embeddings(all_chunks, model_name, cache_dir)
        
        source_embeddings = all_embeddings[:len(source_chunks)]
        suspicious_embeddings = all_embeddings[len(source_chunks):]
        
        print("Calculating similarities between chunks...")
        plagiarized_chunks = []
        chunk_similarities = []
        
        for i, sus_embedding in enumerate(suspicious_embeddings):
            similarities = cosine_similarity(
                sus_embedding.reshape(1, -1), 
                source_embeddings
            ).flatten()
            
            max_similarity = similarities.max()
            max_source_idx = similarities.argmax()
            
            chunk_info = {
                'suspicious_chunk_idx': i,
                'suspicious_chunk_text': suspicious_chunks[i][:100] + "..." if len(suspicious_chunks[i]) > 100 else suspicious_chunks[i],
                'source_chunk_idx': max_source_idx,
                'source_chunk_text': source_chunks[max_source_idx][:100] + "..." if len(source_chunks[max_source_idx]) > 100 else source_chunks[max_source_idx],
                'similarity': max_similarity
            }
            
            chunk_similarities.append(chunk_info)
            
            if max_similarity >= threshold:
                plagiarized_chunks.append({
                    'suspicious_chunk_idx': i,
                    'suspicious_chunk_text': suspicious_chunks[i],
                    'source_chunk_idx': max_source_idx,
                    'source_chunk_text': source_chunks[max_source_idx],
                    'similarity': max_similarity
                })
        
        if chunk_similarities:
            overall_similarity = sum(info['similarity'] for info in chunk_similarities) / len(chunk_similarities)
            max_chunk_similarity = max(info['similarity'] for info in chunk_similarities)
        else:
            overall_similarity = 0.0
            max_chunk_similarity = 0.0
        
        return {
            'overall_similarity': overall_similarity,
            'max_chunk_similarity': max_chunk_similarity,
            'plagiarized_chunks': plagiarized_chunks,
            'all_chunk_similarities': chunk_similarities
        }
    
    except Exception as e:
        print(f"Error in BERT processing: {e}")
        import traceback
        traceback.print_exc()
        return {
            'overall_similarity': 0.0,
            'max_chunk_similarity': 0.0,
            'plagiarized_chunks': [],
            'all_chunk_similarities': [],
            'error': str(e)
        }

def main():
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.source):
        print(f"Error: Source file not found: {args.source}")
        return
    
    if not os.path.exists(args.suspicious):
        print(f"Error: Suspicious file not found: {args.suspicious}")
        return
    
    try:
        print(f"Reading source document: {args.source}")
        with open(args.source, 'r', encoding='utf-8', errors='ignore') as f:
            source_text = f.read()
        
        print(f"Reading suspicious document: {args.suspicious}")
        with open(args.suspicious, 'r', encoding='utf-8', errors='ignore') as f:
            suspicious_text = f.read()
        
        print("Analyzing documents for paraphrased plagiarism...")
        print(f"Using similarity threshold: {args.threshold}")
        print(f"Using BERT model: {args.model}")
        print(f"Using chunk size: {args.chunk_size} sentence(s)")
        
        result = detect_paraphrased_plagiarism(
            source_text, 
            suspicious_text, 
            threshold=args.threshold,
            model_name=args.model,
            chunk_size=args.chunk_size,
            cache_dir=args.cache_dir
        )
        
        print("\n=== PLAGIARISM DETECTION RESULTS ===")
        print(f"Overall Semantic Similarity: {result['overall_similarity']:.4f}")
        print(f"Maximum Chunk Similarity: {result['max_chunk_similarity']:.4f}")
        print(f"Detected Plagiarized Chunks: {len(result['plagiarized_chunks'])}/{len(result['all_chunk_similarities'])}")
        
        result_file = os.path.join(args.output_dir, 'bert_plagiarism_results.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("=== PLAGIARISM DETECTION RESULTS ===\n\n")
            f.write(f"Source Document: {args.source}\n")
            f.write(f"Suspicious Document: {args.suspicious}\n")
            f.write(f"BERT Model: {args.model}\n")
            f.write(f"Similarity Threshold: {args.threshold}\n")
            f.write(f"Chunk Size: {args.chunk_size} sentence(s)\n\n")
            
            f.write(f"Overall Semantic Similarity: {result['overall_similarity']:.4f}\n")
            f.write(f"Maximum Chunk Similarity: {result['max_chunk_similarity']:.4f}\n")
            f.write(f"Detected Plagiarized Chunks: {len(result['plagiarized_chunks'])}/{len(result['all_chunk_similarities'])}\n\n")
            
            if result['plagiarized_chunks']:
                f.write("=== DETECTED PLAGIARIZED CONTENT ===\n\n")
                for i, chunk in enumerate(result['plagiarized_chunks']):
                    f.write(f"Match {i+1} (Similarity: {chunk['similarity']:.4f}):\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"SOURCE [{chunk['source_chunk_idx']}]:\n{chunk['source_chunk_text']}\n\n")
                    f.write(f"SUSPICIOUS [{chunk['suspicious_chunk_idx']}]:\n{chunk['suspicious_chunk_text']}\n")
                    f.write("-" * 80 + "\n\n")
            else:
                f.write("No plagiarized content detected above the threshold.\n")
        
        csv_file = os.path.join(args.output_dir, 'bert_chunk_similarities.csv')
        similarities_df = pd.DataFrame(result['all_chunk_similarities'])
        similarities_df.to_csv(csv_file, index=False)
        
        print(f"\nDetailed results saved to {result_file}")
        print(f"Chunk similarities saved to {csv_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
