import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import download_nltk_resources, preprocess_text
import nltk
from nltk.tokenize import sent_tokenize

# Ensure required NLTK resources are downloaded
download_nltk_resources()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Chunked Document Plagiarism Comparison')
    
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the source document')
    
    parser.add_argument('--suspicious', type=str, required=True,
                        help='Path to the suspicious document')
    
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Similarity threshold for plagiarism detection')
    
    parser.add_argument('--ngram_range', type=str, default='1,2',
                        help='N-gram range for TF-IDF vectorizer (format: min,max)')
    
    parser.add_argument('--chunk_size', type=int, default=5,
                        help='Number of sentences per chunk')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    return parser.parse_args()

def chunk_text(text, chunk_size=5):
    """
    Divide text into chunks of specified number of sentences.
    
    Args:
        text: Text to be chunked
        chunk_size: Number of sentences per chunk
        
    Returns:
        List of text chunks
    """
    try:
        # Try to use NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
    except:
        # Fallback to simple splitting on periods, question marks, and exclamation points
        text = text.replace('!', '.').replace('?', '.')
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    # Create chunks of sentences
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    # If we have no chunks (which might happen with very short texts), 
    # just use the whole text as one chunk
    if not chunks and text.strip():
        chunks = [text]
    
    return chunks

def compare_documents_with_chunking(source_text, suspicious_text, threshold=0.3, ngram_range=(1, 2), chunk_size=5):
    """
    Compare source and suspicious documents using text chunking to detect partial plagiarism.
    
    Args:
        source_text: Raw text of the source document
        suspicious_text: Raw text of the suspicious document
        threshold: Similarity threshold above which to flag plagiarism
        ngram_range: Range of n-gram size for TF-IDF vectorizer
        chunk_size: Number of sentences per chunk
        
    Returns:
        Dictionary with comparison results
    """
    # First create chunks
    source_chunks = chunk_text(source_text, chunk_size)
    suspicious_chunks = chunk_text(suspicious_text, chunk_size)
    
    # Preprocess each chunk
    source_processed = [preprocess_text(chunk) for chunk in source_chunks]
    suspicious_processed = [preprocess_text(chunk) for chunk in suspicious_chunks]
    
    # Remove empty processed chunks
    source_processed = [chunk for chunk in source_processed if chunk.strip()]
    suspicious_processed = [suspicious_processed for chunk in suspicious_processed if chunk.strip()]
    
    # If any document has no valid chunks after preprocessing, return early
    if not source_processed or not suspicious_processed:
        print("Warning: One or both documents have no valid content after preprocessing!")
        return {
            'overall_similarity': 0.0,
            'max_chunk_similarity': 0.0,
            'plagiarized_chunks': [],
            'all_chunk_similarities': []
        }
    
    # Create TF-IDF vectorizer with n-gram support
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    
    # Fit on all chunks from both documents
    all_chunks = source_processed + suspicious_processed
    vectorizer.fit(all_chunks)
    
    # Transform chunks
    source_vectors = vectorizer.transform(source_processed)
    suspicious_vectors = vectorizer.transform(suspicious_processed)
    
    # Calculate similarities between all source and suspicious chunks
    chunk_similarities = []
    plagiarized_chunks = []
    
    # For larger document, we can optimize to avoid excessive memory usage
    for i, sus_vector in enumerate(suspicious_vectors):
        # Calculate similarity with all source chunks at once
        similarities = cosine_similarity(sus_vector, source_vectors).flatten()
        
        # Find max similarity for this suspicious chunk
        max_similarity = similarities.max()
        max_source_idx = similarities.argmax()
        
        chunk_info = {
            'suspicious_chunk_idx': i,
            'suspicious_chunk_text': suspicious_chunks[i][:100] + "...",  # Preview
            'source_chunk_idx': max_source_idx,
            'source_chunk_text': source_chunks[max_source_idx][:100] + "...",  # Preview
            'similarity': max_similarity
        }
        
        chunk_similarities.append(chunk_info)
        
        # If similarity is above threshold, mark as plagiarized
        if max_similarity >= threshold:
            plagiarized_chunks.append({
                'suspicious_chunk_idx': i,
                'suspicious_chunk_text': suspicious_chunks[i],
                'source_chunk_idx': max_source_idx,
                'source_chunk_text': source_chunks[max_source_idx],
                'similarity': max_similarity
            })
    
    # Calculate overall document similarity (average of max similarities for each suspicious chunk)
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

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse n-gram range
    try:
        ngram_min, ngram_max = map(int, args.ngram_range.split(','))
        ngram_range = (ngram_min, ngram_max)
    except:
        print("Warning: Invalid ngram_range format. Using default (1,2).")
        ngram_range = (1, 2)
    
    # Check if files exist
    if not os.path.exists(args.source):
        print(f"Error: Source file not found: {args.source}")
        return
    
    if not os.path.exists(args.suspicious):
        print(f"Error: Suspicious file not found: {args.suspicious}")
        return
    
    try:
        # Read the source document
        print(f"Reading source document: {args.source}")
        with open(args.source, 'r', encoding='utf-8', errors='ignore') as f:
            source_text = f.read()
        
        # Read the suspicious document
        print(f"Reading suspicious document: {args.suspicious}")
        with open(args.suspicious, 'r', encoding='utf-8', errors='ignore') as f:
            suspicious_text = f.read()
        
        # Compare documents with chunking
        print("Comparing documents with chunking...")
        print(f"Using similarity threshold: {args.threshold}")
        print(f"Using n-gram range: {ngram_range}")
        print(f"Using chunk size: {args.chunk_size} sentences")
        
        result = compare_documents_with_chunking(
            source_text, 
            suspicious_text, 
            threshold=args.threshold,
            ngram_range=ngram_range,
            chunk_size=args.chunk_size
        )
        
        # Display results
        print("\n--- RESULTS ---")
        print(f"Overall Similarity: {result['overall_similarity']:.4f}")
        print(f"Max Chunk Similarity: {result['max_chunk_similarity']:.4f}")
        print(f"Plagiarized Chunks: {len(result['plagiarized_chunks'])}")
        
        # Save results to file
        result_file = os.path.join(args.output_dir, 'chunked_comparison_result.txt')
        with open(result_file, 'w') as f:
            f.write(f"Source Document: {args.source}\n")
            f.write(f"Suspicious Document: {args.suspicious}\n")
            f.write(f"Similarity Threshold: {args.threshold}\n")
            f.write(f"N-gram Range: {ngram_range}\n")
            f.write(f"Chunk Size: {args.chunk_size} sentences\n\n")
            f.write(f"Overall Similarity: {result['overall_similarity']:.4f}\n")
            f.write(f"Max Chunk Similarity: {result['max_chunk_similarity']:.4f}\n")
            f.write(f"Plagiarized Chunks: {len(result['plagiarized_chunks'])}\n\n")
            
            # Write detailed information about plagiarized chunks
            if result['plagiarized_chunks']:
                f.write("--- PLAGIARIZED CHUNKS ---\n")
                for i, chunk in enumerate(result['plagiarized_chunks']):
                    f.write(f"\nMatch {i+1} (Similarity: {chunk['similarity']:.4f}):\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Source Chunk [{chunk['source_chunk_idx']}]:\n{chunk['source_chunk_text']}\n\n")
                    f.write(f"Suspicious Chunk [{chunk['suspicious_chunk_idx']}]:\n{chunk['suspicious_chunk_text']}\n")
                    f.write("-" * 80 + "\n")
        
        print(f"\nResults saved to {result_file}")
        
        # Save CSV with all chunk similarities for potential visualization
        similarities_file = os.path.join(args.output_dir, 'chunk_similarities.csv')
        similarities_df = pd.DataFrame(result['all_chunk_similarities'])
        similarities_df.to_csv(similarities_file, index=False)
        print(f"Chunk similarities saved to {similarities_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
