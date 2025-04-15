import os
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocessing import download_nltk_resources, load_pan_sample

# Ensure required NLTK resources are downloaded
download_nltk_resources()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extrinsic Plagiarism Detection System')
    
    parser.add_argument('--corpus_path', type=str, required=True,
                        help='Path to the corpus directory')
    
    parser.add_argument('--max_src', type=int, default=100,
                        help='Maximum number of source documents to process')
    
    parser.add_argument('--max_sus', type=int, default=100,
                        help='Maximum number of suspicious documents to process')
    
    parser.add_argument('--threshold', type=float, default=0.3,  # Lowered default threshold
                        help='Similarity threshold for plagiarism detection')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--ngram_range', type=str, default='1,2',
                        help='N-gram range for TF-IDF vectorizer (format: min,max)')
    
    return parser.parse_args()

def detect_extrinsic_plagiarism(documents_df, threshold=0.3, ngram_range=(1, 2)):
    """
    Detect plagiarism by comparing suspicious documents against source documents.
    
    Args:
        documents_df: DataFrame containing both source and suspicious documents
        threshold: Similarity threshold above which to flag plagiarism
        ngram_range: Range of n-gram size for TF-IDF vectorizer
        
    Returns:
        DataFrame with plagiarism detection results
    """
    # Separate source and suspicious documents
    source_docs = documents_df[documents_df['type'] == 'source']
    suspicious_docs = documents_df[documents_df['type'] == 'suspicious']
    
    # Check if we have data to process
    if source_docs.empty or suspicious_docs.empty:
        print("Warning: No source or suspicious documents found!")
        return pd.DataFrame(columns=['suspicious_id', 'source_id', 'similarity_score', 'ground_truth', 'source_reference'])
    
    # Create TF-IDF vectorizer with n-gram support
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    
    # Check if processed_text contains empty strings
    if source_docs['processed_text'].str.strip().eq('').any() or suspicious_docs['processed_text'].str.strip().eq('').any():
        print("Warning: Some documents have empty processed text! Check preprocessing.")
    
    # Fit and transform the source documents
    source_vectors = vectorizer.fit_transform(source_docs['processed_text'])
    
    # Transform suspicious documents using the same vectorizer
    suspicious_vectors = vectorizer.transform(suspicious_docs['processed_text'])
    
    # Initialize results list
    results = []
    
    # Compare each suspicious document with all source documents
    for i, sus_id in enumerate(suspicious_docs['id']):
        sus_vector = suspicious_vectors[i]
        
        # Calculate cosine similarity with all source documents
        similarities = cosine_similarity(sus_vector, source_vectors).flatten()
        
        # Get indices of documents with similarity above threshold
        matches = np.where(similarities >= threshold)[0]
        
        for match_idx in matches:
            source_id = source_docs.iloc[match_idx]['id']
            similarity_score = similarities[match_idx]
            
            # Check if this plagiarism is known from metadata
            ground_truth = False
            source_ref = ""
            
            if 'plagiarism' in suspicious_docs.iloc[i]['metadata']:
                for plag_info in suspicious_docs.iloc[i]['metadata']['plagiarism']:
                    if source_id in plag_info['source_reference']:
                        ground_truth = True
                        source_ref = plag_info['source_reference']
                        break
            
            results.append({
                'suspicious_id': sus_id,
                'source_id': source_id,
                'similarity_score': similarity_score,
                'ground_truth': ground_truth,
                'source_reference': source_ref
            })
    
    # Convert results to DataFrame
    if results:
        results_df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with proper columns
        results_df = pd.DataFrame(columns=['suspicious_id', 'source_id', 'similarity_score', 'ground_truth', 'source_reference'])
    
    return results_df

def evaluate_detection(results_df, suspicious_docs):
    """Evaluate the performance of the plagiarism detection system."""
    # Handle empty results case
    if results_df.empty:
        # Calculate false negatives from ground truth
        all_ground_truth = 0
        seen_plagiarism_cases = set()
        
        for idx, row in suspicious_docs.iterrows():
            if 'plagiarism' in row['metadata']:
                for plag_info in row['metadata']['plagiarism']:
                    case_id = f"{row['id']}_{plag_info['source_reference']}"
                    if case_id not in seen_plagiarism_cases:
                        all_ground_truth += 1
                        seen_plagiarism_cases.add(case_id)
        
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': all_ground_truth
        }
    
    # For extrinsic, we have direct ground truth comparisons
    true_positives = len(results_df[results_df['ground_truth'] == True])
    false_positives = len(results_df[results_df['ground_truth'] == False])
    
    # Get all ground truth cases from the metadata
    all_ground_truth = 0
    seen_plagiarism_cases = set()
    
    for idx, row in suspicious_docs.iterrows():
        if 'plagiarism' in row['metadata']:
            for plag_info in row['metadata']['plagiarism']:
                case_id = f"{row['id']}_{plag_info['source_reference']}"
                if case_id not in seen_plagiarism_cases:
                    all_ground_truth += 1
                    seen_plagiarism_cases.add(case_id)
    
    false_negatives = all_ground_truth - true_positives
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
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
    
    # Load and preprocess the corpus
    print(f"Loading corpus from {args.corpus_path}...")
    
    try:
        documents_df = load_pan_sample(
            corpus_path=args.corpus_path,
            task_type='extrinsic',  # Fixed to extrinsic only
            max_src=args.max_src,
            max_sus=args.max_sus
        )
        
        print(f"Loaded {len(documents_df)} documents")
        print(f"- Source documents: {len(documents_df[documents_df['type'] == 'source'])}")
        print(f"- Suspicious documents: {len(documents_df[documents_df['type'] == 'suspicious'])}")
        
        # Save the processed corpus for reference
        documents_df.to_csv(os.path.join(args.output_dir, 'processed_corpus.csv'), index=False)
        
        # Perform extrinsic plagiarism detection
        print("Performing extrinsic plagiarism detection...")
        print(f"Using similarity threshold: {args.threshold}")
        print(f"Using n-gram range: {ngram_range}")
        
        results_df = detect_extrinsic_plagiarism(
            documents_df, 
            threshold=args.threshold,
            ngram_range=ngram_range
        )
        
        # Save detection results
        results_df.to_csv(os.path.join(args.output_dir, 'extrinsic_results.csv'), index=False)
        
        # Report on results
        print(f"Detection complete. Found {len(results_df)} potential plagiarism cases.")
        print(f"- Cases with similarity > {args.threshold}: {len(results_df)}")
        
        if not results_df.empty:
            print(f"- Confirmed cases (from ground truth): {len(results_df[results_df['ground_truth'] == True])}")
        else:
            print("- No cases above threshold detected. Consider lowering the threshold.")
        
        # Evaluate detection performance
        print("Evaluating detection performance...")
        suspicious_docs = documents_df[documents_df['type'] == 'suspicious']
        evaluation = evaluate_detection(results_df, suspicious_docs)
        
        print(f"Precision: {evaluation['precision']:.4f}")
        print(f"Recall: {evaluation['recall']:.4f}")
        print(f"F1 Score: {evaluation['f1_score']:.4f}")
        print(f"True Positives: {evaluation['true_positives']}")
        print(f"False Positives: {evaluation['false_positives']}")
        print(f"False Negatives: {evaluation['false_negatives']}")
        
        # Save evaluation metrics
        with open(os.path.join(args.output_dir, 'evaluation_metrics.txt'), 'w') as f:
            for key, value in evaluation.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()
