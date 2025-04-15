#!/usr/bin/env python3
"""
Plagiarism Detection System
--------------------------
A comprehensive tool for detecting plagiarism using multiple methods.
"""

import os
import sys
import argparse
import subprocess

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plagiarism Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Process a corpus for extrinsic plagiarism detection
    python plagiarism_detector.py corpus --corpus_path /path/to/corpus --threshold 0.3
    
    # Compare two documents using TF-IDF
    python plagiarism_detector.py compare --source doc1.txt --suspicious doc2.txt
    
    # Detect paraphrased plagiarism using BERT
    python plagiarism_detector.py bert --source doc1.txt --suspicious doc2.txt
    '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Detection mode')
    
    # Corpus-level detection parser
    corpus_parser = subparsers.add_parser('corpus', help='Process an entire corpus')
    corpus_parser.add_argument('--corpus_path', type=str, required=True,
                              help='Path to the corpus directory')
    corpus_parser.add_argument('--max_src', type=int, default=100,
                              help='Maximum number of source documents to process')
    corpus_parser.add_argument('--max_sus', type=int, default=100,
                              help='Maximum number of suspicious documents to process')
    corpus_parser.add_argument('--threshold', type=float, default=0.3,
                              help='Similarity threshold for plagiarism detection')
    corpus_parser.add_argument('--output_dir', type=str, default='results',
                              help='Directory to save results')
    corpus_parser.add_argument('--ngram_range', type=str, default='1,2',
                             help='N-gram range for TF-IDF vectorizer (format: min,max)')
    
    # TF-IDF document comparison parser
    compare_parser = subparsers.add_parser('compare', help='Compare two documents using TF-IDF')
    compare_parser.add_argument('--source', type=str, required=True,
                               help='Path to the source document')
    compare_parser.add_argument('--suspicious', type=str, required=True,
                               help='Path to the suspicious document')
    compare_parser.add_argument('--threshold', type=float, default=0.3,
                               help='Similarity threshold for plagiarism detection')
    compare_parser.add_argument('--ngram_range', type=str, default='1,2',
                               help='N-gram range for TF-IDF vectorizer (format: min,max)')
    compare_parser.add_argument('--output_dir', type=str, default='results',
                               help='Directory to save results')
    
    # BERT-based paraphrase detection parser
    bert_parser = subparsers.add_parser('bert', help='Detect paraphrased plagiarism using BERT')
    bert_parser.add_argument('--source', type=str, required=True,
                            help='Path to the source document')
    bert_parser.add_argument('--suspicious', type=str, required=True,
                            help='Path to the suspicious document')
    bert_parser.add_argument('--threshold', type=float, default=0.8,
                            help='Similarity threshold for plagiarism detection (higher for BERT)')
    bert_parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                            help='BERT model to use for embeddings')
    bert_parser.add_argument('--chunk_size', type=int, default=1,
                            help='Number of sentences per chunk')
    bert_parser.add_argument('--output_dir', type=str, default='results',
                            help='Directory to save results')
    bert_parser.add_argument('--cache_dir', type=str, default=None,
                            help='Directory to cache BERT model')
    
    # Chunked TF-IDF comparison (optional)
    chunk_parser = subparsers.add_parser('chunk', help='Compare documents using TF-IDF with chunking')
    chunk_parser.add_argument('--source', type=str, required=True,
                             help='Path to the source document')
    chunk_parser.add_argument('--suspicious', type=str, required=True,
                             help='Path to the suspicious document')
    chunk_parser.add_argument('--threshold', type=float, default=0.3,
                             help='Similarity threshold for plagiarism detection')
    chunk_parser.add_argument('--ngram_range', type=str, default='1,2',
                             help='N-gram range for TF-IDF vectorizer (format: min,max)')
    chunk_parser.add_argument('--chunk_size', type=int, default=5,
                             help='Number of sentences per chunk')
    chunk_parser.add_argument('--output_dir', type=str, default='results',
                             help='Directory to save results')
    
    return parser.parse_args()

def run_corpus_detection(args):
    """Run the corpus-level plagiarism detection."""
    from src.main import main
    
    # Call the main function directly
    sys.argv = [
        'main.py',
        '--corpus_path', args.corpus_path,
        '--max_src', str(args.max_src),
        '--max_sus', str(args.max_sus),
        '--threshold', str(args.threshold),
        '--output_dir', args.output_dir,
        '--ngram_range', args.ngram_range
    ]
    
    main()

def run_document_comparison(args):
    """Run the TF-IDF based document comparison."""
    from src.compare_docs import main
    
    # Call the main function directly
    sys.argv = [
        'compare_docs.py',
        '--source', args.source,
        '--suspicious', args.suspicious,
        '--threshold', str(args.threshold),
        '--ngram_range', args.ngram_range,
        '--output_dir', args.output_dir
    ]
    
    main()

def run_bert_detection(args):
    """Run the BERT-based paraphrase detection."""
    from src.bert_detector import main
    
    # Call the main function directly
    sys.argv = [
        'bert_detector.py',
        '--source', args.source,
        '--suspicious', args.suspicious,
        '--threshold', str(args.threshold),
        '--model', args.model,
        '--chunk_size', str(args.chunk_size),
        '--output_dir', args.output_dir
    ]
    
    if args.cache_dir:
        sys.argv.extend(['--cache_dir', args.cache_dir])
    
    main()

def run_chunk_comparison(args):
    """Run the chunked TF-IDF comparison."""
    from src.chunk_compare import main
    
    # Call the main function directly
    sys.argv = [
        'chunk_compare.py',
        '--source', args.source,
        '--suspicious', args.suspicious,
        '--threshold', str(args.threshold),
        '--ngram_range', args.ngram_range,
        '--chunk_size', str(args.chunk_size),
        '--output_dir', args.output_dir
    ]
    
    main()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if hasattr(args, 'output_dir'):
        os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Route to the appropriate detection method
        if args.command == 'corpus':
            print("Running corpus-level plagiarism detection...")
            run_corpus_detection(args)
        
        elif args.command == 'compare':
            print("Running TF-IDF document comparison...")
            run_document_comparison(args)
        
        elif args.command == 'bert':
            print("Running BERT-based paraphrase detection...")
            run_bert_detection(args)
        
        elif args.command == 'chunk':
            print("Running chunked TF-IDF comparison...")
            run_chunk_comparison(args)
        
        else:
            print("Please specify a detection mode. Use --help for more information.")
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
