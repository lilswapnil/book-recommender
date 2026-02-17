#!/usr/bin/env python3
"""
Data setup script for Book Recommender.

This script ensures all required data files are present and properly formatted.
On first run, it downloads the Goodreads dataset and generates necessary files.
"""

import os
import sys
import pandas as pd
import kagglehub
from pathlib import Path


def ensure_directory_structure():
    """Create necessary data directories if they don't exist."""
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "data/chroma_db"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✓ Directory structure verified")


def check_required_files():
    """Check which data files are missing."""
    required_files = {
        "data/processed/sentiment_analyzed_books.csv": "Sentiment-analyzed book metadata",
        "data/raw/tagged_description.txt": "Raw book descriptions for embeddings"
    }
    
    missing_files = {}
    for filepath, description in required_files.items():
        if not os.path.exists(filepath):
            missing_files[filepath] = description
    
    return missing_files


def download_goodreads_dataset():
    """Download the Goodreads 100k books dataset from Kaggle."""
    print("\nDownloading Goodreads dataset from Kaggle...")
    print("(This requires kagglehub to be configured with your Kaggle API credentials)")
    
    try:
        # Download the dataset
        dataset_path = kagglehub.dataset_download("mdhamani/goodreads-books-100k")
        print(f"✓ Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTo download the dataset manually:")
        print("1. Visit: https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k")
        print("2. Click 'Download' and extract the files")
        print("3. Place the CSV files in the 'data/raw' directory")
        sys.exit(1)


def process_goodreads_data(dataset_path):
    """Process raw Goodreads data into required formats."""
    print("\nProcessing Goodreads dataset...")
    
    # Find the main CSV file (usually named something like 'books.csv')
    csv_files = list(Path(dataset_path).glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in downloaded dataset")
        sys.exit(1)
    
    # Load the main dataset
    books_csv = str(csv_files[0])
    print(f"  Reading {Path(books_csv).name}...")
    df = pd.read_csv(books_csv, low_memory=False)
    
    # Create sentiment-analyzed version (add placeholder sentiment scores if not present)
    processed_df = df.copy()
    
    # Add sentiment columns if they don't exist
    sentiment_columns = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
    for col in sentiment_columns:
        if col not in processed_df.columns:
            # Add random sentiment scores as placeholder (in production, would use actual sentiment analysis)
            processed_df[col] = 0.5
    
    # Ensure required columns exist
    required_columns = ['title', 'authors', 'description', 'category', 'isbn', 'cover_image_url']
    for col in required_columns:
        if col not in processed_df.columns:
            if col == 'cover_image_url':
                processed_df[col] = ''
            elif col == 'category':
                processed_df[col] = 'Fiction'  # Default category
            else:
                processed_df[col] = 'Unknown'
    
    # Save processed data
    output_path = "data/processed/sentiment_analyzed_books.csv"
    processed_df.to_csv(output_path, index=False)
    print(f"✓ Saved {output_path} ({len(processed_df)} books)")
    
    # Create raw descriptions file for vector embeddings
    print("  Creating descriptions file for embeddings...")
    descriptions_output = "data/raw/tagged_description.txt"
    
    with open(descriptions_output, 'w', encoding='utf-8') as f:
        for idx, row in processed_df.iterrows():
            if pd.notna(row.get('description')):
                # Format: ISBN description
                isbn = str(row.get('isbn', idx))
                description = str(row.get('description', ''))[:500]  # Truncate long descriptions
                f.write(f"{isbn} {description}\n")
    
    print(f"✓ Saved {descriptions_output} ({len(processed_df)} entries)")


def generate_sample_data():
    """Generate sample data for demo purposes."""
    print("\nGenerating sample dataset (demo mode)...")
    
    # Create sample books dataframe
    sample_data = {
        'title': [
            'The Great Gatsby',
            'To Kill a Mockingbird',
            'Pride and Prejudice',
            '1984',
            'The Catcher in the Rye'
        ],
        'authors': [
            'F. Scott Fitzgerald',
            'Harper Lee',
            'Jane Austen',
            'George Orwell',
            'J.D. Salinger'
        ],
        'description': [
            'A classic American novel set in the Jazz Age about wealth, love, and the American Dream',
            'A gripping tale of racial injustice and childhood innocence in the American South',
            'A witty romance following Elizabeth Bennet as she navigates marriage prospects',
            'A dystopian novel depicting a totalitarian government and individual resistance',
            'A coming-of-age story narrated by a cynical teenage protagonist'
        ],
        'category': ['Fiction', 'Fiction', 'Romance', 'Dystopian', 'Coming-of-Age'],
        'isbn': ['978-0743273565', '978-0061120084', '978-0141439518', '978-0451524935', '978-0316769174'],
        'cover_image_url': ['', '', '', '', ''],
        'joy': [0.5, 0.3, 0.7, 0.2, 0.4],
        'sadness': [0.4, 0.6, 0.2, 0.5, 0.3],
        'anger': [0.3, 0.5, 0.1, 0.8, 0.2],
        'fear': [0.2, 0.4, 0.1, 0.7, 0.3],
        'surprise': [0.5, 0.5, 0.6, 0.4, 0.5],
        'neutral': [0.6, 0.5, 0.5, 0.6, 0.7]
    }
    
    df = pd.DataFrame(sample_data)
    output_path = "data/processed/sentiment_analyzed_books.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Created sample {output_path}")
    
    # Create descriptions file
    with open("data/raw/tagged_description.txt", 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            f.write(f"{row['isbn']} {row['description']}\n")
    print("✓ Created sample data/raw/tagged_description.txt")


def main():
    """Main setup workflow."""
    print("=" * 60)
    print("Book Recommender - Data Setup")
    print("=" * 60)
    
    # Step 1: Ensure directories exist
    ensure_directory_structure()
    
    # Step 2: Check for missing files
    missing_files = check_required_files()
    
    if not missing_files:
        print("\n✓ All required data files found!")
        return
    
    print(f"\n⚠️  Missing {len(missing_files)} data file(s):")
    for filepath, description in missing_files.items():
        print(f"   - {filepath} ({description})")
    
    # Step 3: Offer options
    print("\nOptions:")
    print("1) Download full Goodreads dataset (requires Kaggle API)")
    print("2) Generate sample dataset for demo")
    
    try:
        choice = input("\nSelect option (1 or 2) [default=2]: ").strip() or "2"
    except (KeyboardInterrupt, EOFError):
        choice = "2"
    
    if choice == "1":
        dataset_path = download_goodreads_dataset()
        process_goodreads_data(dataset_path)
    else:
        generate_sample_data()
    
    print("\n" + "=" * 60)
    print("✓ Data setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
