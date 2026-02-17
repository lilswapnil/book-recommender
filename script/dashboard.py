import pandas as pd
import numpy as np
import os
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

# Check if required data files exist, if not run setup
if not os.path.exists('../data/processed/sentiment_analyzed_books.csv'):
    print("Required data files not found. Running setup...")
    import setup_data
    setup_data.main()

# Verify OpenAI API key is configured before proceeding
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add your OpenAI API key.")

# Load book metadata with sentiment analysis scores
books = pd.read_csv('../data/processed/sentiment_analyzed_books.csv')

# Configure vector database persistence
persist_directory = "../data/chroma_db"
embeddings = OpenAIEmbeddings()

# Create persist directory if it doesn't exist
os.makedirs(persist_directory, exist_ok=True)

# Initialize vector database - either load existing or create new
if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
    # Load existing database for faster startup on subsequent runs
    print("Loading existing vector database...")
    db_books = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("✓ Vector database loaded")
else:
    # First run: create embeddings from raw documents
    print("Creating vector database (this may take a few minutes on first run)...")
    raw_document = TextLoader('../data/raw/tagged_description.txt').load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_document)
    print(f"Processing {len(documents)} document chunks...")
    db_books = Chroma.from_documents(
        documents, 
        embeddings,
        persist_directory=persist_directory
    )
    print("✓ Vector database created and saved")

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    """Retrieve book recommendations using semantic similarity search.
    
    Args:
        query: User search query or book description
        category: Optional book category filter
        tone: Optional emotion/sentiment tone filter
        initial_top_k: Number of candidates to retrieve from vector DB
        final_top_k: Final number of recommendations to return
    
    Returns:
        DataFrame with recommended books sorted by sentiment scores
    """
    # Perform semantic similarity search in the vector database
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    book_list = [int(rec.page_content.split(' ')[0]) for rec, _ in recs]
    book_recs = books[books['isbn'].isin(book_list)].head(final_top_k)

    # Apply category filter if specified
    if category != 'All':
        book_recs = book_recs[book_recs['category'] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort by sentiment tone scores to match user's mood preference
    if tone == 'Happy':
        book_recs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == 'Sad':
        book_recs.sort_values(by='sadness', ascending=False, inplace=True)
    elif tone == 'Angry':
        book_recs.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == 'Suspenseful':
        book_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == 'Surprising':
        book_recs.sort_values(by='surprise', ascending=False, inplace=True)
    else:
        book_recs.sort_values(by='neutral', ascending=False, inplace=True)

    return book_recs

def recommend_books(
        query: str, 
        category: str, 
        tone: str
    ):
    """Format semantic recommendations for display in the UI.
    
    Truncates descriptions, formats author lists, and prepares image/caption pairs.
    """
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        # Truncate description to first 30 words for UI display
        description = row['description']
        truncated_desc_split = description.split()
        truncated_description = ' '.join(truncated_desc_split[:30]) + '...'

        # Format author names properly (handle single, dual, and multiple authors)
        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = ', '.join(authors_split[:-1]) + f", and {authors_split[-1]}"
        else:
            authors_str = row['authors']

        # Create display caption with title, authors, and preview
        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row['cover_image_url'], caption))
    return results
    
# Build list of available categories and sentiment tones from the data
categories = ['All'] + sorted(books['category'].unique())
tones = ['All', 'Happy', 'Sad', 'Angry', 'Suspenseful', 'Surprising', 'Neutral']

# Create Gradio interface with Glass theme
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("## Book Recommender System")
    
    # Input section: query, category, and tone selection
    with gr.Row():
        query_input = gr.Textbox(
            label="Enter a book description or theme", 
            placeholder="e.g. A thrilling mystery set in Victorian London..."
        )
        category_input = gr.Dropdown(label="Select Category", choices=categories, value='All')
        tone_input = gr.Dropdown(label="Select Tone", choices=tones, value='All')
    
    # Action button and results display
    submit_button = gr.Button("Get Recommendations")
    results_gallery = gr.Gallery(label="Recommended Books", columns=4, height="auto")

    # Connect button click to recommendation function
    submit_button.click(
        recommend_books, 
        inputs=[query_input, category_input, tone_input], 
        outputs=results_gallery
    )

# Launch the Gradio app
if __name__ == "__main__":
    dashboard.launch()
