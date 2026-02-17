# Book Recommender using OpenAI Embeddings and LangChain
 
A full-stack book recommendation system that uses semantic similarity search and sentiment analysis to deliver personalized book recommendations. Built on the Goodreads 100k dataset, the system features a modern Gradio web interface and persistent vector embeddings for fast inference.

## Features

🔍 **Semantic Search**: Find books based on natural language descriptions, not just keywords  
📊 **Sentiment-Based Filtering**: Filter recommendations by mood (happy, sad, angry, suspenseful, surprising, neutral)  
🏷️ **Category Filtering**: Narrow results by book genre  
⚡ **Persistent Vector Database**: Pre-computed embeddings, zero latency on second run  
💻 **Interactive UI**: Modern Gradio interface with book cover gallery display  

## Tech Stack

- **LangChain**: Document processing and embedding orchestration
- **OpenAI Embeddings**: Vector representations of book descriptions
- **Chroma**: Vector database with SQLite persistence
- **Gradio**: User-friendly web interface
- **Pandas**: Data manipulation and filtering
=======
An end-to-end book recommendation project that explores the Goodreads 100k dataset and sets the stage for building an AI-assisted recommender. The current work focuses on data loading, inspection, and cleaning in a Jupyter notebook, with dependencies included for later modeling and app UI work.

## Highlights

- Dataset download and exploration using `kagglehub`.
- Basic data quality checks and missing-value handling.
- Visualization of missing data patterns with `seaborn` and `matplotlib`.

## Project Structure

```
├── dashboard.py                           # Main Gradio application
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── data/
│   ├── raw/                               # Raw input documents
│   │   └── tagged_description.txt
│   ├── processed/                         # Processed dataset files
│   │   ├── sentiment_analyzed_books.csv
│   │   ├── categorized_books.csv
│   │   └── books.csv
│   └── chroma_db/                         # Persisted vector database
│       └── chroma.sqlite3
└── notebook/                              # Analysis and development notebooks
    ├── data_exploration.ipynb
    ├── sentiment_analysis.ipynb
    ├── text_classification.ipynb
    └── vector_search.ipynb
```

## Quick Start

### Prerequisites

- Python 3.13+ (required for chromadb compatibility)
- OpenAI API key (for embeddings)

### Installation

1. **Clone and navigate to the project**
```bash
cd book-recommender
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application

```bash
python dashboard.py
```

The app will automatically:
1. **Check for data**: If required data files don't exist, it prompts you to either:
   - Download the full Goodreads 100k dataset (requires Kaggle API)
   - Generate sample data for demo purposes
2. **First run**: Create vector embeddings from book descriptions (~5-10 minutes, one-time cost)
3. **Subsequent runs**: Load pre-computed embeddings instantly
4. **Launch**: Open your browser to the provided local URL

### Data Setup

The app automatically handles data setup, but you can also run it manually:

```bash
python script/setup_data.py
```

This script will:
- ✅ Check for required data files
- ✅ Download Goodreads dataset (if Kaggle API configured) or offer sample data
- ✅ Process raw data into required formats
- ✅ Create directory structure

## How It Works

### Data Pipeline

1. **Raw Documents**: Book descriptions from Goodreads
2. **Sentiment Analysis**: Each book tagged with emotion scores (joy, sadness, anger, fear, surprise, neutral)
3. **Text Chunking**: Descriptions split into semantic chunks (1000 tokens, no overlap)
4. **Embeddings**: OpenAI's text-embedding model (3-small by default) encodes chunks
5. **Vector Storage**: Chroma stores embeddings with SQLite persistence

### Recommendation Flow

1. User enters search query + selects category and mood
2. Query is embedded using the same model
3. Vector DB returns top-50 semantically similar books
4. Results filtered by category (optional)
5. Results sorted by sentiment scores matching the selected mood
6. Top 16 recommendations displayed with cover images and descriptions

## Development Notebooks

The `notebook/` directory contains Jupyter notebooks for exploration and experimentation:

- **data_exploration.ipynb**: Dataset inspection, schema analysis, and quality checks
- **sentiment_analysis.ipynb**: Sentiment scoring and emotion labeling pipeline
- **text_classification.ipynb**: Category classification and metadata enrichment
- **vector_search.ipynb**: Embedding and vector store experimentation

Run notebooks with:
```bash
jupyter notebook
```

## Configuration

### Environment Variables

```env
OPENAI_API_KEY=sk-...           # Required: OpenAI API key for embeddings
```

### Customization

Edit these variables in `dashboard.py` to customize behavior:

- `persist_directory`: Change vector DB location
- `initial_top_k`: Number of candidates before filtering (default: 50)
- `final_top_k`: Number of final recommendations (default: 16)
- `chunk_size`: Document chunk size in tokens (default: 1000)

## Performance Notes

- **First run**: ~5-10 minutes (depends on document size and API rate limits)
- **Subsequent runs**: <1 second to load vector DB
- **Recommendation latency**: ~2-3 seconds per query (mostly OpenAI API time)
- **Vector DB size**: ~50-100 MB (depends on dataset size)

## Future Enhancements

- [ ] Caching for frequently searched queries
- [ ] User feedback loop to improve recommendations
- [ ] Alternative embedding models (local, faster inference)
- [ ] A/B testing framework for recommendation quality
- [ ] Multi-language support
- [ ] Collaborative filtering integration

## Troubleshooting

**"OPENAI_API_KEY not found"**
- Ensure `.env` file exists in project root with valid API key

**"Data files not found" during first run**
- The app will automatically prompt you to download or generate sample data
- To use the full Goodreads dataset, ensure Kaggle API is configured:
  ```bash
  pip install kagglehub
  # Then create ~/.kaggle/kaggle.json with your Kaggle API credentials from:
  # https://www.kaggle.com/account
  ```
- Or simply choose option 2 to use sample data

**Vector DB creation hangs**
- Check API rate limits; OpenAI may be throttling requests
- Reduce chunk size in `dashboard.py` if memory is constrained

**Gradio app won't launch**
- Ensure port 7860 is available
- Check that all dependencies are installed: `pip install -r requirements.txt`

## License

MIT License - feel free to use for personal or commercial projects

## Dataset Attribution

Goodreads 100k Books Dataset  
Source: https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k


```
├── jupyter_setup.ipynb
├── requirements.txt
└── notebook/
	└── data_exploration.ipynb


```
## Getting Started

### 1) Create and activate a virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```
pip install -r requirements.txt
```

### 3) Launch Jupyter Notebook

```
jupyter notebook
```

Open `notebook/data_exploration.ipynb` and run the cells top to bottom.

## Dataset

The notebook downloads the Goodreads 100k books dataset via `kagglehub`:

- Kaggle dataset: `mdhamani/goodreads-books-100k`

The download path is printed in the notebook so you can inspect the files locally.

## Notebooks

- `notebook/data_exploration.ipynb`: Loads the dataset, inspects schema and missing values, and performs initial cleaning steps.
- `jupyter_setup.ipynb`: Quick guide for setting up Jupyter in a fresh environment.

## Roadmap Ideas

- Build embeddings for book descriptions and metadata.
- Add a retrieval layer (e.g., vector search) for semantic recommendations.
- Create a Gradio UI for interactive recommendations.
- Add evaluation metrics and offline validation.

## Notes

If you plan to use `langchain-openai` or other API-backed features, create a `.env` file and provide the required API keys before running those parts.
