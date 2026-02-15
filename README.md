# AI-Driven Book Recommender

An end-to-end book recommendation project that explores the Goodreads 100k dataset and sets the stage for building an AI-assisted recommender. The current work focuses on data loading, inspection, and cleaning in a Jupyter notebook, with dependencies included for later modeling and app UI work.

## Highlights

- Dataset download and exploration using `kagglehub`.
- Basic data quality checks and missing-value handling.
- Visualization of missing data patterns with `seaborn` and `matplotlib`.

## Project Structure

```
.
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
