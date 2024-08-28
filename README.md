# Financial Question Answering with LLMs

This repository leverages Large Language Models (LLMs) to answer financial questions based on structured data. The data used for evaluation comes from the [ConvFinQA dataset](https://github.com/czyssrs/ConvFinQA).

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Answer Generation](#answer-generation)
- [Streamlit App](#streamlit-app)
- [Installation](#installation)
- [Environment Variables](#environment-variables)

## Overview

This project utilizes LLMs to generate answers to financial questions based on provided contexts from financial datasets. The data is derived from the ConvFinQA dataset, which includes fields like `pre_text`, `post_text`, `table`, and a pair of question and answer in the `qa` field.

## Data Preprocessing

The data preprocessing step is executed in the `notebooks/preprocess.ipynb` notebook. This script reads the raw data from `local_data/train.json` (not included in the repo) and processes it into a more structured format, saving the output to `data/extracted_data.json`.

## Answer Generation

The core functionality of this repository is encapsulated in the `FinQA` class, which is defined in a helper module. The class includes the `get_answer` method, designed to interact with an LLM agent to produce answers to financial questions based on input data. 

### `get_answer` Method

This method generates an answer to a financial question using the following arguments:

- **pre_text**: Contextual information preceding the table.
- **table**: The financial data presented in a table format.
- **post_text**: Contextual information following the table.
- **question**: The financial question to be answered.

**Returns**: The final answer as a string, derived from the LLM.

This method is used to generate answers for all training data and within a Streamlit app for live demonstrations.

## Streamlit App

The Streamlit application provides an interactive interface with two primary tabs:

1. **Sample Review**: This tab allows users to review sample training data and generate answers using the LLM in real-time.
2. **Accuracy**: This tab displays accuracy metrics for the LLM based on the entire dataset.

## Installation

To set up the project, follow these steps:

1. Clone the repository.
2. Create a virtual environment:
    ```bash
    python -m venv .venv
    ```
3. Activate the virtual environment:
    - On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    - On Windows:
        ```bash
        .venv\Scripts\activate
        ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
5. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Environment Variables

Ensure that your OpenAI API key is set in a `.env` file with the following variable:

```plaintext
OPENAI_API_KEY=your_openai_api_key
```

This `.env` file should be placed in the root directory of the project.
