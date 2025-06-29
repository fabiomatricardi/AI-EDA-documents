# AI-EDA-documents
LLM call to llama-server to parse new pdf documents into meaningful database

## Python is the best for AI automation

Call llama-server to perform automatic scan of a folder `documents`: if there are new pdf files it will start to 
explore the pdf as follows"
- open a pdf
- transform it into text
- extract the summary
- extract the 5 most relevant topics
A local database in python `pickle` will contain all these information for further GUI interactions or Q&A sessions.


### Requirements
```bash
pip install requests rich langchain-text-splitters tiktoken pypdf easygui
```
Then we need
- [llama.cpp binaries (version b5686)](https://github.com/ggml-org/llama.cpp/releases/download/b5686/llama-b5686-bin-win-cpu-x64.zip), to download and extract the ZIP file in your project directory (mine is called `localFolderSCAN`)
- [Qwen2.5-1.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q6_k.gguf?download=true): a good small LLM with great context window and amazing performance. Download it in the same project folder.
- a PDF document as example: download it in the same directory (from this repo, called AIgentsFail.pdf)

### Run the llama.cpp server
First thing, let’s make sure our LocalGPT engine is running. Open one terminal window in the BETTER_RAG folder and execute:
```bash
llama-server.exe -m .\qwen2.5-1.5b-instruct-q6_k.gguf -c 8192
```
This will start the server and listen the API endpoints at  http://127.0.0.1:8080 with a context window of 8k tokens.

Run from the terminal `python test_mylib3_noStream.py` 

In the first run it will create:
- a subfolder called `documents`
- `doc_db.pkl` containing a list of dict with rich information about every pdf added to the folder
- `index.pkl` the list of pdf files to spot new entries and start the process again.

### local libs
I created 2 different local libraries
- RAGLIB.py
- mydoclib.py

The first one is used for LLM calls with json and requests
The second one handles the file operations and the Exploratory Document Analysis (EDA)


### How to browse the `doc_db.pkl`
For now I have not prepared yet a gradio GUI
You can use ipython to explore the db
```bash
pip install ipython
```
from the terminal run
```bash
ipython
```
And then use the following:
```python
import mydoclib as ml
import pickle
import os
import warnings
warnings.filterwarnings(action='ignore')
import requests
import sys
import json
import pypdf
import tiktoken
from datetime import datetime
import RAGLIB as rag

parent_folder="."
documents_path = os.path.join(parent_folder, "documents")
index_file_path = os.path.join(documents_path, "index.pkl")
db_file_path = os.path.join(documents_path, "doc_db.pkl")
first_pdf_files_list = []

# Load existing doc_db.pkl using the helper function
existing_doc_db = []
if os.path.exists(db_file_path):
    loaded_db = ml.load_list_from_pickle(db_file_path)
    if loaded_db is not None:
        existing_doc_db = loaded_db

len(existing_doc_db)
existing_doc_db[0]['filename']  #filename of the PDF file
existing_doc_db[0]['summary']   #summary generated with Qwen2.5-1.5b
existing_doc_db[0]['topics']    #topics extracted with Qwen2.5-1.5b
existing_doc_db[0]['tokens']    #tokens counted with tiktoken


```

---




