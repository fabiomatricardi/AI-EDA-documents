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
# rag.countTokens(text)  rag.PDFtoText(pdffile)  rag.bot(messages)

# --- Configuration ---
LLAMA_CPP_SERVER_URL = "http://127.0.0.1:8080"
MODEL_NAME = "Qwen2.5-1.5B-instruct" 
NCTX = 8192  # Example context length, adjust as needed for your model/setup
COUNTERLIMITS = 16 # Reset history after this many turns
# Define stop sequences relevant to your model to prevent run-on responses
STOPS = ['<|im_end|>']

"""
ml.check_documents_subfolder(parent_folder="."):
ml.check_index_pkl_in_documents(parent_folder="."):
ml.save_list_to_pickle(data_list, filename):
ml.load_list_from_pickle(filename):
ml.list_pdfs_and_save_index(parent_folder="."):
ml.myEDAdocs(pdffile):
"""

    


# FIST RUN CHECK
parent_folder="."
documents_path = os.path.join(parent_folder, "documents")
index_file_path = os.path.join(documents_path, "index.pkl")
db_file_path = os.path.join(documents_path, "doc_db.pkl")
first_pdf_files_list = []
print("Checking if a document folder exists...")
if ml.check_documents_subfolder(parent_folder="."):
    print("✅ Good let's move on")
else:
    print("🚫 this is the first run... craeting the document subfolder...")
    test_folder = "."
    documents_subfolder = os.path.join(test_folder, "documents")
    os.makedirs(documents_subfolder)
    print(f"✅ Created directory: {documents_subfolder}")
    print("Generating first empty index list...")
    print("Generating first empty Document db list...")
    ml.save_list_to_pickle(first_pdf_files_list, index_file_path)
    ml.save_list_to_pickle(first_pdf_files_list, db_file_path)
    print(f"✅ Successfully created index file: {index_file_path}")
    print(f"✅ Successfully created Docunent DB file: {db_file_path}")

# Load existing doc_db.pkl using the helper function
existing_doc_db = []
if os.path.exists(db_file_path):
    loaded_db = ml.load_list_from_pickle(db_file_path)
    if loaded_db is not None:
        existing_doc_db = loaded_db      

from tqdm import tqdm
# LIST THE NEW PDF FILES / COMPARE AGAINST THE EXISTING ONES
success, new_files = ml.list_pdfs_and_save_index(parent_folder=".")        
if success:
    if new_files:
        # Use LLM to PROCESS
        print("\nNewly added PDF files:", [os.path.basename(f) for f in new_files])
        for f in tqdm(new_files, desc="Processing new PDF files"):
            print(f"Exploratory Document Analysis on {os.path.basename(f)} ...")
            f_EDA = ml.myEDAdocs2(f)
            existing_doc_db.append(f_EDA)
            ml.save_list_to_pickle(existing_doc_db, db_file_path)
            
    else:
        # NO LLM NEED
        print("\nNo new PDF files detected after adding one.")