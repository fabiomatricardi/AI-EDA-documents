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

def check_documents_subfolder(parent_folder="."):
    """
    Checks if a subfolder named "documents" exists within the specified parent_folder.

    Args:
        parent_folder (str): The path to the parent folder to check within.
                             Defaults to the current directory (".").

    Returns:
        bool: True if the "documents" subfolder exists, False otherwise.
    """
    documents_path = os.path.join(parent_folder, "documents")
    return os.path.isdir(documents_path)
    
def check_index_pkl_in_documents(parent_folder="."):
    """
    Checks if a file named "index.pkl" exists within the "documents" subfolder
    of the specified parent_folder.

    Args:
        parent_folder (str): The path to the parent folder where the "documents"
                             subfolder is expected. Defaults to the current directory (".").

    Returns:
        bool: True if "index.pkl" exists in the "documents" subfolder, False otherwise.
    """
    documents_path = os.path.join(parent_folder, "documents")
    index_file_path = os.path.join(documents_path, "index.pkl")

    # First, ensure the 'documents' subfolder itself exists
    if not os.path.isdir(documents_path):
        return False

    # Then, check if the 'index.pkl' file exists within it
    return os.path.isfile(index_file_path)    
    

def save_list_to_pickle(data_list, filename):
    """
    Saves a Python list to a pickle file.

    Args:
        data_list: The list to be saved.
        filename: The name of the pickle file (e.g., "my_list.pkl").
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data_list, f)
        print(f"List successfully saved to '{filename}'")
    except Exception as e:
        print(f"Error saving list to pickle file: {e}")

def load_list_from_pickle(filename):
    """
    Loads a Python list from a pickle file.

    Args:
        filename: The name of the pickle file to load.

    Returns:
        The loaded list, or None if an error occurred.
    """
    try:
        with open(filename, 'rb') as f:
            loaded_list = pickle.load(f)
        print(f"List successfully loaded from '{filename}'")
        return loaded_list
    except Exception as e:
        print(f"Error loading list from pickle file: {e}")
        return None


def list_pdfs_and_save_index(parent_folder="."):
    """
    Lists all PDF files in the 'documents' subfolder of the specified parent_folder,
    and saves this list (with full paths) to an 'index.pkl' file within
    the 'documents' subfolder.
    Always returns the list of new PDF files found (could be empty if none).

    Args:
        parent_folder (str): The path to the parent folder.
                             Defaults to the current directory (".").

    Returns:
        tuple: A tuple containing:
               - bool: True if the operation was successful, False otherwise.
               - list: A list of new PDF files found. This will be empty if no new
                       files are detected or if an error occurs.
    """
    documents_path = os.path.join(parent_folder, "documents")
    index_file_path = os.path.join(documents_path, "index.pkl")
    current_pdf_files_list = []
    new_pdf_files = [] # Initialize new_pdf_files here

    # 1. Check if the "documents" subfolder exists
    if not os.path.isdir(documents_path):
        print(f"Error: The 'documents' subfolder does not exist at '{documents_path}'.")
        print("Please create the 'documents' folder first or specify a different parent_folder.")
        return False, new_pdf_files # Return empty list on error

    print(f"Searching for PDF files in: {documents_path}")

    # 2. List all files in the "documents" directory
    try:
        for filename in os.listdir(documents_path):
            file_path = os.path.join(documents_path, filename)
            # Check if it's a file and ends with .pdf (case-insensitive)
            if os.path.isfile(file_path) and filename.lower().endswith(".pdf"):
                current_pdf_files_list.append(file_path)

        print(f"Found {len(current_pdf_files_list)} PDF files in current scan.")
        for pdf in current_pdf_files_list:
            print(f"- {os.path.basename(pdf)}")

    except OSError as e:
        print(f"Error accessing directory '{documents_path}': {e}")
        return False, new_pdf_files # Return empty list on error

    # 3. Load existing index.pkl using the helper function
    existing_pdf_files_list = []
    if os.path.exists(index_file_path):
        loaded_list = load_list_from_pickle(index_file_path)
        if loaded_list is not None:
            existing_pdf_files_list = loaded_list
        else:
            print(f"Warning: Could not load existing index from '{index_file_path}'. Treating as empty.")
            # existing_pdf_files_list remains [] if load fails

    # 4. Determine new PDF files
    current_set = set(current_pdf_files_list)
    existing_set = set(existing_pdf_files_list)
    
    new_pdf_files = list(current_set - existing_set) # This list will be empty if no new files

    if new_pdf_files:
        print(f"\nFound {len(new_pdf_files)} new PDF files:")
        for new_pdf in new_pdf_files:
            print(f"- {os.path.basename(new_pdf)}")
    else:
        print("\nNo new PDF files found.")

    # 5. Save the *current* list of PDF files to index.pkl using the helper function
    save_list_to_pickle(current_pdf_files_list, index_file_path)
    
    # Check if saving was successful by trying to load it back (optional, but robust)
    # A simpler check is to assume save_list_to_pickle prints its own error and
    # the function continues to return based on previous logic for file discovery.
    # However, to explicitly tie success to the save operation, we could modify save_list_to_pickle
    # to return a boolean, or check os.path.exists(index_file_path) after saving.
    # For now, we'll assume save_list_to_pickle handles its own errors via print and we proceed.
    
    # The return value for success should reflect the overall process,
    # primarily the scanning and the determination of new files.
    # If the saving itself fails, it's handled by save_list_to_pickle's print,
    # and we still return the new_pdf_files.
    return True, new_pdf_files

def myEDAdocs(pdffile):
    """
    read a pdf file ospath and return a document dict containing:
    filename
    summary
    5 main topics
    text
    tokens
    """
    print(f"Analysing PDF file {os.path.basename(pdffile)}")
    # LOAD PDF and convert to TEXT - RETRIEVAL         
    context, numtokens = rag.PDFtoText(pdffile)   

    # Feed the TEXT the the LLM prompt - AUGMENTED  
    prompt = f"""Read the provided text, and when you finished say "I am ready".
<passage>
{context}
</passage>

"""
    history = [{"role": "user","content": prompt}]
    # Call the API to have the LLM understand your text - GENERATION  
    start = datetime.now()
    response = rag.bot(history)
    history.append(response)
    delta = datetime.now() - start
    print(f"execution time: {delta.total_seconds()} seconds")
    # create summary
    history.append({"role": "user","content": "Write a short summary of the provided text"})
    start = datetime.now()
    summary = rag.bot(history)
    history.append(summary)
    my_summary = summary["content"]
    delta = datetime.now() - start
    print(f"execution time: {delta.total_seconds()} seconds")
    # create TAble of contents
    history.append({"role": "user","content": "Write the five main topics and why they are relevant"})
    start = datetime.now()
    topics = rag.bot(history)
    history.append(topics)
    my_topics = topics["content"]
    delta = datetime.now() - start
    print(f"execution time: {delta.total_seconds()} seconds")
    EDA_data = {
        'filename': os.path.basename(pdffile),
        'summary' : my_summary,
        'topics' : my_topics,
        'text': context,
        'tokens': numtokens
    }
    return EDA_data

def myEDAdocs2(pdffile):
    """
    read a pdf file ospath and return a document dict containing:
    filename
    summary
    5 main topics
    text
    tokens
    """
    print(f"Analysing PDF file {os.path.basename(pdffile)}")
    # LOAD PDF and convert to TEXT - RETRIEVAL         
    context, numtokens = rag.PDFtoText(pdffile)   

    # Feed the TEXT the the LLM prompt - AUGMENTED  
    prompt = f"""Read the provided text, and when you finished say "I am ready".
<passage>
{context}
</passage>

"""
    history = [{"role": "user","content": prompt}]
    # Call the API to have the LLM understand your text - GENERATION  
    start = datetime.now()
    response = rag.bot2(history)
    history.append(response)
    delta = datetime.now() - start
    print(f"execution time: {delta.total_seconds()} seconds")
    # create summary
    history.append({"role": "user","content": "Write a short summary of the provided text"})
    start = datetime.now()
    summary = rag.bot2(history)
    history.append(summary)
    my_summary = summary["content"]
    delta = datetime.now() - start
    print(f"execution time: {delta.total_seconds()} seconds")
    # create TAble of contents
    history.append({"role": "user","content": "Write the five main topics and why they are relevant"})
    start = datetime.now()
    topics = rag.bot2(history)
    history.append(topics)
    my_topics = topics["content"]
    delta = datetime.now() - start
    print(f"execution time: {delta.total_seconds()} seconds")
    EDA_data = {
        'filename': os.path.basename(pdffile),
        'summary' : my_summary,
        'topics' : my_topics,
        'text': context,
        'tokens': numtokens
    }
    return EDA_data
"""
import os
# Assuming the modified list_pdfs_and_save_index function is in the same file or imported

# Example Usage:
if __name__ == "__main__":
    test_folder = "test_parent_folder"
    documents_subfolder = os.path.join(test_folder, "documents")
    
    # Create a dummy 'test_parent_folder/documents' if it doesn't exist
    if not os.path.exists(documents_subfolder):
        os.makedirs(documents_subfolder)
        print(f"Created directory: {documents_subfolder}")

    # --- First Run: No index.pkl yet, all files will be "new" ---
    print("\n--- First Run ---")
    # Create some dummy PDF files
    with open(os.path.join(documents_subfolder, "doc1.pdf"), "w") as f:
        f.write("This is a dummy PDF content.")
    with open(os.path.join(documents_subfolder, "doc2.pdf"), "w") as f:
        f.write("Another dummy PDF.")

    success, new_files = list_pdfs_and_save_index(parent_folder=test_folder)
    if success:
        if new_files:
            print("\nNewly added PDF files on first run:", [os.path.basename(f) for f in new_files])
        else:
            print("\nNo new PDF files detected on first run.")
    else:
        print("First run failed.")

    # --- Second Run: Add a new PDF file ---
    print("\n--- Second Run: Adding a new PDF file ---")
    with open(os.path.join(documents_subfolder, "new_doc.pdf"), "w") as f:
        f.write("This is a newly added PDF.")

    success, new_files = list_pdfs_and_save_index(parent_folder=test_folder)
    if success:
        if new_files:
            print("\nNewly added PDF files after adding one:", [os.path.basename(f) for f in new_files])
        else:
            print("\nNo new PDF files detected after adding one.")
    else:
        print("Second run failed.")

    # --- Third Run: No changes ---
    print("\n--- Third Run: Running again with no changes ---")
    success, new_files = list_pdfs_and_save_index(parent_folder=test_folder)
    if success:
        if new_files:
            print("\nNewly added PDF files on third run:", [os.path.basename(f) for f in new_files])
        else:
            print("\nNo new PDF files detected on third run.")
    else:
        print("Third run failed.")

    # --- Clean up dummy files and folder ---
    print("\n--- Cleaning up ---")
    try:
        for f_name in os.listdir(documents_subfolder):
            os.remove(os.path.join(documents_subfolder, f_name))
        os.rmdir(documents_subfolder)
        os.rmdir(test_folder)
        print("Cleanup complete.")
    except OSError as e:
        print(f"Error during cleanup: {e}")
"""