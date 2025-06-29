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
ml.myEDAdocs2(pdffile): #no stresaming
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
            f_EDA = ml.myEDAdocs(f)
            existing_doc_db.append(f_EDA)
            ml.save_list_to_pickle(existing_doc_db, db_file_path)
            
    else:
        # NO LLM NEED
        print("\nNo new PDF files detected after adding one.")


# EXAMPLE OF DOC_DB
"""
In [4]: existing_doc_db
Out[4]:
[{'filename': 'AI Just Learned to Code Like a Human And It Changes Everything _ by Rohit Kumar Thakur _ Jun, 2025 _ Medium.pdf',
  'summary': "The passage discusses a breakthrough method called Execution-Guided Code Generation (EG-CFG) that allows an AI model to think like a real developer and run code while writing it, rather than just generating and testing code in batches. This new technique enables the AI to receive instant feedback on its progress, allowing it to adjust its reasoning based on real-world execution results. The method was tested using an open-source model called DeepSeek-V3 and showed significant improvements over previous methods, including a 96.6% accuracy rate for MBPP benchmarks and a 87.2% accuracy rate for HumanEval-ET benchmarks. This breakthrough represents a fundamental shift in AI's ability to reason about its own code, paving the way for more trustworthy and autonomous AI agents capable of building full applications end-to-end.",
  'topics': "The five main topics discussed in the passage are:\n\n1. Execution-Guided Code Generation (EG-CFG): This is a new method that allows an AI model to think like a real developer and run code while writing it, rather than just generating and testing code in batches. This is relevant because it represents a fundamental shift in AI's ability to reason about its own code, paving the way for more trustworthy and autonomous AI agents capable of building full applications end-to-end.\n\n2. Instant Feedback: The method provides instant feedback on the progress of the AI model, allowing it to adjust its reasoning based on real-world execution results. This is relevant because it enables the AI model to learn from its mistakes in real-time, improving its accuracy and efficiency over time.\n\n3. Open-Source Model: The breakthrough was achieved using an open-source model called DeepSeek-V3, which means that this power won't be locked away in the vaults of a few mega-corporations. This is relevant because it democratizes access to AI technology and allows developers, researchers, and startups everywhere to build on this work.\n\n4. New State-of-the-Art Results: The method achieved new State-of-the-Art (SOTA) results on several industry-standard coding benchmarks, including MBPP (Mostly Basic Python Problems) with 96.6% accuracy and HumanEval-ET with 87.2% accuracy. This is relevant because it demonstrates the effectiveness of the new method in practical applications.\n\n5. Fundamental Step Toward Grounding AI in Reality: The breakthrough represents a fundamental step toward grounding AI in reality, as it enables AI models to think about their own code and reason about its correctness, rather than just generating plausible-looking code that may not actually work. This is relevant because it has significant implications for the future of software development and artificial intelligence more broadly.",
  'text': 'AI Just Learned to Code Like a Human And It Changes Everything\nFor years, LLMs wrote code that looked perfect but failed to run. A new real-time feedback loop just fixed that for good.\nPhoto by Solen Feyissa on Unsplash\nHave you ever asked a LLM like ChatGPT to write code for you? It looks perfect. The syntax is clean, the logic seems sound..\nand then you run it.\nIt crashes.\nThis is one of the most frustrating — and dangerous — problems with AI today. LLMs have become incredible at mimicking\nthe look of human code, but they often fail to grasp the fundamental executability. They’re like an actor who has memorized\na script in a foreign language; they can recite the words flawlessly, but they have no idea what they actually mean.\nI’ve been tracking the progress of AI code generation for years, and in this article, I want to talk about a monumental shift.\nA new research paper has just introduced a method that, for the first time, allows an LLM to think like a real developer.\nNo hype. Just the facts on a breakthrough that might finally make autonomous AI programmers a reality.\nWhat’s Wrong With AI Coders and Why Do They Fail?\nTo understand why this new method is such a big deal, we first need to understand why current AI models fail so often.\nMost code-generating AIs, including the most advanced ones from Google and OpenAI, work on a simple, flawed principle:\nGenerate, then Test.\nOpen in app\nThe AI writes an entire block of code: a full function, or even a whole class.. based on your prompt. Then, and only then,\ndoes it (or you) try to run it. If it fails, the AI might try to debug the whole thing and generate a new version.\nImagine a chef trying to cook a five-course meal. But instead of tasting the soup, seasoning the sauce, or checking if the\nsteak is cooked, they prepare everything blindly from start to finish. They only taste the final meal once it’s on the plate. If\nthe soup is too salty, it’s too late. The entire process was a shot in the dark.\nThat’s how AI has been coding. It’s a process based on pattern recognition, not on real-time understanding. It’s guessing what\nworking code looks like, instead of confirming it at every step.\nHuman developers don’t work this way. We write a few lines, we run them, we see what breaks, and we fix it. It’s an\niterative, constant feedback loop.\nUntil now, AI couldn’t do that.\nThe Breakthrough: Execution-Guided Code Generation (EG-CFG)\nYes, you read that right. AIs are now learning to run code while they are writing it.\nResearchers have developed a new method called Execution-Guided Classifier-Free Guidance (EG-CFG). That’s a mouthful,\nso let me break down what it actually does in simple terms.\nInstead of writing code in one big chunk, an LLM using EG-CFG thinks like a chess grandmaster, constantly evaluating its\nnext move.\nHere’s the step-by-step process:\n1. Write a single line of code. Just one.\n2. Pause and Think. Before writing the next line, the AI generates several possible next lines. Think of these as different\npaths it could take.\n3. Run a Quick Test. It takes the existing code, appends each of the possible new lines, and executes these tiny,\nincomplete code snippets against the test cases.\n4. Get Instant Feedback. The AI analyzes the results. Did Path A cause an error? Did Path B get closer to the correct\noutput? This feedback is called an "execution trace": a detailed log of what happened.\n5. Choose the Best Path. Armed with this real-world feedback, the AI discards the paths that failed and intelligently\nchooses the one most likely to succeed. It then writes that line for real and repeats the process.\nThis isn’t just debugging. This is a live, line-by-line feedback loop.\nIt’s like choosing between a map printed last year and a live GPS that instantly reroutes you around traffic. One is static\nguesswork; the other is dynamic, real-time intelligence.\nWhat Does This “AI Thinking” Actually Look Like?\nLet’s make this concrete. In the research paper, the AI is asked to write a Python function to find the first non-repeating\ncharacter in a string (e.g., in “aabc”, the answer is “c”).\nHere’s how an old AI might fail:\nIt might write a loop that incorrectly returns the first character it sees only once, like ‘b’, even though ‘c’ also appears once\nand comes later. It completes the whole function before realizing its logic is flawed for certain test cases.\nHere’s how the new EG-CFG method works:\nThe AI writes the first few lines, setting up a character count.\nIt gets to a critical for loop. Now it pauses. It generates a few possible next steps.\nCandidate A: if count == 1: return character\nCandidate B: if count == 2: return character\nThe AI runs a “mental test” on both. It executes the partial code with the test case “aabc”.\nIt sees that with Candidate A, it would incorrectly return ‘b’ too early. Failure signal.\nIt sees that with Candidate B, it correctly identifies ‘a’ as a duplicate. Success signal.\nBased on this live feedback, it understands the logic needs to be more robust. It learns, in real-time, that simply\nchecking for a count of 1 isn’t enough. It adjusts its path forward.\nThis is the moment where the AI stops being a parrot and starts being a problem-solver. It’s using execution feedback to\nguide its reasoning, just like a human developer does.\nThe Results: Open-Source Just Lapped the Tech Giants\nSo, does this fancy new method actually work? The results are not just good; they are world-class.\nThe researchers tested EG-CFG on several industry-standard coding benchmarks. These are brutally difficult tests\ndesigned to push AI to its limits. And they did it using DeepSeek-V3, an open-source model.\nHere’s how it performed:\nMBPP (Mostly Basic Python Problems): 96.6% accuracy. This beats nearly every other model, including results from\nGPT-4 and Claude 3.5 Sonnet on similar benchmarks.\nHumanEval-ET (Extended Tests): 87.2% accuracy. This is a new State-of-the-Art (SOTA) result, meaning it’s the best\nscore ever recorded on this tough benchmark.\nCodeContests: 58.18% accuracy. This benchmark features competitive programming problems that require complex\nalgorithmic thinking. EG-CFG set another new SOTA, significantly outperforming previous GPT-4-based methods.\nLet that sink in.\nAn open-source model, available to anyone, using a clever new technique, is now outperforming the billion-dollar,\nproprietary models from the biggest names in tech.\nThis isn’t just an incremental improvement. It’s a paradigm shift. It proves that the path to better AI isn’t just about making\nmodels bigger; it’s about making them smarter.\nWhy This Is More Than Just a Win for Coders\nThe arrival of AI that can reason about its own code has massive implications that go far beyond just writing Python\nscripts.\n1. The Era of “Plausible” Code Is Over: We’re entering a new phase of AI, one where models don’t just generate code that\nlooks right, but code that actually works. This shift lays the groundwork for trustworthy, autonomous AI agents that can\nbuild, test, and deploy full applications end-to-end.\n2. Democratization of Power: The fact that this breakthrough was achieved with an open-source model is huge. It means\nthis power won’t be locked away in the vaults of a few mega-corporations. Developers, researchers, and startups\neverywhere can build on this work.\n3. A Blueprint for General Intelligence: This method of generating possibilities, testing them against reality, and using\nfeedback to guide the next step is a core component of human intelligence. We’ve just successfully implemented it in a\nmachine for a complex, logical task. Imagine applying this “generate-test-guide” loop to other fields, like scientific\ndiscovery, engineering design, or medical diagnosis.\nWe’re Watching the Next Chapter of AI\nFor years, we’ve been amazed by AI’s ability to write poetry, create art, and carry conversations. But deep down, we knew\nits grip on hard logic and real-world execution was shaky. It could imagine, but it couldn’t really do.\nThe shift from simply generating code to executing it line-by-line is a fundamental step toward grounding AI in reality. We\nhaven’t just taught the machine to speak the language of code; we’ve taught it how to think in it.\nThe question is no longer if AI can be a reliable programmer. The question is how we adapt when it becomes the best\nprogrammer in the world.\nWhat are your thoughts? Is this the most significant leap for AI in software development you’ve seen yet?',
  'tokens': 1908}]

In [5]: existing_doc_db[0]['summary']
Out[5]: "The passage discusses a breakthrough method called Execution-Guided Code Generation (EG-CFG) that allows an AI model to think like a real developer and run code while writing it, rather than just generating and testing code in batches. This new technique enables the AI to receive instant feedback on its progress, allowing it to adjust its reasoning based on real-world execution results. The method was tested using an open-source model called DeepSeek-V3 and showed significant improvements over previous methods, including a 96.6% accuracy rate for MBPP benchmarks and a 87.2% accuracy rate for HumanEval-ET benchmarks. This breakthrough represents a fundamental shift in AI's ability to reason about its own code, paving the way for more trustworthy and autonomous AI agents capable of building full applications end-to-end."
"""