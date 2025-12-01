
import tkinter as tk
from tkinter import filedialog

def select_pdf():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select PDF",
        filetypes=[("PDF Files", "*.pdf")]
    )
    return file_path

pdf_path = select_pdf()
print("Selected PDF:", pdf_path)
# Upload file dialog (small & quick)
uploaded = files.upload()  # click "Choose Files" and select TravelGuide.pdf
# After upload, the file will be in the current working dir
for name in uploaded.keys():
    print("Uploaded:", name)

!pip install pdfplumber
import pdfplumber

# Path to your uploaded file
pdf_path = "TravelGuide.pdf"

# Extract text from all pages
extracted_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"

# Save text to a file (optional)
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

print("✅ Text extraction complete!")
print("Total characters extracted:", len(extracted_text))

!pip install langchain-text-splitters

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load extracted text
with open("extracted_text.txt", "r", encoding="utf-8") as f:
    extracted_text = f.read()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # each chunk ≈ 1000 characters
    chunk_overlap=200, # overlap so context isn’t lost
    length_function=len
)

# Split into chunks
chunks = text_splitter.split_text(extracted_text)

print(f"✅ Split into {len(chunks)} chunks.")
print("\n🔹 Sample chunk:\n")
print(chunks[0][:500])

!pip install openai chromadb tiktoken

from openai import OpenAI
import chromadb

# ✅ Replace YOUR_OPENAI_API_KEY with your real API key
client = OpenAI(api_key="ENTER_YOUR_API_KEY_HERE")

# Initialize Chroma client and create (or reuse) collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="travel_guide")

print("✅ Connected to ChromaDB collection:", collection.name)

import pdfplumber

# Upload your TravelGuide.pdf file in the left-hand panel (Files)
pdf_path = "/content/TravelGuide.pdf"

extracted_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"

with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

print("✅ Text extracted successfully! Characters:", len(extracted_text))

!pip install langchain-text-splitters --quiet

from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("extracted_text.txt", "r", encoding="utf-8") as f:
    extracted_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_text(extracted_text)

print(f"✅ Split into {len(chunks)} chunks")
print("Sample chunk:\n", chunks[0][:500])

from openai import OpenAI
import chromadb

# Initialize clients
client = OpenAI(api_key="ENTER_YOUR_API_KEY_HERE")  # 🔑 replace with your real API key
chroma_client = chromadb.Client()

# Create or connect to a collection
collection = chroma_client.get_or_create_collection(name="travel_guide")

print("✅ Connected to collection: travel_guide")

!pip install PyPDF2 langchain-text-splitters --quiet

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load your TravelGuide.pdf from Google Colab
uploaded = files.upload()

pdf_path = list(uploaded.keys())[0]
print(f"📄 Uploaded file: {pdf_path}")

# Extract text from the PDF
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text()

# Split text into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(f"✅ PDF split into {len(chunks)} chunks.")

from openai import OpenAI
import chromadb

# Initialize clients
client = OpenAI(api_key="ENTER_YOUR_API_KEY_HERE")  # 🔑 Replace with your real key
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="travel_guide")

print("✅ Connected to collection: travel_guide")

# Store embeddings in batches
batch_size = 20
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    print(f"Processing batch {i//batch_size + 1} / {(len(chunks)//batch_size)+1} ...")

    # Generate embeddings
    embeddings = [
        client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding for text in batch
    ]

    # Add to Chroma collection
    collection.add(
        ids=[str(i+j) for j in range(len(batch))],
        embeddings=embeddings,
        documents=batch
    )

print("✅ All embeddings stored successfully!")

from openai import OpenAI
import numpy as np

# Reconnect to collection (just to be sure)
client = OpenAI(api_key="ENTER_YOUR_API_KEY_HERE")  # same API key
collection = chroma_client.get_collection(name="travel_guide")

# Function to query your Travel Guide
def ask_travel_guide(question):
    # Create embedding for the question
    question_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # Search in ChromaDB for similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    # Combine top matching chunks
    context = " ".join(results["documents"][0])

    # Ask GPT to answer based on context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are TravelMateAI, a helpful travel assistant."},
            {"role": "user", "content": f"Answer based on the travel guide:\n\nContext: {context}\n\nQuestion: {question}"}
        ]
    )

    return response.choices[0].message.content.strip()


# 💬 Try it out!
question = "What kind of travel insurance does the guide mention?"
answer = ask_travel_guide(question)
print("🧭 Question:", question)
print("💡 Answer:", answer)

from IPython.display import display, HTML
import ipywidgets as widgets
from openai import OpenAI

# Initialize OpenAI and Chroma again (in case runtime refreshed)
client = OpenAI(api_key="ENTER_YOUR_API_KEY_HERE")
collection = chroma_client.get_collection(name="travel_guide")

# Helper function to get context-based answers
def get_travelmate_response(question):
    try:
        # Generate embedding for the user's question
        q_embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        # Search top results from ChromaDB
        results = collection.query(query_embeddings=[q_embed], n_results=3)
        context = " ".join(results["documents"][0])

        # Generate an answer using the context
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are TravelMateAI, a friendly and smart travel assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# --- UI Section ---
chat_history = widgets.Output()
input_box = widgets.Text(placeholder='Ask TravelMateAI something about your travel guide...')
send_button = widgets.Button(description='Send', button_style='success')

def on_send(_):
    user_input = input_box.value.strip()
    if user_input:
        with chat_history:
            display(HTML(f"<b>👤 You:</b> {user_input}"))
        input_box.value = ""

        answer = get_travelmate_response(user_input)
        with chat_history:
            display(HTML(f"<b>🤖 TravelMateAI:</b> {answer}"))
            display(HTML("<hr>"))

send_button.on_click(on_send)

# Display everything
display(chat_history)
display(widgets.HBox([input_box, send_button]))

