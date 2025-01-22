import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# Load environment variables
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print("No .env file found. Please check its location.")

# Initialize API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_Valor")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENV_Valor")
OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY_Valor")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the Pinecone index name and specs
index_name = "json"
if index_name not in pc.list_indexes().names():
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region='us-east-1')
    )
index = pc.Index(index_name)

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def get_embeddings(text):
    if not text or not text.strip():
        print("Skipping empty or invalid text.")
        return None
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

# Function to load JSON file
def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

# Function to process data into a unified format
def process_data(data):
    entries = []
    for item in data:
        submission_id = item['submission_id']
        created_utc = item['created_utc']
        title = item.get('title', '')
        selftext = item.get('selftext', '')
        url = item.get('url', '')
        submission_text = f"{title} {selftext}"
        comments = [comment.strip() for comment in item.get('comments', []) if isinstance(comment, str) and comment.strip()]

        entries.append({
            "id": submission_id,
            "created_utc": created_utc,
            "type": "submission",
            "text": submission_text,
            "comments": comments,
            "url": url,
        })
    return entries

# Function to generate embeddings for entries
def generate_embeddings(entries):
    for entry in entries:
        vector = get_embeddings(entry['text'])
        if vector is not None:
            entry['vector'] = vector
        else:
            print(f"Skipping entry with ID: {entry['id']} due to empty text.")
    return entries

# Function to upload data to Pinecone
def upload_to_pinecone(index, entries, batch_size=100):
    valid_entries = [entry for entry in entries if 'vector' in entry]
    for i in range(0, len(valid_entries), batch_size):
        batch = valid_entries[i:i + batch_size]
        pinecone_vectors = [
            (entry['id'], entry['vector'], {k: v for k, v in entry.items() if k != 'vector'})
            for entry in batch
        ]
        index.upsert(vectors=pinecone_vectors)
        print(f"Uploaded batch {i // batch_size + 1} to Pinecone.")
    print("All data uploaded to Pinecone!")

# Main execution
if __name__ == "__main__":
    folder_path = rf"Your File Path Here"

    # Iterate over all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        print(f"Processing file: {json_file}")

        # Load, process, and upload data
        print("Loading JSON data...")
        data = load_json(file_path)

        print("Processing data...")
        entries = process_data(data)

        print("Generating embeddings...")
        entries = generate_embeddings(entries)

        # Remove entries without a valid vector
        entries = [entry for entry in entries if 'vector' in entry]

        print(f"Uploading data to Pinecone from file: {json_file}")
        upload_to_pinecone(index, entries)

    print("All files processed and uploaded to Pinecone!")
