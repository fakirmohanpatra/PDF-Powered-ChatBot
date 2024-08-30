import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
 
from fastapi import FastAPI, HTTPException
import ollama
import chromadb
import uvicorn
import PyPDF2
import re
import requests
from collections import deque
import gradio as gr
 
 
app = FastAPI()
# Initialize the cache with a maximum size of 1
recent_queries = deque(maxlen=1)
 
def fetch_training_data_as_string(folder_path: str):
    documents = []
 
    def extract_from_folder(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
 
            if os.path.isdir(file_path):
                extract_from_folder(file_path)
            elif os.path.isfile(file_path) and filename.lower().endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                documents.append(pdf_text)
 
    extract_from_folder(folder_path)
    return documents
 
# Initialize ChromaDB client and collection
client = chromadb.Client()
collection = client.create_collection(name="training_data", get_or_create=True)
 
folder_path = r"D:\TechRaiders\Hackathon\WorkingPoC\DataSource"
documents = fetch_training_data_as_string(folder_path)
 
# Process each document and add to ChromaDB collection
for i, d in enumerate(documents):
    try:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
        embedding = response["embedding"]
 
        url_pattern = r'https?://\S+'
        links = re.findall(url_pattern, d)
        unique_links = set(links)
        links_string = '\n'.join(unique_links)
 
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[f"{d}\n<DocumentLink>{links_string}</DocumentLink>"]
        )
    except Exception as e:
        print(f"Error processing document {i}: {e}")
 
def contains_offensive_content(text: str) -> bool:
    # List of keywords related to racism and religious discrimination (extend this list as needed)
    offensive_keywords = [
        "racist", "n-word", "hate speech", "discrimination",
        "religious intolerance", "blasphemy", "extremist"
    ]
   
    # Convert the text to lowercase for case-insensitive comparison
    text_lower = text.lower()
   
    # Check for the presence of any offensive keywords
    for keyword in offensive_keywords:
        if keyword in text_lower:
            return True
   
    return False
 
def process_input(message, history, scroll):
    try:
        recent_queries.append(message)
        final_prompt = "\n".join(recent_queries)
        message += ".If the answer is not in the provided context, return 'I don not know'"
        response = requests.get("http://localhost:8000/get-semantics-along-with-query", params={"prompt": message})
        response.raise_for_status()
        response_data = response.json()
        cleaned_data = re.sub(r'\s+', ' ', response_data.replace("\n", " ").replace("\r", " "))
        prompt = f"Using this data: {cleaned_data}. Respond to this prompt: {prompt}"
       
        response = ollama.generate(
            model="llama3.1",
            prompt=prompt
        )
        result = response["response"]
 
        # Filter the result
        if contains_offensive_content(result):
            result = "The content contains inappropriate material and has been filtered out. Try another query."
 
        if "I do not know" in result:
            return "Sorry, no related information is found."
        else:
            if('<DocumentLink>' in response):
                return response + '\n\n Find more at: ' + ''.join(response.split('<DocumentLink>')[1].split('</DocumentLink>')[0])
            else:
                return response
 
        # return result
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
 

description = """
<center>
<img src="https://cdn.prod.website-files.com/629a43aecefd2ef05ec8ae82/629a43aecefd2e6e83c8af6c_kongsberg_logo.png" width=80px>
\nThis tool is to provide tailored content that addresses the unique requirements and interests of customers, thereby improving their experience and engagement with our Product Marketing.
</center>
"""
gr.ChatInterface(
    fn = process_input,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask your question!", container=False, scale=7),
    title ="TECH RAIDERS",
    description=description,
    theme=gr.themes.Soft(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink),
    examples=["How to install VI app?", "Where can I find isolation in Kognitwin?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
).launch()


@app.get("/get-all-training-data")
def read_all_training_data():
    return documents
 
@app.get("/get-semantics-along-with-query")
def read_semantics_along_with_query(prompt: str):
    try:
        response = requests.get("http://localhost:8000/get-embeddings", params={"data": prompt})
        response.raise_for_status()
        response_data = response.json()
        embedding = response_data["embedding"]
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1
        )
        return results['documents'][0][0]
   
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@app.get("/get-query-response")
def read_query_response(prompt: str):
    try:
        recent_queries.append(prompt)
        final_prompt = "\n".join(recent_queries)
        final_prompt += ".If the answer is not in the provided context, return 'I don not know'"
        response = requests.get("http://localhost:8000/get-semantics-along-with-query", params={"prompt": final_prompt})
        response.raise_for_status()
        response_data = response.json()
        cleaned_data = re.sub(r'\s+', ' ', response_data.replace("\n", " ").replace("\r", " "))
        prompt = f"Using this data: {cleaned_data}. Respond to this prompt: {prompt}"
       
        response = ollama.generate(
            model="llama3.1",
            prompt=prompt
        )
        result = response["response"]
 
        # Filter the result
        if contains_offensive_content(result):
            result = "The content contains inappropriate material and has been filtered out. Try another query."
 
        if "I do not know" in result:
            return "Sorry, no related information is found."
        else:
            if('<DocumentLink>' in response):
                return response + '\n\n Find more at: ' + ''.join(response.split('<DocumentLink>')[1].split('</DocumentLink>')[0])
            else:
                return response
 
        # return result
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@app.get("/get-embeddings")
def read_embeddings(data: str):
    try:
        response = ollama.embeddings(
            prompt=data,
            model="mxbai-embed-large"
        )
        return {"embedding": response["embedding"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
 
