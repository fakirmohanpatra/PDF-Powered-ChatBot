
import time
import gradio as gr
import ollama
import chromadb
import PyPDF2
import re
import os
from tkinter import filedialog
from fpdf import FPDF
import re

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
                        # links = re.search("(?P<url>https?://[^\s]+)", pdf_text).group("url")
                        # unique_links = set(links)
                        # links_string = '\n'.join(unique_links)
                        # pdf_text = f"{pdf_text}\n<DocumentLink>{links_string}</DocumentLink>"
                documents.append(pdf_text)
 
    extract_from_folder(folder_path)
    return documents

folder_path = r"D:\TechRaiders\Hackathon\WorkingPoC\DataSource"
documents = fetch_training_data_as_string(folder_path)

client = chromadb.Client()
collection = client.create_collection(name="docs",get_or_create= True)

# store each document in a vector embedding database
for i, document in enumerate(documents):
  response = ollama.embeddings(model="mxbai-embed-large", prompt=document)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[document]
  )

# Gradio UI
description = """
<center>
<img src="https://cdn.prod.website-files.com/629a43aecefd2ef05ec8ae82/629a43aecefd2e6e83c8af6c_kongsberg_logo.png" width=80px>
\nThis tool is to provide tailored content that addresses the unique requirements and interests of customers, thereby improving their experience and engagement with our Product Marketing.
</center>
"""

def process_input(message, history, scroll):   
    response = ollama.embeddings(
    prompt=message,
    model="mxbai-embed-large"
    )
    results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
    )

    data = results['documents'][0][0]

    #generate a response based on data received from previous step
    output = ollama.generate(
    model="llama3.1",
    prompt=f"Using this data: {data}. Respond to this prompt: {message}. If the answer is not in the provided context, return 'I don not know'"
    )
    response = output['response']
    global responseForChat
    responseForChat = output['response']

    if "I do not know" in response:
       responseForChat = "Sorry, no related information is found."
       return responseForChat
    else:
       if('<DocumentLink>' in data):
        responseForChat = response + '\n\n Find more at: ' + ''.join(data.split('<DocumentLink>')[1].split('</DocumentLink>')[0])
        return responseForChat
       else:
          return responseForChat

def save_pdf(chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
   
    pdf.multi_cell(0, 10, chat_history)
    print("PDF created successfully!")
 
    file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files","*.pdf")])
    pdf.output(file_path)
 
    return 0

def hide_popup():
    time.sleep(1)
    return gr.update(visible=False)
 
def download_conversion(popup_message):
    print("response")
    save_pdf(responseForChat)
    return show_popup()

def show_popup():
    # This function creates the HTML for the popup and displays it.
    return gr.update(value="""
    <div id="popup-background" style="display: block;">
        <div id="popup-message" style="position: fixed; top: 5%; left: 50%;  background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);">
            <p>✅ PDF saved successfully!</p>
        </div>
    </div>
    """, visible=True)

chat = gr.ChatInterface(
    fn=process_input,
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
)

with gr.Blocks() as demo:
    chat.render()
    download_pdf=gr.Button("Download")
    popup_message = gr.HTML("", visible=False)
    download_pdf.click(download_conversion,popup_message,outputs=popup_message).then(hide_popup, None, popup_message)
demo.launch()



