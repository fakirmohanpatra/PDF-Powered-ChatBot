Data input - Texts from PDFs stored in a Folder is being fetched
Chunking & Embedding - "mxbai-embed-large" from Ollama (OpenSource) is used as the embedding model and 
Vector Database - Chroma DB
Simmilarity Search - to Get the context for the current Query passed by the User
Llama 3.1 - Used as the LLM to fetch the results from the trained PDFs in a different Formats
Output Contains Links where more informations can be fetched about the user query
Output then can be downloaded as a PDF
Output can be customized as Marketing content, Social Media Post, Translated to other languages
If the Query does not have info from the trained Data (Odd query), user will see a pre defined Message

APIs are being exposed which can be used to Integrate this application with various other backend and frontend systems

Future scope is to add a multi-modal LLM to include images to the output to generate full fledged contents for real life use.
