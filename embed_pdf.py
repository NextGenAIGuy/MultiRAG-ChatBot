from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path 
from langchain.schema import Document  
import PyPDF2

directory = "Data/pdf_data"


def load_pdf_text_pypdf2(file_path):
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  
    return text


documents = []
for pdf_file in Path(directory).glob("*.pdf"):
    text = load_pdf_text_pypdf2(pdf_file)
    documents.append(Document(page_content=text, metadata={"source": str(pdf_file)}))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "pdf_vectordb"
vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)

print("PDF documents embedded and saved.")
