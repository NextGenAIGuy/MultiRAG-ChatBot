import streamlit as st
import pandas as pd
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv
from prompts import prompt_template_for_question
import os
import re
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_community.vectorstores import Chroma
import chromadb
import warnings


load_dotenv()
warnings.filterwarnings("ignore")


api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-70b-versatile", api_key=api_key)


df = pd.read_csv('Data\csv_data\grocery_data.csv')
df = df[['description', 'categoryName', 'categoryID', 
                  'price', 'nutritions', 'img', 'name']]

chroma_client = chromadb.PersistentClient(path="Vector_database")
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()
image_vdb = chroma_client.get_or_create_collection(
    name="image", embedding_function=CLIP, data_loader=image_loader
)



def search_image_database(query: str, results: int = 4):
    try:
        response = image_vdb.query(
            query_texts=[query], n_results=results, include=['uris', 'distances']
        )
        return response
    except Exception as e:
        st.error(f"Error querying the database: {e}")
        return None

def check_if_data_needed(user_input: str) -> str:
    prompt = """The user has asked: "{user_input}".
    Determine the appropriate response based on the question.
    - If the question is related to **Cloth image datasets**, respond with "Image".
    - If it is related to **grocery store items**, respond with "Grocery".
    - If it is related to **Finetunning LLM Models**, respond with "Finetunning"
    - else, respond with **normal** 

    Output should be **one word**: "Image" or "Grocery" or "Finetunning" or "normal".
    """
    template = PromptTemplate(template=prompt, input_variables=["user_input"])
    chain = template | llm
    decision = chain.invoke({"user_input": user_input}).content.strip().lower()
    print("decision --------------------> ", decision)
    return decision


def get_data_from_csv(user_query: str) -> str:
    prompt = PromptTemplate(template=prompt_template_for_question, input_variables=["user_query"])
    chain = prompt | llm
    output = chain.invoke({"user_query": user_query})
    
    pattern = r"Python Code: ```(.*?)```"
    matches = re.findall(pattern, output.content, re.DOTALL)
    
    if matches:
        try:
            result = eval(matches[0])
            return result.to_json(orient='records')
        except Exception:
            return "Error in retrieving data."
    return "No matching data found."


def respond_to_user(user_input: str):
    decision = check_if_data_needed(user_input)
    
    if decision == "grocery":
        retrieved_data = get_data_from_csv(user_input)
        st.write(f"Retrieved Grocery Data:\n{retrieved_data}")
    elif decision == "image":
        results = search_image_database(user_input, results=4)
        if results:
            st.write("Image Search Results:")
            for result in results['uris']:
                st.image(result)
    elif decision == "finetunning":
        persist_directory = "pdf_vectordb"
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        vectorstore_retriever = vectordb.as_retriever(
                            search_kwargs={
                                "k": 1 })
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                                chain_type="stuff",
                                                retriever=vectorstore_retriever,
                                                return_source_documents=True)

        question = "You are a helpfull AI Assistant. Your Job is to generate output based on the query."
        query = question + " Requirement: " +  user_query
        llm_response = qa_chain(query)
        st.write(llm_response["result"])


    else:
        prompt = PromptTemplate(
            template="You are a helpful Assistant. Respond to the user's query: '{user_input}'. Don't add prducts details from your side. Just Ask the user How can you help with greetings.",
            input_variables=["user_input"]
        )
        chain = prompt | llm
        output = chain.invoke({"user_input": user_input})
        response = output.content
        st.write(response)


st.title("AI Assistant: One chatBot for MultiDataBase")

user_query = st.text_input("Enter your query:")
if st.button("Submit"):
    if user_query:
        respond_to_user(user_query)
    else:
        st.warning("Please enter a query.")
