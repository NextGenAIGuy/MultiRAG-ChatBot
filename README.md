# MultiRAG-ChatBot
# AI Assistant for MultiDatabase

This project is an AI-powered chatbot capable of interacting with multiple data , including images, grocery datasets, PDFs, and standard queries. The assistant utilizes vector databases, advanced embedding models, and a robust LLM for effective responses.

---

## Key Features

- **Image Dataset Search**: Perform semantic searches over an image dataset stored in a vector database.
- **Grocery Data Interaction**: Query a grocery dataset and retrieve details like prices, categories, and nutritional values.
- **Fine-tuning Information**: Answer questions using PDF data embedded in a vector database.
- **General Assistance**: Handle everyday queries and offer helpful responses.

---



## Installation Guide

### 1. Clone the Repository

Clone the project to your local machine:
```bash
git clone https://github.com/NextGenAIGuy/MultiRAG-ChatBot.git
cd MultiRAG-ChatBot
```

### 2. Create conda environment
```bash
conda create -n multirag -y
conda activate multirag
```

### 3. Install Dependency
```bash
pip insall -r requirements.txt
```
### 4. Run the application
```bash
streamlit run app.py
```


