import os
import time
import gdown
import streamlit as st

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from utils import PROMPT_TEMPLATE

from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login


def load_func(file_path_posts, file_url, file_name, output_path, model=False):

    if os.path.exists(file_path_posts):
        st.success(f'"{file_name}" model was loaded')
    elif model:
        st.warning(f'Wait for "{file_name}" model been loaded')
        os.system(
            "huggingface-cli download TheBloke/Mistral-7B-v0.1-GGUF mistral-7b-v0.1.Q4_K_M.gguf --local-dir models --local-dir-use-symlinks False"
        )
        st.success(f"'{file_name}' model was loaded")
    else:
        st.warning(f'Wait for "{file_name}" model been loaded')
        gdown.download(file_url, output_path)
        st.success(f"'{file_name}' model was loaded")


def download_data():
    login(token="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")
    if not os.path.exists("models"):
        os.mkdir("models")
    if not os.path.exists("database"):
        os.mkdir("database")

    file_name_model = "mistral-7b-v0.1.Q4_K_M.gguf"
    file_path_model = "models/" + file_name_model
    file_url_model = "https://drive.google.com/uc?id=1kX4gWcKOTbXRji0twLCkDFqlvijS72j1&export=download"

    load_func(
        file_path_model, file_url_model, file_name_model, file_path_model, model=True
    )


def generate_answer(query: str, llm, embeddings):
    start_time = time.time()  # record the start time

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    db = FAISS.load_local(
        "data/faiss_data", embeddings, allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})  # Retrieve 5 similar documents

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True,
    )

    response = qa({"query": query})

    answer = response["result"]
    source_documents = response["source_documents"]

    # Prepare source information including page numbers
    source_info = []
    for doc in source_documents:
        page_number = doc.metadata.get("page", "Unknown")
        source_file = doc.metadata.get("source", "Unknown")
        source_info.append(
            {"content": doc.page_content, "page": page_number, "source": source_file}
        )

    end_time = time.time()  # record the end time
    elapsed_time = end_time - start_time  # calculate the elapsed time in seconds

    response_data = {
        "answer": answer,
        "sources": source_info,
        "elapsed_time": elapsed_time,
    }

    return response_data
