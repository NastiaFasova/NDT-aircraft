# NDT-aircraft
This is a RAG implementation using Open Source stack. BioMistral 7B has been used to build this app along with PubMedBert as an embedding model, Qdrant as a self hosted Vector DB, and Langchai; Llama CPP as an orchestration frameworks.

To run app:
1. Download "BioMistral-7B.Q4_K_M.gguf" from HuggingFace [MaziyarPanahi/BioMistral-7B-GGUF](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF) to the folder "models".
2. Create virtual environment: `python -m venv .venv`.
3. Actiavate env: `.\.venv\Scripts\activate` (on Windows) or `source .venv\bin\activate` (on Linux).
4. Install all requirements: `python install -r requirements.txt`.
5. Install Qdrant with Docker. First of all - install Docker, then run `docker pull qdrant/qdrant` and then `docker run -p 6333:6333 qdrant/qdrant`. It will create Qdrant container, you may observe it by this url: http://localhost:6333/dashboard.
6. Create vector representation of data by using next command: `python ingest.py`. It will take some time to convert data into vectors. After the process will be finished - you will see "vector_db" database in Collections of Qdrant. By default (using only files provide with this project) - there will be 542 points_count.
7. You may start app by running `uvicorn app:app` and open it in your browser.# NDT-aircraft
