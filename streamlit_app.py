import time

import streamlit as st
from huggingface_hub import login

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from streamlit_extras.add_vertical_space import add_vertical_space

from functions import download_data, generate_answer


def streamlit_config() -> None:
    """
    Configures the Streamlit app for the Resume Analyzer AI.
    """
    st.set_page_config(page_title="AI system", layout="wide")

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    st.markdown(
        '<h1 style="text-align: center;">AI-Powered airplain damage detection',
        unsafe_allow_html=True,
    )


# Streamlit Configuration Setup
def initialize_session_state():
    if "use_openai" not in st.session_state:
        st.session_state.use_openai = False
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = None
    if "analyze_resume_full" not in st.session_state:
        st.session_state.analyze_resume_full = None
        st.session_state.suggested_jobs = None
    if "scrap_vacancy" not in st.session_state:
        st.session_state.scrap_vacancy = None


def initialize_llm(use_openai=False, openai_api_key=None):
    if use_openai and openai_api_key:
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo-0125",
            temperature=0.2,
        )
        model_name = "OpenAI GPT"
    else:
        download_data()
        MODEL_CONTEXT_WINDOW = 8192
        llm = LlamaCpp(
            model_path="models/mistral-7b-v0.1.Q4_K_M.gguf",
            n_ctx=MODEL_CONTEXT_WINDOW,
            temperature=0.2,
            verbose=True,
        )
        model_name = "Local LlamaCpp (Mistral 7B)"
    return llm, model_name


def sidebar_model_selection():
    with st.sidebar:
        add_vertical_space(4)
        st.header("Model Selection")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Open Source Model",
                type="primary" if not st.session_state.use_openai else "secondary",
            ):
                st.session_state.use_openai = False
        with col2:
            if st.button(
                "OpenAI GPT",
                type="primary" if st.session_state.use_openai else "secondary",
            ):
                st.session_state.use_openai = True

        if st.session_state.use_openai:
            openai_api_key = st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                value=st.session_state.openai_api_key or "",
            )
            if not openai_api_key:
                st.warning("Please enter your OpenAI API key to use GPT model.")
            elif openai_api_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = openai_api_key
        else:
            st.session_state.openai_api_key = None

    # Check if the model needs to be initialized or updated
    if st.session_state.llm is None or st.session_state.use_openai != (
        st.session_state.model_name == "OpenAI GPT"
    ):
        st.session_state.llm, st.session_state.model_name = initialize_llm(
            st.session_state.use_openai, st.session_state.openai_api_key
        )
        st.sidebar.success(f"Switched to {st.session_state.model_name}")

    st.sidebar.write(f"Current model: {st.session_state.model_name}")


def main():
    streamlit_config()
    # login(token="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")
    initialize_session_state()
    sidebar_model_selection()
    # if "llm" not in st.session_state:
    #     download_data()
    #     st.warning(f"Initializing LLM...")
    #     local_llm = "models/mistral-7b-v0.1.Q4_K_M.gguf"
    #     st.session_state.llm = LlamaCpp(
    #         model_path=local_llm, temperature=0.2, max_tokens=2048, top_p=1
    #     )
    #     st.success("LLM Initialized!")
    st.session_state.embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Logos
    logos = {
        "Mistral AI": "https://plugins.matomo.org/MistralAI/images/5.6.3/A_Mistral_AI_logo.png?w=1024",
        "Transformers": "https://miro.medium.com/v2/resize:fit:631/0*ewH4dvb8djn6KenV.png",
        "Qdrant": "https://qdrant.tech/images/logo_with_text.png",
        "LangChain": "https://deepsense.ai/wp-content/uploads/2023/10/LangChain-announces-partnership-with-deepsense.jpeg",
        "Llama CPP": "https://repository-images.githubusercontent.com/612354784/c59e3320-a236-4182-941f-ea3f1a0f50e7",
        "Docker": "https://blog.codewithdan.com/wp-content/uploads/2023/06/Docker-Logo.png",
    }

    # Display logos in a single row with white background
    st.markdown(
        """
    <table style='border-collapse: collapse; width: 100%; padding-top: 20px; padding-bottom: 20px;'>
    <tr>
        <td style='text-align: center; border: 0; padding: 10px 0;'>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style='background-color: white; display: inline-block; padding: 10px;'>
        <img src="{logos["Mistral AI"]}" alt="{"Mistral AI"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
        <img src="{logos["Transformers"]}" alt="{"Transformers"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
        <img src="{logos["Qdrant"]}" alt="{"Qdrant"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
        <img src="{logos["LangChain"]}" alt="{"LangChain"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
        <img src="{logos["Llama CPP"]}" alt="{"Llama CPP"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
        <img src="{logos["Docker"]}" alt="{"Docker"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        </td>
    </tr>
    </table>
    """,
        unsafe_allow_html=True,
    )

    # Title
    st.markdown(
        """
    <h1 style='text-align:center; color:#007bff; padding-top: 20px;'>Chatbot for consulting</h1>
    """,
        unsafe_allow_html=True,
    )

    # User input
    user_input = st.text_area("Ask your medical question")

    if st.button("Get Answer"):
        with st.spinner("Processing..."):
            start_time = time.time()
            response_data = generate_answer(
                user_input, st.session_state.llm, st.session_state.embeddings
            )
            end_time = time.time()

        if response_data:
            st.markdown(f"**Answer:** {response_data['answer']}")
            # st.markdown(f"**Context:**\n")
            # for i, r in enumerate(response_data['source_documents']):
            #     st.markdown(f'Document #{i}')
            # # st.markdown(f"**Source:**")
            for doc, page in zip(response_data['docs'].split('\n'), response_data['page_numbers']):
                print(f"- {doc} (Page {page})")
            st.markdown(f"**Time taken:** {end_time - start_time:.2f} seconds")
        else:
            st.error("Error processing your request")


if __name__ == "__main__":
    main()
