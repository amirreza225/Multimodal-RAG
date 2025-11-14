## Adapted from streamlit tutorial. Refrence link below:
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps)

import streamlit as st
import os
import base64
import tempfile
import uuid
import time
import gc
import nest_asyncio 
nest_asyncio.apply()

from src.utils import convert_pdf_to_markdown
from src.chunk_embed import chunk_markdown, EmbedData, save_embeddings, load_embeddings
from src.index import QdrantVDB
from src.retriever import Retriever
from src.rag_engine import RAG
from llama_index.core import Settings

# Configurazioni della pagina
st.set_page_config(
    page_title="Exam Trainer Agent",
    page_icon="./images/logo1.png"
)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

# Function to display the uploaded PDF in the app
def display_pdf(file):
    st.markdown("### 📄 PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)


# Sidebar: Upload Document
with st.sidebar:
    st.image("./images/cluster_reply.png")

    st.markdown("<h1 style='text-align: center;'> Use Exam Trainer Agent to test yourself</h1>", unsafe_allow_html=True)
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

    st.header("🤖 LLM Provider")
    llm_provider = st.radio(
        "Select LLM provider",
        options=["Ollama (Local)", "Azure OpenAI (Cloud)"],
        index=0,
        label_visibility="collapsed"
    )

    # Convert display name to backend value
    provider_value = "ollama" if "Ollama" in llm_provider else "azure"

    # Initialize provider in session state
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = provider_value

    # Model selection based on provider
    if provider_value == "ollama":
        st.subheader("Ollama Model")
        ollama_model = st.selectbox(
            "Select Ollama model",
            options=["llama3.2", "mistral", "phi3", "gemma3:1b", "qwen2.5"],
            index=0,
            label_visibility="collapsed"
        )
        if "ollama_model" not in st.session_state:
            st.session_state.ollama_model = ollama_model
        if st.session_state.ollama_model != ollama_model:
            st.session_state.ollama_model = ollama_model
            st.session_state.model_changed = True
    else:  # azure
        st.subheader("Azure Model")
        azure_model = st.text_input(
            "Azure deployment name",
            value="gpt-5",
            label_visibility="collapsed"
        )
        if "azure_model" not in st.session_state:
            st.session_state.azure_model = azure_model
        if st.session_state.azure_model != azure_model:
            st.session_state.azure_model = azure_model
            st.session_state.model_changed = True

    # Detect provider change
    if st.session_state.llm_provider != provider_value:
        st.session_state.llm_provider = provider_value
        st.session_state.provider_changed = True

    st.header("Question Difficulty")
    difficulty = st.select_slider(
        "Select difficulty level",
        options=["Easy", "Medium", "Hard"],
        value="Medium",
        label_visibility="collapsed"
    )

    # Store difficulty in session state
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = difficulty

    # Detect difficulty change
    if st.session_state.difficulty != difficulty:
        st.session_state.difficulty = difficulty
        st.session_state.difficulty_changed = True

    # Memory Chat History Controls
    st.header("💬 Chat Memory")

    # Display conversation history status
    if st.session_state.get("rag") is not None:
        history_count = len(st.session_state.rag.conversation_history) // 2
        max_turns = st.session_state.rag.max_history_turns
        st.info(f"**Memory:** {history_count}/{max_turns} conversation turns stored")
    else:
        st.info("**Memory:** No active session")

    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        if st.session_state.get("rag") is not None:
            st.session_state.rag.clear_history()
            reset_chat()
            st.success("Chat history cleared successfully!")
        else:
            st.warning("No active RAG session to clear.")



    if uploaded_file:
        file_key = f"{session_id}-{uploaded_file.name}"
        if file_key not in st.session_state.file_cache:
            status_placeholder = st.empty()
            status_placeholder.info("📥 File uploaded successfully")
        
            time.sleep(2.5)  # Delay before switching message
            name = uploaded_file.name.rsplit('.', 1)[0]

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                print(f"Temporary file path: {file_path}")
                # Save uploaded file to temp dir
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                status_placeholder.info("Identifying document layout...")
                progress_bar = st.progress(10)

                found = any(
                    os.path.isfile(f) and f.startswith(f"embeddings_{name}" + '.')
                    for f in os.listdir('.')
                )

                if not found:
                    # Convert to markdown
                    markdown_text = convert_pdf_to_markdown(file_path)
                    st.session_state.markdown_text = markdown_text

                    status_placeholder.info("Generating embeddings...")
                    progress_bar.progress(50)
                    
                    chunks = chunk_markdown(markdown_text)
                    st.session_state.chunks = chunks

                    embeddata = EmbedData(batch_size=8)
                    embeddata.embed(chunks)
                    save_embeddings(embeddata, f"embeddings_{name}.pkl")

                    st.session_state.embeddata = embeddata
                
                else:
                    # se avevo già calcolato l'embeddings lo ricarico invece di ricalcolarmelo
                    embeddata = load_embeddings(f"embeddings_{name}.pkl")

                status_placeholder.info("Indexing the document...")
                progress_bar.progress(80)

                database = QdrantVDB(collection_name="MultiMod_collection", vector_dim=len(embeddata.embeddings[0]), batch_size=7)
                database.create_collection()
                database.ingest_data(embeddata)

                st.session_state.database= database

                # After vector DB and embeddata have been defined...
                retriever = Retriever(database, embeddata=embeddata)

                # Set environment variables based on UI selection
                os.environ['LLM_PROVIDER'] = st.session_state.llm_provider
                if st.session_state.llm_provider == 'ollama':
                    os.environ['OLLAMA_MODEL'] = st.session_state.ollama_model
                else:
                    os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] = st.session_state.azure_model

                rag = RAG(retriever, difficulty=st.session_state.difficulty)
                st.session_state.rag = rag
                status_placeholder = st.empty()
                st.success("Ready to Chat...")
                progress_bar.progress(100)
                st.session_state.file_cache[file_key] = True
                
        else:
            st.success("Ready to Chat...")  

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Show message history (preserved across reruns)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle provider/model change - recreate RAG instance
if st.session_state.get("provider_changed", False) or st.session_state.get("model_changed", False):
    if st.session_state.get("rag") is not None and st.session_state.get("database") is not None:
        retriever = Retriever(st.session_state.database, embeddata=st.session_state.embeddata)

        # Update environment variables
        os.environ['LLM_PROVIDER'] = st.session_state.llm_provider
        if st.session_state.llm_provider == 'ollama':
            os.environ['OLLAMA_MODEL'] = st.session_state.ollama_model
            model_display = st.session_state.ollama_model
        else:
            os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] = st.session_state.azure_model
            model_display = st.session_state.azure_model

        # Recreate RAG with new provider/model
        st.session_state.rag = RAG(retriever, difficulty=st.session_state.difficulty)
        reset_chat()
        provider_display = "Ollama (Local)" if st.session_state.llm_provider == "ollama" else "Azure OpenAI (Cloud)"
        st.info(f"Switched to **{provider_display}** with model **{model_display}**. Conversation history cleared.")

    st.session_state.provider_changed = False
    st.session_state.model_changed = False

# Handle difficulty change - reset conversation and update RAG
if st.session_state.get("difficulty_changed", False):
    if st.session_state.get("rag") is not None:
        st.session_state.rag.difficulty = st.session_state.difficulty
        st.session_state.rag.clear_history()  # Use new clear_history() method
        reset_chat()
        st.info(f"Difficulty changed to **{st.session_state.difficulty}**. Conversation history cleared.")
    st.session_state.difficulty_changed = False

# Accept user query
if prompt := st.chat_input("Ask a question..."):

    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate RAG-based response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Thinking..."):

            rag = st.session_state.get("rag")

            if rag is None:
                st.warning("Please upload a PDF to initialize the RAG system first.")
            else:
                response_text = rag.query(prompt)
                message_placeholder.markdown(response_text)



    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_text})