import streamlit as st
from back import LLMHandler, VectorDatabase, QuestionAnsweringChain
from dotenv import load_dotenv
import os

# Parameters
load_dotenv()
gemini_key = os.getenv('gemini_key')
rerank=True
rewrite=False
num_docs=5




# Initialize or get existing instances from session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = VectorDatabase(
        model_name="hiieu/halong_embedding",
        collection_name='cmc_corp_full_web',
        db_path='db_qdrant'
    )
if 'llm_handler' not in st.session_state:
    st.session_state.llm_handler = LLMHandler(model_name="gemini-1.5-flash", gemini_key=gemini_key)
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = QuestionAnsweringChain(
        llm_handler=st.session_state.llm_handler,
        vector_db=st.session_state.vector_db,
        num_docs=num_docs,
        apply_rerank=rerank,
        apply_rewrite=rewrite
    )

# Initialize or get existing chat history from session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Streamlit app
st.title("Chat with the CMC AI")

# Display chat history
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.write(f"**You:** {msg['content']}")
    else:
        st.write(f"**AI:** {msg['content']}")

# Input field for user question
question = st.text_input("Enter your message:")

if st.button('Send'):
    if question:
        # Add user message to chat history
        st.session_state.messages.append({'role': 'user', 'content': question})

        # Get AI response
        result = st.session_state.qa_chain.run(question)
        st.write(result)
        # Add AI response to chat history
        st.session_state.messages.append({'role': 'ai', 'content': result})

  
        

