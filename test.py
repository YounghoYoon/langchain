import streamlit as st
import io
import pandas as pd
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(page_title="DirChat", page_icon=":books:")
    st.title("_Private Data :red[QA Chat]_ :books:")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "system", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'pptx', 'csv'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.warning("Please add your OpenAI API key to continue.")
            st.stop()
        if not uploaded_files:
            st.warning("Please upload at least one document.")
            st.stop()

        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorestore, openai_api_key)
        st.session_state.processComplete = True

    for message in st.session_state.messages:
        st.write(f"{message['role']}: {message['content']}")

    if st.session_state.processComplete:
        query = st.text_input("질문을 입력해주세요.")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})

            if st.session_state.conversation:
                with st.spinner("Thinking..."):
                    result = st.session_state.conversation({"question": query})
                    response = result['answer']
                    source_documents = result['source_documents']

                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(f"- **Source**: {doc.metadata['source']}")
                            st.markdown(f"```{doc.page_content}```")

def get_text_from_csv(file_buffer):
    df = pd.read_csv(file_buffer)
    text_data = ' '.join(df.astype(str).sum())
    return text_data

def get_text(docs):
    doc_list = []
    for doc in docs:
        if doc.type == "application/pdf":
            loader = PyPDFLoader(io.BytesIO(doc.getbuffer()))
            documents = loader.load_and_split()
            doc_list.extend(documents)
        elif doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(io.BytesIO(doc.getbuffer()))
            documents = loader.load_and_split()
            doc_list.extend(documents)
        elif doc.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            loader = UnstructuredPowerPointLoader(io.BytesIO(doc.getbuffer()))
            documents = loader.load_and_split()
            doc_list.extend(documents)
        elif doc.type == "text/csv":
            text_data = get_text_from_csv(io.BytesIO(doc.getbuffer()))
            doc_list.append(text_data)
        else:
            continue
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="retrieval",
        retriever=vectorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

if __name__ == '__main__':
    main()