import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# No API key needed for local models (Ollama + local embeddings)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Initialize session state for FAISS_DB if it doesn't exist
if 'FAISS_DB' not in st.session_state:
    st.session_state.FAISS_DB = None

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatOllama(
        model="gemma3:1b",
        temperature=0.3,
        num_gpu=1,  # Use GPU
        num_thread=8  # Adjust based on your CPU
    )
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    
    return chain

chain = get_conversational_chain()



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.session_state.FAISS_DB = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)




def user_input(user_question):
    if st.session_state.FAISS_DB is None:
        st.error("Please upload and process PDF files first!")
        return
    
    print("User question received:", user_question)
    docs = st.session_state.FAISS_DB.similarity_search(user_question)
    print(docs)
    print(f"Retrieved {len(docs)} documents from vector store for the question.")
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    

    user_question = st.text_input("Ask a Question from the PDF Files")

    # Debug print to show script execution and current state
    st.sidebar.write("Current session state:", st.session_state)
    print("Script is running - Session State:", dict(st.session_state)) 

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")  



if __name__ == "__main__":
    main()