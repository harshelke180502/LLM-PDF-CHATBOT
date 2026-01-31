import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()



# Set up LangSmith for monitoring
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "llm-pdf-chatbot")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


if 'FAISS_DB' not in st.session_state:
    st.session_state.FAISS_DB = None

@st.cache_resource
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Use OpenAI instead of Ollama
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # or "gpt-4" for better quality
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# chain = get_conversational_chain()



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text




def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    st.session_state.FAISS_DB = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )





@traceable  # LangSmith tracing decorator
def user_input(user_question):
    if st.session_state.FAISS_DB is None:
        st.error("Please upload and process PDF files first!")
        return None
    
    try:
        chain = get_conversational_chain()
        
        print("User question received:", user_question)
        docs = st.session_state.FAISS_DB.similarity_search(user_question, k=4)
        print(f"Retrieved {len(docs)} documents from vector store for the question.")
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        answer = response["output_text"]
        print(response)
        st.write("Reply: ", answer)
        
        # Return the answer so LangSmith can track it
        return answer
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        print(f"Error occurred: {e}")
        return None




def main():
    st.set_page_config(page_title="Chat with PDFs")

    st.header("Chat with PDFs 💁")

    

    user_question = st.text_input("Ask a Question from the PDF Files")

    # # Debug print to show script execution and current state
    # st.sidebar.write("Current session state:", st.session_state)
    # print("Script is running - Session State:", dict(st.session_state)) 

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file!")


if __name__ == "__main__":
    main()