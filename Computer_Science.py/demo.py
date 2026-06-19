import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
st.markdown("""
<style>
  .minimal-header {
        font-family: 'Helvetica', sans-serif;
        letter-spacing: 1px;
        color: #333;
        text-align: center;
        margin-bottom: 30px;
    }
    .custom-input>label {
        font-size: 18px;
        color: #2c3e50;
        font-weight: bold;
    }
    .custom-input>input {
        padding: 12px;
        border-radius: 8px;
        border: 2px solid #4a6fa5;
    }

    .stButton>button {
        background-color: #4a6fa5;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        width: 100%;
    }
    .answer-box {
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
<h2 class='minimal-header'>Information About DCS</h2>
""", unsafe_allow_html=True)


os.environ["OPENROUTER_API_KEY"]=os.getenv("OPENROUTER_API_KEY")
upload_file=PyMuPDFLoader("Require/Department of Computer Science.pdf")
docs=upload_file.load()
all_text = " ".join([doc.page_content for doc in docs])
if docs is not None:
    text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
     )
    chunks=text_splitter.split_text(all_text)
    embedding=HuggingFaceEmbeddings()
    vector_store=FAISS.from_texts(
        chunks,
        embedding=embedding
    )        
    User_input = st.text_input(
    "Department Related Question",
    placeholder="What would you like to know?",
    help="Ask anything about the DCS"
     )
    if st.button("Get answer"):
        if User_input:
            search_docs=vector_store.similarity_search(User_input,k=3)
            llm=ChatOpenAI(model="openai/gpt-3.5-turbo",temperature=0)
            prompt_template = """Use the following context to answer the question:
                Context: {context}
                Question: {question}
                Answer:"""
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
                )
            chain = load_qa_chain(llm,chain_type="stuff", prompt=prompt)
            answer = chain.run(input_documents=docs, question=User_input)
            st.subheader("The Answer as:")
            st.write(answer)
        else:
            st.warning("No matching text found in PDF.")





