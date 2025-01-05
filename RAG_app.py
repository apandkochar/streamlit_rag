import os 
import io
import streamlit as st
import tempfile
import zipfile
from langchain.schema import Document
from langchain_community.vectorstores import Chroma , FAISS
from langchain_community.document_loaders import PyPDFLoader , PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.retrievers import EnsembleRetriever 
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain_google_genai.llms import GoogleGenerativeAI 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough

def load_single_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def load_pdf_folder(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    return loader.load()

def document_processing(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
    processed_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        processed_documents.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks])
    return processed_documents

def hf_embeddings():
    model = 'BAAI/bge-small-en-v1.5',
    embeddings = HuggingFaceEmbeddings(
        model_name = model,
        model_kwargs = {'device':'cpu'}
    )
    return embeddings

def google_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
    return embeddings

def setup_retriever(document , embeddings):
    vectorstore = FAISS.from_documents(document , embeddings)
    retriever1 = vectorstore.as_retriever(search_kwargs = {'k':3})
    # vectorstore2 = Chroma.from_documents(document , embeddings)
    # retriever2 = vectorstore2.as_retriever(search_kwargs = {'k':3})
    # retriever = EnsembleRetriever(retrievers=[retriever1 , retriever2] , weights=[0.5 , 0.5] )
    return retriever1
    
def main():
    st.title("RAG APP")
    upload_options = st.selectbox('upload options', ['upload a PDF' , 'upload a PDF Folder'])
    document = None
    if upload_options == 'upload a PDF':
        file_upload = st.file_uploader("Upload your pdf file here:" , type='pdf')
        if file_upload : 
            with tempfile.NamedTemporaryFile(suffix= '.pdf' , delete= False) as temp_file:
                temp_file.write(file_upload.read())
                document = load_single_pdf(temp_file.name)
                document = document_processing(document)
    elif upload_options == 'upload a PDF Folder':
        folder_upload = st.file_uploader("Upload the pdf folder:" , type= 'ZIP')
        if folder_upload:
            tmp_folder = '/tmp/odf_folder'
            os.makedirs(tmp_folder , exist_ok = True)
            
            with zipfile.ZipFile(folder_upload , 'r') as zfile:
                zfile.extractall(tmp_folder)
                document = load_pdf_folder(tmp_folder)
                document = document_processing(document)

                
    if document:
        st.write(f"Loaded {len(document)} documents.")
        embeddings = google_embeddings()
        retriever = setup_retriever(document, embeddings)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)

        # prompt_template = ChatPromptTemplate.from_messages([
        #     ("system", "Provide detailed responses based on context."),
        #     ("human", "{input}")
        # ])
        template="""
        <|system|>>

        You are a very helpful AI Assistant that follows given instruction extreamly well . Use the  context which is given in the documents to give the answers of the user's question

        think very carefully before answering  the question . If  you give the most accurate answer you will get a $100 tip ok!

        CONTEXT:{context}
        </s>
        <|user|>
        {query}
        </s>
        <|assistant|>
        """

        prompt = ChatPromptTemplate.from_template(template)
        output_parsers=StrOutputParser()
        chain=(
            {"context":retriever , "query": RunnablePassthrough()}
            |prompt
            |llm
            |output_parsers
        )
        query = st.text_input("Enter your query:")
        if st.button("Get Answer"):
            if query:
                result = chain.invoke(query)
                st.write("### Answer:")
                st.write(result)
            else:
                st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

                
    
    

