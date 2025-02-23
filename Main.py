from langchain.llms import HuggingFaceHub
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.document_loaders import DirectoryLoader
import streamlit as st

persist_directory = 'wroks'


# Get the hyperlinks data from main link
def get_links(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.content, "html.parser")
  links = soup.find_all("a")
  return [link["href"] for link in links]

# Add Prefix to a website which can contain multiple hyperlinks
def add_prefix(list_items, prefix):
  new_list = []
  for item in list_items:
    new_list.append(f"{prefix}{item}")
  return new_list

# Function to Load URL as Document
def urlloader(website):
  url=[website]
  loader = UnstructuredURLLoader(urls=url)
  data = loader.load()
  return data

# Split URL to Chunks
def get_text_chunks(data):
    text_splitter = CharacterTextSplitter(chunk_size=2000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_documents(data)
    return chunks

# Store chunks into the vetorstore
def get_vectorstore(chunks):
    vectorstore = Chroma.from_documents(chunks, embedding_function,persist_directory=persist_directory)
    vectorstore.persist()
    return vectorstore

# Load Already created vectorstore
def load_vectorstore(dir):
   persist_directory=dir
   vectorstore = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding_function)
   return vectorstore

# Langchain Qachain with the created vectorstore
def conversation(vectorstore):
   llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo-16k')
   qa_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",
                                       retriever=vectorstore.as_retriever(),
                                       return_source_documents=True)
   
   return qa_chain

# Handel Outputs from LLM
def handle_userinput(user_question):
     response = st.session_state.conversation(user_question)
     st.session_state.chat_history = response['result']
     st.write(user_template.replace(
                  "{{MSG}}", response['query']), unsafe_allow_html=True)
              
     st.write(bot_template.replace(
                  "{{MSG}}", response['result']), unsafe_allow_html=True)
     st.write('\n\nSources:')
     for source in response["source_documents"]:
        st.write(source.metadata['source'],unsafe_allow_html=True)
     for Doc in response["source_documents"]:
        st.write(Doc.page_content,unsafe_allow_html=True)
      
      
def main():
    load_dotenv()
    global embedding_function
    st.set_page_config(page_title="Refer GPT",
                       page_icon=	":mag_right:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Refer GPT :memo::mag:")
    user_question = st.text_input("Ask a question from the website:")
    if user_question:
      handle_userinput(user_question)

    with st.sidebar:
        st.title("Option Menu")
        st.subheader("Enter Website")
        user_url = st.text_input("Website:")

        box = st.checkbox("Use Alredy Created Vectorstore")
        if (box):
              cvectorstore = st.text_input("Enter Name Created Vectorstore:")

              vbox = st.button("Load Verctorstore")
              if vbox:
               with st.spinner("Loading"):

                  embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

                  persist_directory = cvectorstore
                  vectorstore = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding_function)

                  st.session_state.conversation = conversation(vectorstore)
                  st.success("Loaded Vectorstore")

        option = st.selectbox("Select an Embedding",['All MiniLM L6 V2 - Small/Fastest','E5 base - Medium/Fast','E5 large V2 - Large/Slow'])
        if option == 'All MiniLM L6 V2 - Small/Fastest':
           embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
           st.write("Selected Embedding (Small)")

        elif option == 'E5 base - Medium/Fast':
           embedding_function = HuggingFaceInstructEmbeddings(model_name="intfloat/e5-base")
           st.write("Selected Embedding (Medium)")

        else:
           embedding_function = HuggingFaceInstructEmbeddings(model_name="intfloat/e5-large-v2")
           st.write("Selected Embedding (Large)") 

        if st.button("Process"):
            with st.spinner("Processing"):
              

              # get URL to text 
              url = urlloader(user_url)

              # get the text chunks
              chunks = get_text_chunks(url) 

              # create vector store
              vectorstore = get_vectorstore(chunks)
                 

              # create conversation chain
              st.session_state.conversation = conversation(vectorstore)

if __name__ == "__main__":
   main()
   

