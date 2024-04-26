import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

st.markdown('<h1><span style="color: royalblue;">📚 RAG</span> <span style="color: royalblue;">System</span> <span style="color: royalblue;">on “Leave No Context Behind” Paper</span> <span style="color: royalbllue;">🤖</span></h1>', unsafe_allow_html=True)

# Initialize loaders and embeddings
loader = PyPDFLoader(r"C:\Users\Sampada\Downloads\LangChain_RAG.pdf")
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyAC_2dR1YnoO-tHe5l8ew1z8fXNAhFymXA", 
                                               model="models/embedding-001")

# Load and split the document
data = loader.load_and_split()
text_splitter = NLTKTextSplitter(chunk_size=200, chunk_overlap=200)
chunks = text_splitter.split_documents(data)

# Create and persist the Chroma database
try:
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
except Exception as e:
    st.error(f"Error creating Chroma database: {e}")

# Connect to the persisted Chroma database
try:
    db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
except Exception as e:
    st.error(f"Error connecting to Chroma database: {e}")
    raise e

retriever = db_connection.as_retriever(search_kwargs={"k": 1})

# Define chat template
chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chat_model = ChatGoogleGenerativeAI(google_api_key="AIzaSyAC_2dR1YnoO-tHe5l8ew1z8fXNAhFymXA", 
                                   model="gemini-1.5-pro-latest")

output_parser = StrOutputParser()
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit input and processing
user_input = st.text_input("Ask a question:")
if st.button("Search"):
    # Retrieve context based on the user input
    try:
        retrieved_docs = rag_chain.invoke(user_input)
        st.write(retrieved_docs)
    except Exception as e:
        st.error(f"Error during chain invocation: {e}")
