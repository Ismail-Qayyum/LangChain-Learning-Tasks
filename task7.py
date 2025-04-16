from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
import os 
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from task2 import chain 

load_dotenv()

file_path = r"C:\\Users\\IsmailQayyum\Desktop\\Langchain\\LangChain-Learning-Tasks\\ai-ethics.pdf"
pdf_loader = PyPDFLoader(file_path)
#print(pdf_loader)

pdf_docs = pdf_loader.load()
#print(len(docs))

from langchain_community.document_loaders import WebBaseLoader

web_loader = WebBaseLoader("https://medium.com/@API4AI/top-ai-trends-in-the-sports-industry-for-2025-98758bb89378")
web_docs = web_loader.load()
#print(web_docs[0].page_content)

charac_text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size = 150 , 
    chunk_overlap = 30 , 

)
pdf_chunks = charac_text_splitter.split_documents(pdf_docs) #each chunk[i] holds a sentence
web_chunks = charac_text_splitter.split_documents(web_docs) #each chunk[i] holds a sentence

embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION")
)

pdf_vectorstore = InMemoryVectorStore.from_documents(
   pdf_chunks,
    embedding=embeddings,
)

web_vectorstore = InMemoryVectorStore.from_documents(
   web_chunks,
    embedding=embeddings,
)

pdf_retriever = pdf_vectorstore.as_retriever()
pdf_retrieved_documents = pdf_retriever.invoke("AI Milestones")
print('\n========================================= PDF ===========================================')
print(f'\nRetrived text from vector store:\n{pdf_retrieved_documents[0].page_content}')
pdf_result = chain.invoke(pdf_retrieved_documents[0].page_content)
print(f'\nSummarization of retrieved text:\n{pdf_result.content}')


web_retriever = web_vectorstore.as_retriever()
web_retrieved_documents = web_retriever.invoke("AI Milestones")
print('\n========================================= WEB ===========================================')
print(f'\nRetrived text from vector store:\n{web_retrieved_documents[0].page_content}')
web_result = chain.invoke(web_retrieved_documents[0].page_content)
print(f'\nSummarization of retrieved text:\n{web_result.content}')


print('\n========================================= COMPARISION ===========================================')
print(f'\nPDF SUMMARIZATION:\n{pdf_result.content}')
print(f'\nWEB SUMMARIZATION:\n{web_result.content}')
