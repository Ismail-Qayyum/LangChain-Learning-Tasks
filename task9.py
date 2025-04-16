from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os 
from langchain_core.vectorstores import InMemoryVectorStore
from task2 import chain,llm 
from langchain.retrievers.multi_query import MultiQueryRetriever 
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
api_version = os.getenv("API_VERSION")
deployment_name = os.getenv("DEPLOYMENT_NAME")
endpoint_url = os.getenv("ENDPOINT_URL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

loader = TextLoader("ai_intro.txt", encoding="utf-8")  
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(docs)


embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION")
)

vectordb = InMemoryVectorStore.from_documents(
    splits,
    embedding=embeddings,
)

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

question = "Ai Advancements"
llm = llm 
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)


#Set logging for the queries
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
print('\n================================ Multiple Queries Generated ===============================')

unique_docs = retriever_from_llm.invoke(question)
print(len(unique_docs))
# print('\n1st Query Retrievel:\n',unique_docs[0].page_content)
# print('\n2nd Query Retrievel:\n',unique_docs[1].page_content)
# print('\n3rd Query Retrievel:\n',unique_docs[2].page_content)

# # 

# Checking the length of the retrieved documents
print(f"Total Retrieved Documents: {len(unique_docs)}\n")

# Print out the retrieved documents' content
total_text=' '
for i, doc in enumerate(unique_docs, 1):
    print(f"\nRetrieval {i}:\n{doc.page_content}")
    total_text= total_text + doc.page_content

print('\nTotal Retrieved Text: ',total_text)

response = chain.invoke(total_text)
print('\nSummarized Total Retrieved Text: ',response.content)


retriever = vectordb.as_retriever()
retrieved_documents = retriever.invoke(response.content) #sending summary 
print(f'\nRetrived text from vector store for summary:\n{retrieved_documents[0].page_content}')
