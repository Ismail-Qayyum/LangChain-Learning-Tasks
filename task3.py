from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os 
from langchain_core.vectorstores import InMemoryVectorStore
from task2 import chain
load_dotenv()
api_version = os.getenv("API_VERSION")
deployment_name = os.getenv("DEPLOYMENT_NAME")
endpoint_url = os.getenv("ENDPOINT_URL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

loader = TextLoader("ai_intro.txt", encoding="utf-8")  
docs = loader.load()

charac_text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size = 200 , 
    chunk_overlap = 20 , 

)
chunks = charac_text_splitter.split_documents(docs) #each chunk[i] holds a sentence

embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION")
)

vectorstore = InMemoryVectorStore.from_documents(
    chunks,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

if __name__ == "__main__": 
    retrieved_documents = retriever.invoke("AI Milestones")
    print(f'\nRetrived text from vector store:\n{retrieved_documents[0].page_content}')
    result = chain.invoke(retrieved_documents[0].page_content)
    print(f'\nSummarization of retrieved text:\n{result.content}')