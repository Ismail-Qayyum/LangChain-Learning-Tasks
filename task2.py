import langchain 
import openai 
import os 
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
load_dotenv()
api_version = os.getenv("API_VERSION")
deployment_name = os.getenv("DEPLOYMENT_NAME")
endpoint_url = os.getenv("ENDPOINT_URL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

pt = PromptTemplate.from_template("Summarize the topic in exactly three sentences. {topic}")

para = 'Artificial intelligence (AI) is a rapidly evolving field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence. These tasks include problem-solving, learning, decision-making, speech recognition, language translation, and visual perception. AI is broadly categorized into two types: narrow AI, which is designed for specific tasks like voice assistants or recommendation systems, and general AI, which aims to perform any intellectual task a human can do. In recent years, advances in machine learning, deep learning, and neural networks have significantly accelerated the capabilities of AI systems. Today, AI is integrated into everyday applications—from virtual assistants like Siri and Alexa to autonomous vehicles and medical diagnosis tools. Industries such as healthcare, finance, manufacturing, and education are leveraging AI to increase efficiency, reduce costs, and improve decision-making. Despite its benefits, AI also poses ethical and societal challenges. Concerns about data privacy, algorithmic bias, and job displacement are increasingly under discussion. It is essential for developers, businesses, and governments to implement responsible AI practices that prioritize transparency, fairness, and accountability. As AI continues to evolve, its potential to transform nearly every aspect of human life is immense, making it one of the most important technologies of the 21st century.'

llm = AzureChatOpenAI(
    azure_endpoint=endpoint_url,
    api_key=api_key,
    azure_deployment=deployment_name, 
    api_version=api_version,  
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    
)
pt2 = PromptTemplate.from_template("Summarize the topic in exactly one sentence. {topic}")

chain = pt | llm 
chain2 = pt2 | llm 
result = chain.invoke(para)
result2 = chain2.invoke(result.content)

if __name__ == "__main__":

    print(f' 3 line summarization:\n{result.content}')
    print(f' \n1 line summarization:\n{result2.content}')
