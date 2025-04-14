import langchain 
from task2 import chain, llm , para
from langchain_core.tools import tool 
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage 

load_dotenv()



text='Artificial Intelligence (AI) is revolutionizing the healthcare industry by enhancing diagnostics, treatment, and operational efficiency. AI-powered tools assist doctors in identifying diseases with higher accuracy through advanced imaging analysis, enabling early detection of conditions like cancer or heart disease. Natural language processing allows AI to analyze electronic health records, extract critical insights, and reduce administrative burdens for clinicians. Personalized medicine is another major advancement, where AI tailors treatment plans based on a patient’s genetic makeup, lifestyle, and medical history, improving outcomes and reducing side effects. In hospitals, AI streamlines workflows, manages patient flow, predicts readmissions, and optimizes resource allocation. Virtual health assistants and chatbots enhance patient engagement by providing instant responses and monitoring chronic conditions remotely. Moreover, AI accelerates drug discovery by analyzing vast biological datasets, significantly reducing development time and cost. However, challenges like data privacy, algorithmic bias, and regulatory hurdles persist. Ensuring ethical AI deployment with transparent decision-making is critical to its success. Despite these challenges, AIs role in healthcare continues to grow, offering scalable solutions to pressing global health issues. By augmenting human intelligence, AI empowers healthcare professionals to deliver faster, more accurate, and more personalized care, ultimately transforming the way we diagnose, treat, and manage health.'

@tool 
def TextSummarizer(query:str):
    """ summarizes the given text in exactly 3 sentences"""
    return chain.invoke(str).content

agent = create_react_agent(llm,[TextSummarizer])

response = agent.invoke({"messages": [HumanMessage(content= f"Summarize the following text: {text}")]})
print('\nText Summarization:\n')
print(response['messages'][-1].content)

print('\nSummarizing something interesting:\n')
response2 = agent.invoke({"messages": [HumanMessage(content= f"Summarize something interesting")]})
print(response2['messages'][-1].content)


