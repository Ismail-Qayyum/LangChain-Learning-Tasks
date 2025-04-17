from task2 import chain , llm
from task3 import retriever
from langchain_core.tools import tool 
from langgraph.prebuilt import create_react_agent
from task3 import docs 
from langchain_core.messages import HumanMessage 
import datetime 

def mock_web_search_tool(query: str) -> str:
    return (
        "This is a mock web search result for your query. "
        "Recent updates in AI include advancements in generative models, increased regulation interest, and widespread enterprise adoption. "
        "AI agents and multi-modal models are trending. Tech giants are investing heavily. "
        "Real-time AI integration is expanding fast."
    )


@tool 
def retrieve_text(text:str):
    """finds or retrieves the text from the vectorstore"""
    print("[Tool] Retrieving relevant text...")
    retrieved_documents = retriever.invoke(text)
    return retrieved_documents[0].page_content


@tool 
def TextSummarizer(text:str):
    """ summarizes the given text in exactly 3 sentences"""
    print("[Tool] Summarizing text...")
    return chain.invoke(text).content

@tool 
def count_words_in_text(text: str) -> str:
    """Counts the number of words in the input text."""
    print("[Tool] Counting words...")
    return f"Word count: {len(text.split())}"

@tool 
def date_time():
    """Return two variables. The first one is current date and secone one is current time"""
    print('[Tool] Date-Time...')
    x = datetime.datetime.now()
    return x.date(), x.time()

@tool
def mock_web_search(query: str) -> str:
    """Mock tool that simulates a web search with a static 50-word response."""
    print('[Tool] Mock Web Search...')
    return mock_web_search_tool(query)

agent = create_react_agent(llm,[TextSummarizer,count_words_in_text,retrieve_text , date_time , mock_web_search])
response = agent.invoke({"messages": [HumanMessage(content="Find and summarize text about AI breakthroughs from the document , and count the words in the summary. Additionally fetch current date and time and perform a mock web searc. Print the retrieved text , summarized text and word count , date and time and mock web search result")]})

print('\nText Summarization:\n')
print(response['messages'][-1].content)