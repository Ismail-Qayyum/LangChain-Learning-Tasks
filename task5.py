from task2 import chain , llm
from task3 import retriever
from langchain_core.tools import tool 
from langgraph.prebuilt import create_react_agent
from task3 import docs 
from langchain_core.messages import HumanMessage 

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

agent = create_react_agent(llm,[TextSummarizer,count_words_in_text,retrieve_text])
response = agent.invoke({"messages": [HumanMessage(content="Find and summarize text about AI breakthroughs from the document , and count the words in the summary. Print the retrieved text , summarized text and word count")]})

print('\nText Summarization:\n')
print(response['messages'][-1].content)