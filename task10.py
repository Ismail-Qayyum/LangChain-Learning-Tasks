from task2 import llm, chain  
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate

loader = TextLoader("ai_intro.txt", encoding="utf-8")  
docs = loader.load()

summarized_ouput = chain.invoke(docs)
print('\n======= Summarized Content ===========')
print(summarized_ouput.content)

event_prompt = PromptTemplate.from_template(
'What''s the key event mentiontioned in the following text. \n Text: {text}'
)

event_chain = event_prompt | llm
event_response = event_chain.invoke({"text":summarized_ouput}) 
print('\n====== Key Events ==============')
print(event_response.content)


print('\n======== Key Events on Full Text ============')
full_text_events = event_chain.invoke({'text':docs})
print(full_text_events.content)