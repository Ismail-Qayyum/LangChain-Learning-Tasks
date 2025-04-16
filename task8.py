from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from task2 import llm  

ai_text = """
Artificial intelligence (AI) is revolutionizing many industries by automating tasks, enhancing decision-making, and enabling new capabilities. 
In healthcare, AI assists with diagnostics and personalized treatment plans. 
In finance, it detects fraud and optimizes investments. 
In retail, AI powers recommendation systems and manages inventory efficiently. 
AI is also vital in transportation, driving innovations in autonomous vehicles and route optimization. 
Education benefits from AI-driven tutoring systems and personalized learning experiences. 
Overall, AI applications are transforming how businesses operate, boosting productivity, and enabling smarter solutions to complex problems across domains.
"""

response_schemas = [
    ResponseSchema(name="summary", description="A brief summary of the input text"),
    ResponseSchema(name="length", description="The character length of the summary"),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()


prompt = PromptTemplate(
    template="""
Summarize the following text and return a JSON with the summary and its character length.

{format_instructions}

Text:
{text}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions}
)

chain = prompt | llm | parser 
result = chain.invoke({"text":ai_text})

print('\nSummary:')
print(result['summary'])
print(f'\nLength: ' ,result['length'])


