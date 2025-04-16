from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from task2 import llm , chain 
from langchain.memory import ConversationSummaryMemory


ml_text = 'Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed. It involves algorithms that identify patterns in data, make predictions, or take actions based on input. Common types include supervised learning, unsupervised learning, and reinforcement learning. Applications range from recommendation systems and fraud detection to speech recognition and image classification. The performance of machine learning models depends on the quality of data and the choice of algorithms. It’s widely used in various industries to automate processes, make data-driven decisions, and enhance user experiences across digital platforms.'

dl_text = 'Deep learning is a specialized branch of machine learning that uses neural networks with multiple layers (deep neural networks) to model complex patterns in large datasets. Inspired by the human brain, it excels at tasks like image recognition, natural language processing, and speech synthesis. Deep learning requires large amounts of data and powerful hardware (especially GPUs) to train effectively. Popular architectures include convolutional neural networks (CNNs) for vision tasks and transformers for language models. Deep learning drives innovations behind technologies like self-driving cars, virtual assistants, and generative AI tools. Its power lies in automatically extracting high-level features from raw data.'


memory = ConversationBufferWindowMemory(k=3)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

response = conversation.predict(input= chain.invoke(ml_text).content + 'this is a summarized version of the topic')
conversational_response = conversation.predict(input=dl_text + '\n summarize the above text taking in consideration the summarized text provided earlier')

new_memory = ConversationSummaryMemory(llm=llm)
new_conversation = ConversationChain(
    llm=llm,
    memory=new_memory,
    verbose=True
)
response = new_conversation.predict(input= chain.invoke(ml_text).content + 'this is a summarized version of the topic')
summary_response = new_conversation.predict(input=dl_text + '\n summarize the above text taking in consideration the summarized text provided earlier')


print('\n=========================Conversational Buffer Memory Response===========================')
print(conversational_response)
print('\n=========================Conversational Summary Memory Response===========================')
print(summary_response)
