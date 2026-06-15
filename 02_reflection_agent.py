from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import Sequence, List
from langgraph.graph import MessageGraph, END
from dotenv import load_dotenv
load_dotenv()
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-chat",
    temperature=0
)

## Optimized Linkedin Post generator

generation_prompt = ChatPromptTemplate.from_messages(
    [("system", 
      """You are a linkedin content assistant to create crafting, deliberate and insightful posts.
      Generate posts based on users request",
      If the user responds with a feedback, respond with a refined version of your previous prompts based on the feedback.
      """),
      MessagesPlaceholder(variable_name="messages")]
)
generate_chain = generation_prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages(
    [("system",
      """You are a linkedin post evaluator. Your task is to provide critical feedback to make the post more
      professional, engaging and insightful.
      1. Evaluate the quality of post based on linkedin best practises.
      2. Analyze the post's potential for engagement likes, commments, reposts.
      3. Examine the formatting like bullet points and icons and hashtags and mentions
      4. Evaluate the post to avoid repetition within the post keeping it concise, cohorent and clear.
      Provide a detailed critique speficifying the areas of improvement based on your evaluation with actionable suggestions.
      """),
      MessagesPlaceholder(variable_name='messages')
      ]
)
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

# Using MessageGraph for conversation workflows (underlying state is updated automatically using AI/Human messages, 
# no manual state management req.)
graph = MessageGraph()
def generation_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    generated_post = generate_chain.invoke({"messages": state})
    return [AIMessage(content=generated_post.content)]

# messaeges are sequence of base messages which are previous AI responses, user input and system level instructions,
# providing full context to reflection node. Feedback from llm is wrapped in Human Message to treat it as if directly 
# coming from the user guiding the llm to refine its prompt
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    feedback = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=feedback.content)]

# router logic
def should_continue(state):
    if len(state)>6:
        return END
    else:
        return "reflect"

graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)
graph.add_edge("reflect", "generate")
graph.set_entry_point("generate")
graph.add_conditional_edges("generate", should_continue)
wf = graph.compile()
response = wf.invoke(HumanMessage(content="Write a linkedin post after getting a quantitative researcher job at JP Morgan Chase"))
print(response[0].content) # original prompt
print(response[1].content) # first generation
print(response[2].content) # first critique
print(response[-1].content) # final generation
