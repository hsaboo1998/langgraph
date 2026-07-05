### Reflexion Agent for Diet Recommendation
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import MessageGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
import json
from dotenv import load_dotenv
load_dotenv()
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

tavily_tool = TavilySearchResults(max_results=1) # scientific paper papers search tool
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-chat",
    temperature=0
)
prompt_template = ChatPromptTemplate.from_messages([
    (
        'system',
        """You are an dietician who specializes in recommending vegetarian diet for nutrition.
        Your response must follow these steps:
        1. {first_instruction}
        2. Present rationale behind nutritional advice
        3. Reflect and critique your answer emphaszing what information is missing and what information is extra and unneccesary.
        4. After reflection **list 1-3 search queries separately** for researching on few references for supporting your answer. 
        """),
        MessagesPlaceholder(variable_name='messages'),
        (
            "system",
            "Answer the above question using the required format"
        )
])
class Reflection(BaseModel):
    missing: str = Field(description="What information is missing?")
    extra: str = Field(description="What information is extra/unneccesary")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="Main response to the question")
    reflection: Reflection = Field(description="Self critique of answer")
    search_queries: List[str] = Field(description="Search queries to find evidence to support your answer using a web search tool")

first_responder_prompt = prompt_template.partial(first_instruction="Provide a detailed 250 word answer")
# Automatically generates an AIMessage as the llm response, so can be used directly in the graph
first_responder_chain = first_responder_prompt | llm.bind_tools(tools=[AnswerQuestion])

revise_instructions = """
Revise the previous answer. Use the previous critique to remove any speculated information and improve the answer using new information.
Do not speculate or make assumptions. Give direct and precise answer instead of adding extra explanations.
Add a **reference** section at the bottom of your answer with 1-3 references.
Also add a **search_queries** section with new queries to find evidence using a web search tool to support or improve your answer further.
"""
revisor_prompt = prompt_template.partial(first_instruction=revise_instructions)
class ReviseAnswer(AnswerQuestion):
    "Revise the answer using previous critique and new information. Add a reference section at the bottom of your answer with 1-3 references."
    reference: List[str] = Field(description="References to support your answer using a web search tool")
# Automatically generates an AIMessage as the llm response, so can be used directly in the graph
revisor_chain = revisor_prompt | llm.bind_tools(tools=[ReviseAnswer])
tavily_tool = TavilySearchResults(max_results=3)

def execute_tavily(state: List[BaseMessage]):
    ai_message = state[-1]
    tool_messages = []
    for tool_call in ai_message.tool_calls:
        if tool_call['name'] in ['AnswerQuestion', 'ReviseAnswer']:
            tool_id = tool_call['id']
            search_queries = tool_call['args'].get('search_queries', [])
            search_queries = [search_queries] if isinstance(search_queries, str) else search_queries
            results = {}
            for query in search_queries:
                results[query] = tavily_tool.invoke(query)
            tool_messages.append(ToolMessage(content=json.dumps(results), tool_call_id=tool_id))
    return tool_messages

def event_loop(state: List[BaseMessage]):
    num_iterations = sum([isinstance(msg, ToolMessage) for msg in state])
    if num_iterations > 2:
        return END
    return "execute_tavily"

graph = MessageGraph()
graph.add_node("first_responder", first_responder_chain)
graph.add_node("revisor", revisor_chain)
graph.add_node("execute_tavily", execute_tavily)

graph.add_edge("first_responder", "execute_tavily")
graph.add_edge("execute_tavily", "revisor")
graph.add_conditional_edges("revisor", event_loop)

graph.set_entry_point("first_responder")

app = graph.compile()
query = """
I am 53 kgs and a vegetarian. Recommend a plant-based diet for weight gain to 60 kgs in 3 months without any strength training and only light walking.
Give me a detailed diet plan with meal timings and nutritional information and caloric count.
"""
response = app.invoke([HumanMessage(content=query)])
print(response[-1].content) # final generation