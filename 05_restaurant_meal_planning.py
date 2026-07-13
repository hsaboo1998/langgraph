### Orchestrator worker pattern (dynamic worker assignment) for meal planning and recipe generation
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from typing import Dict, TypedDict, Annotated, List
import operator
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-chat",
    temperature=0,
    max_tokens=4000
)

class Dish(BaseModel):
    name: str = Field(description="Name of the dish")
    ingredients: str = Field(description="Ingredients used in the dish")
    location: str = Field(description="Cultural origin of the dish")

class Dishes(BaseModel):
    dishes: List[Dish] = Field(description="List of dishes with their ingredients and cultural origin")

head_chef_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an a head chef that generates structured action plan for preparing meals.
        Following are the meal requests provided:
        Meal Requests: {meals}.
        Extract the meals from the meal requests and generate a structured output for each meal consisting of following sections:
        1. **name**: Name of the meal
        2. **Ingredients**: Comma separated list all the ingredients required for that meal.
        3. **Cultural Origin**: Mention the cultural origin of the meal.
        """
    )
])
head_chef_chain = head_chef_prompt | llm.with_structured_output(Dishes)

class State(TypedDict):
    meals: str
    dishes: List[Dish]
    # operator.add tells langgraph how to update this state variable when multiple nodes return this variable
    completed_menu: Annotated[List[str], operator.add] # operator.add is equal to python add Eg. operator.add(2, 3)
    final_recipies: str

def head_chef(state: State) -> Dict:
    "defines structured dish list from meal requests"
    response = head_chef_chain.invoke({"meals": state["meals"]})
    return {"dishes": response.dishes}

chef_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a chef from the location: {location}\n"
        "Provide a detailed walkthrough for preparing the dish {name}"
        "Create a list of steps for preparing the dish"
        "In each step clearly specify the ingredients name and quantity used in that step"
        "Provide additional details like cooking time, temperature, and any other relevant information for each step"
        "Use the following ingredients for the dish: {ingredients}"
    )
])

chef_prompt_chain = chef_prompt | llm

class WorkerState(TypedDict):
    dish: Dish
    completed_menu: Annotated[List[str], operator.add]

def chef_worker(state: WorkerState) -> Dict:
    "creates recipe for a dish"
    dish = state["dish"]
    response = chef_prompt_chain.invoke({
        "name": dish.name,
        "ingredients": dish.ingredients,
        "location": dish.location
    })
    return {"completed_menu": [response.content]}

def assign_workers(state: State):
    "Asssign chef_worker for each dish"
    return [Send("chef_worker", {"dish": dish}) for dish in state["dishes"]]

def synthesizer(state: State) -> Dict:
    "synthesizes the final recipes from all the completed dishes"
    return {"final_recipies": "\n\n-------\n\n".join(state["completed_menu"])}

workflow = StateGraph(State)
workflow.add_node("head_chef", head_chef)
workflow.add_node("chef_worker", chef_worker)
workflow.add_node("synthesizer", synthesizer)
workflow.add_edge(START, "head_chef")
workflow.add_conditional_edges("head_chef", assign_workers, ["chef_worker"]) # list of allowed targets
workflow.add_edge("chef_worker", "synthesizer")
workflow.add_edge("synthesizer", END)
workflow = workflow.compile()
state = workflow.invoke({"meals": "I want to have an Indian Curry, an Italian Pasta and Jalebi with Rabri"})
print(state["final_recipies"])