from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

### User Authentication application using LangGraph

class AuthState(TypedDict):
    username: Optional[str]
    password: Optional[str]
    is_authenticated: Optional[str]
    output: Optional[str]

def input_node(state):
    username = state.get("username")
    username = input("Enter Username: ")
    password = input("Enter Password: ")
    print(f"Username: {username}, Password: {password}")
    return {"username": username, "password": password}

def validate_node(state):
    username = state.get("username")
    password = state.get("password")
    if username == "admin" and password == "password":
        is_authenticated = True
    else:
        is_authenticated = False
    return {"is_authenticated": is_authenticated}

def success_node(state):
    return {"output": "Authentication Successful!"}

def failure_node(state):
    return {"output": "Authentication Failed!"}

def router(state):
    is_authenticated = state.get("is_authenticated")
    if is_authenticated:
        return "success_node"
    else:
        return "failure_node"

workflow = StateGraph(AuthState)
workflow.add_node("input_node", input_node)
workflow.add_node("validate_node", validate_node)
workflow.add_node("success_node", success_node)
workflow.add_node("failure_node", failure_node)

workflow.add_edge("input_node", "validate_node")
workflow.add_edge("success_node", END)
workflow.add_edge("failure_node", "input_node")
workflow.add_conditional_edges("validate_node", router, {"success_node": "success_node", "failure_node": "failure_node"})

workflow.set_entry_point("input_node")

app = workflow.compile() # ready to use app

result = app.invoke({})
print(app.get_graph().draw_mermaid())