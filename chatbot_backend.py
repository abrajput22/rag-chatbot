from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from embedding import embedder,index
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import cohere

load_dotenv()

# Database configuration
DB_NAME = "chat_memory2.db"


llm = ChatOpenAI(
    model="gemini-2.0-flash", 
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    augmented_prompt:str
    

def retrieval_node(state: ChatState):
    
    messages = state["messages"]
    user_query = messages[-1].content
    print(f"User Query: {user_query}")   
    query_vec = embedder.embed_query(user_query)
    results = index.query(vector=query_vec, top_k=4, include_metadata=True)  

    # for m in results["matches"]:
    #     print(m['score'],"  text=",m["metadata"]["text"])
    
    docs_text=[m["metadata"]["text"] for m in results["matches"]]
    
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    resp = co.rerank(
    model="rerank-v3.5",
    query=user_query,
    documents=docs_text,
    top_n=2
    )

    context_docs=[]
    rerank_results = []
    for r in resp.results:
      context_docs.append(f"{docs_text[r.index]}")
      rerank_results.append(f"Score: {r.relevance_score:.3f} | Text: {docs_text[r.index][:100]}...")
    
    print(f"Reranked Results:\n {' \n '.join(rerank_results)}")

    #context_docs='\n\n'.join(m["metadata"]["text"] for m in results["matches"])

    augmented_prompt = f"""You are a helpful assistant.
    Give answer in formal way as you are an assistant chatbot of a corporate company.
    Use the retrieved context to help answer, but also consider the previous conversation.
    Retrived context:

    {context_docs}

    Question: {user_query}
    """
    # print(augmented_prompt)
    print("-" * 50)
    return {"augmented_prompt": augmented_prompt}



def chat_node(state: ChatState):

    history = state["messages"]
    augmented_prompt = state["augmented_prompt"]
    system_msg = SystemMessage(content=augmented_prompt)
    r = llm.invoke(history + [system_msg])
    return {"messages": [r]}


conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()  

cursor.execute("""
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    created_at REAL
)
""")


class ModelAgnosticSqliteSaver(SqliteSaver):
    def key(self, thread_id: str, model: str | None = None):
        # Ignore model name, only use thread_id
        return thread_id
    
memory = ModelAgnosticSqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("retrieval_node", retrieval_node)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, "retrieval_node")
graph.add_edge("retrieval_node", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=memory)

initial_state = {
    "messages": [HumanMessage(content="Can I skip work for two days without notifying anyone?")],
    "augmented_prompt": ""
}


#res=chatbot.invoke(initial_state,config={'configurable':{'thread_id':'12345678'}})
#print("AI:",res['messages'][-1].content)

def retrive_all_threads():
    cursor.execute("""
        SELECT thread_id 
        FROM threads 
        ORDER BY created_at DESC
    """)
    threads = cursor.fetchall()
    thread_ids = [t[0] for t in threads]
    return thread_ids


#    python chatbot_backend.py