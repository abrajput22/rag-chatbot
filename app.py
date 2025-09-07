from chatbot_backend import chatbot,retrive_all_threads
import streamlit as st
from langchain_core.messages import HumanMessage
import uuid
import sqlite3
import time

# Database configuration
DB_NAME = "chat_memory2.db"

def generate_thread_id():    
    id=uuid.uuid4()
    return id


def reset_chat():
    id=generate_thread_id() 
    add_thread(st.session_state['thread_id'])
    st.session_state['thread_id'] =id
    st.session_state['message_history'] = []
    

def add_thread(id):
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO threads (thread_id, created_at) VALUES (?, ?)",
        (str(id), time.time())
    )
    conn.commit()
    conn.close()

    

def load_conversations(thread_id):
    messages_list=chatbot.get_state(config={"configurable": {"thread_id": thread_id }}).values.get('messages',[])
    return messages_list


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] =generate_thread_id() 

add_thread(st.session_state['thread_id'])

st.session_state['chat_threads'] = retrive_all_threads()

st.sidebar.title('ChatBot')

if st.sidebar.button('New Chat'):
    reset_chat()
    

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id']=thread_id
        messages=load_conversations(thread_id)
        new_message_history=[]
        for msg in messages:
            if isinstance(msg,HumanMessage):
                new_message_history.append({'role':'user','content':msg.content})
            else:
                new_message_history.append({'role':'assistant','content':msg.content})
    
        st.session_state['message_history']=new_message_history




for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type_here:')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Streaming response
    
    with st.chat_message('assistant'):
        ai_message=st.write_stream(
            message_chunk.content for message_chunk,meta_data in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": st.session_state['thread_id']}},
            stream_mode='messages'
            )
        )
      
    st.session_state['message_history'].append({'role': 'AI', 'content':ai_message})


    #    streamlit run app.py
