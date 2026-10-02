from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Optional
from graph.state import AgentState
from graph.LLM_call import Model, JSON_Model
from store_in_DB import query, perform_OCR
from langgraph.checkpoint.redis import RedisSaver
from redis import Redis

from pathlib import Path
import logging
import json
import os

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
)


#Node (Routing node)
def Message_classification(state : AgentState):
    user_message = state['user_message']
    chat_history = state['chat_history']
    knowledge_history = state['knowledge_history']
    prompt = '''
User's message:
{user_message}

Previous conversation:
{chat_history}

Previous retrieved knowledge:
{knowledge_history}

Act as a routing classifier for a video question-answering assistant.
The user has already uploaded a video and is chatting about it. Questions such as
"what is the main topic", "what is discussed", "summarize", "what does he say about X"
always refer to that video, even when the word "video" is not used.


Classify the user's message into exactly one category:

1. {{'category': 'General'}}
Use this when the question can be answered without using:
- previous conversation
- uploaded video
- transcript
- OCR data
- retrieved documents

2. {{'category': 'History_Only'}}
Use this when the answer depends on information from the previous conversation,
such as:
- user's name
- something the user said earlier
- a previous question or answer
- previous conversation context

3. {{'category': 'Retrieval_Required'}}
Use this when the answer requires information from the uploaded video,
transcript, OCR, or vector database, including:
- what the video is about
- summarizing the video
- key points
- things said in the video
- events discussed in the video
- specific information from the video

IMPORTANT:

- If the user asks about something from previous conversation, choose History_Only.
- If the user asks about the uploaded video, choose Retrieval_Required.
- Do not choose General when previous conversation or video information is required.
- Choose General only for greetings, small talk, or questions clearly unrelated to the video.
- When unsure, choose Retrieval_Required.


Respond only in JSON.
'''
    question = prompt.format(
    user_message=user_message,
    chat_history=chat_history,
    knowledge_history=knowledge_history
)   

    decision = JSON_Model(model_name='gpt-4o-mini', question=question)
    logging.info(f'decision is {decision}')
    
    decision_dict=json.loads(decision)
    return {
        **state,
        'router_decision' : decision_dict['category']
    }

def routing_function(state:AgentState):
    if state['router_decision']=='General':
        return 'General'
    
    elif state['router_decision']=='History_Only':
        return 'History_Only'
    
    elif state['router_decision']=='Retrieval_Required':
        return 'Retrieval_Required'

#Node General Question
def General(state: AgentState):
    prompt=state['user_message']
    return {
        **state,
        'final_prompt' : prompt
    }

#Node Retrieving 
def Retrieval_Required(state: AgentState):
    parent_dir=state['parent_dir']
    Chroma_path=state['chroma_path']

    question=state['user_message']
    collection_name=state['collection_name']
    collection_path=state['chroma_path']    
    parent_dir=state['parent_dir']
    
    try:
        logging.info(f'Querying the question {question}')
        query_result=query(question=question, 
                       collection_name=collection_name,
                       collection_path=collection_path)
        logging.info('Queried')

        docs=query_result['documents']

        query_text=','.join(str(doc) for doc in docs)
        
        logging.info('performing OCR')
        OCR_=perform_OCR(query_result=query_result, parent_dir=parent_dir)
        logging.info(f'OCR performed : {OCR_}')
        
        ocr_text=','.join(str(ocr) for ocr in OCR_)

    except Exception as e:
        logging.error(e)
        raise

    final_prompt=f'''
    User Message : {state['user_message']}

    Answer user's message using the information provided: 
    Chat History : {state['chat_history']}
    Knowledge History : {state['knowledge_history']}

    Retrieved document from Vector database : {query_text}
    Object Charecter Recognition data : {ocr_text}
    Add extra information only if required for better explanation    
   '''

    Knowledge_History=f'''
    Retrieved Document : {query_text}, OCR text : {ocr_text}
    '''

    return{
        **state,
        'knowledge_history': state['knowledge_history'] + [Knowledge_History],
        'retrieved_docs': query_text, 
        'ocr_data': ocr_text, 
        'final_prompt': final_prompt}
    

def History_Only(state:AgentState):
    prompt=f'''
    User Message : {state['user_message']}

    Answer user's message using the information provided: 
    Chat History : {state['chat_history']}
    Knowledge History : {state['knowledge_history']}
    Add extra information only if required for better explanation    
    '''
    return {
        **state,
        'final_prompt': prompt
    }

def LLM_Call(state:AgentState):
    prompt=state['final_prompt']
    try:
        response=Model(model_name='gpt-4o-mini', question=str(prompt))
        print(response)
    except Exception as e:
        logging.error(e)
        raise

    Chat_History={
        'User' : str(state['user_message']),
        'LLM'  : response
    }
    return {
        **state,
         'chat_history': state['chat_history'] + [str(Chat_History)], 
        'LLM_response': response}
    

def Builder_workflow():
    graph=StateGraph(AgentState)

    graph.add_node('message_classification', Message_classification)
    graph.add_node('General', General)
    graph.add_node('Retrieval Required', Retrieval_Required)
    graph.add_node('History Only', History_Only)
    graph.add_node('LLM Call', LLM_Call)
    
    graph.add_edge(START, 'message_classification')

    graph.add_conditional_edges(
    'message_classification',
    routing_function,
    {
        "General": "General",
        "History_Only": "History Only",
        "Retrieval_Required": "Retrieval Required"
    })

    graph.add_edge('General', 'LLM Call')
    graph.add_edge('Retrieval Required', 'LLM Call')
    graph.add_edge('History Only', 'LLM Call')
    graph.add_edge('LLM Call', END)

    return graph

def Chatbot_initiate(job_id:str):
    try:
        redis_client = Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=False)

        checkpointer = RedisSaver(redis_client=redis_client)
        checkpointer.setup()

        graph=Builder_workflow()
        config={'configurable':{
            "thread_id": job_id}}
        workflow=graph.compile(checkpointer=checkpointer)
       
        return workflow
    
    except Exception as e:
        logging.error(e)
        raise