from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, WebSocketException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torch
import ffmpeg
import logging
import shutil
import uuid
import whisper
import json
import os


from graph.graph_nodes import Chatbot_initiate
from graph.state import AgentState
from app import download_yt, process_video
from store_in_DB import create_db

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
)

UPLOAD_DIR=Path(os.getenv('UPLOAD_DIR', 'upload_videos'))
UPLOAD_DIR.mkdir(exist_ok=True)

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Your existing mount and routes follow...
app.mount("/videos", StaticFiles(directory=UPLOAD_DIR), name="videos")
app.mount("/static", StaticFiles(directory="static"), name="static")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisper.load_model(os.getenv('WHISPER_MODEL', 'base'), device=device)



# About def upload_url
#1.Uploading Youtube video url -> receives seperate audio and video
#2.Extracting audio from video
#3.Creating transcription from audio

from fastapi.responses import FileResponse

@app.get("/")
async def serve_frontend():
    index_path = Path("frontend2.html")
    if not index_path.exists():
        return JSONResponse(status_code=404, content={"error": "frontend.html not found"})
    return FileResponse('frontend2.html')

@app.post("/upload-url")
# Plain def: FastAPI runs it in a worker thread so the blocking download doesn't freeze the server
def upload_url(bg:BackgroundTasks,url:str):
    job_id=str(uuid.uuid4())
    parent_dir=UPLOAD_DIR/job_id
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        download_yt(parent_dir, url)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail='Error while downloading youtube video'
        )
    video_path=parent_dir/'video.mp4'
    bg.add_task(process_video, UPLOAD_DIR, video_path, job_id)
    
    return JSONResponse(
        status_code=200,
        content={
            'job_id':job_id,
            'status':'In Process'
        }
    )
#status route for the downloading and extracting of the audio to finish
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id == "undefined" or not job_id:
        raise HTTPException(status_code=400, detail="Invalid Job ID")
        
    audio_path = UPLOAD_DIR / job_id / 'audio.wav'
    if audio_path.exists():
        return {"status": "ready"}
    
    # Check if the folder even exists; if not, the download failed
    if not (UPLOAD_DIR / job_id).exists():
        return {"status": "failed"}
        
    return {"status": "processing"}


@app.post("/upload/{job_id}")
def save_DB(job_id: str):
    audio_path = UPLOAD_DIR / job_id / "audio.wav"

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found for job {job_id}"
        )

    try:
        # Whisper transcription
        result = model.transcribe(
            str(audio_path),
            fp16=False
        )

        logging.info("transcription completed")

        # Validate transcription
        if not result or not result.get("text"):
            raise ValueError("Transcription is empty or failed")

        segments = result.get("segments", [])

        # Job directory
        parent_dir = UPLOAD_DIR / job_id
        parent_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Parent Directory: {parent_dir}")

        # Transcript file
        text_path = parent_dir / "transcript.jsonl"

        with open(text_path, "w", encoding="utf-8") as f:
            for segment in segments:
                temp_dict = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                }

                f.write(
                    json.dumps(
                        temp_dict,
                        ensure_ascii=False
                    ) + "\n"
                )

        # Create ChromaDB
        collection_name = "transcription"

        ChromaDB_dir = create_db(
            parent_dir,
            job_id,
            collection_name
        )

        logging.info("ChromaDB created successfully")

        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "Chroma DB": str(ChromaDB_dir)
            }
        )

    except Exception as e:
        logging.exception("Error while processing transcription")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
#perform queries on the video

@app.websocket("/upload/{job_id}/query")
async def query(websocket: WebSocket,job_id:str):
    await websocket.accept()
    collection_name='transcription'
    Parent_dir=UPLOAD_DIR/job_id
    chroma_path=Parent_dir/'chromaDB'

    new_state:AgentState={
        'user_message': None,
        'router_decision' : None,
        'chat_history' : [],
        'knowledge_history' : [],
        'parent_dir' :  Path(Parent_dir),
        'chroma_path' : Path(chroma_path),
        'collection_name' : collection_name,
        'retrieved_docs' : None,
        'ocr_data' : None,
        'final_prompt' : None,
        'LLM_response' : None
    }
    try:
        workflow=Chatbot_initiate(job_id)
        config = {
            'configurable': {
            "thread_id": job_id}}
        
        while True:
            data = await websocket.receive_text()
            new_state['user_message']=data
            try:
                result=workflow.invoke(new_state, config=config)
            except Exception:
                # Keep the socket open so the user can retry instead of a silent disconnect
                logging.exception("Error while answering query")
                await websocket.send_text("Sorry, something went wrong while answering. Please try again.")
                continue
            response=result['LLM_response']
            await websocket.send_text(response)
            new_state=result
    except WebSocketDisconnect:
        print("Client disconnected")
            

#Delteing the job_id folder once task performed
@app.delete("/uploads/{job_id}/delete")
async def delete_job(job_id:str):
    job_dir=UPLOAD_DIR/job_id
    if not job_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f'{job_dir} not found'
        )
    
    try:
        shutil.rmtree(job_dir)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Unexpected error occurred : {e}'
        )
    return JSONResponse(
        status_code=200,
        content={
            'job id': f'{job_id}',
            'status':'job delted'
        }
    )