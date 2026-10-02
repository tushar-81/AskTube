from yt_dlp import YoutubeDL
from pathlib import Path
import logging
import ffmpeg
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def download_yt(video_dir, url:str):
    ydl_opts = {
    # Prefer H.264 <=1080p: plays in every browser and avoids huge 4K AV1/VP9 downloads
    "format": "bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]/b[vcodec^=avc1][height<=1080]/bv*[height<=1080]+ba/b",
    "merge_output_format": "mp4",
    "outtmpl": str(video_dir / "video.%(ext)s"),
    "noplaylist": True,
    # YouTube throttles single long streams to ~50 KB/s; 10 MB range requests avoid it
    "http_chunk_size": 10 * 1024 * 1024,
    "quiet": False,
    "ignoreerrors": False
}
    # On Windows ffmpeg is usually not on PATH; in Docker it is installed via apt
    ffmpeg_location = os.getenv("FFMPEG_LOCATION", r"C:/ffmpeg/bin" if os.name == "nt" else None)
    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location

    logging.info('Downloading Youtube video')
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def process_video(UPLOAD_DIR, video_path, job_id:str):
    audio_dir= UPLOAD_DIR/job_id
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_path=audio_dir/'audio.wav'

    if not video_path.exists():
        raise ValueError("Video file doesn't exist")

    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(audio_path),
                format='wav',
                acodec="pcm_s16le",
                ac=1,
                ar=16000).overwrite_output().run(capture_stdout=True, capture_stderr=True))
    
    except ffmpeg.Error as e:
        raise ValueError(f"Error {e} occurred")
    pass