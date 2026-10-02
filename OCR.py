from pathlib import Path
import pytesseract
from PIL import Image
import ffmpeg
import logging
import os
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# In Docker tesseract is on PATH; on Windows fall back to the default install location
tesseract_cmd = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe" if os.name == "nt" else None)
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

#upload_video/job_id will be parent_directory
def get_frame(i:int, start:int, end:int, parent_dir:Path):
    '''
    i : to represent Document number
    start : starting time of the transcript
    end : ending time of the transcript
    parent_dir : job_id folder
    '''

    video_path=f'{parent_dir}/video.mp4'
    mid=(start+end)/2

    images_folder=Path(parent_dir)/'images'
    images_folder.mkdir(parents=True, exist_ok=True)
    image_path=images_folder/f'{i}.jpg'

    # ffmpeg (not OpenCV) so every codec YouTube serves works, including AV1
    try:
        (
            ffmpeg
            .input(video_path, ss=mid)
            .output(str(image_path), vframes=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True))
    except ffmpeg.Error as e:
        raise ValueError(f'Frame at {mid}s not captured: {e.stderr.decode(errors="ignore")[-300:]}')

    if not image_path.exists():
        raise ValueError(f'Frame at {mid}s not captured')

    return str(image_path)

def tesseract_OCR(image_path: str):
    try:
        text=pytesseract.image_to_string(Image.open(image_path), lang='eng')
    except Exception as e:
        logging.error(e)
        raise

    return text
