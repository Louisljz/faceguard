from fastapi import FastAPI, Request, Response
from utils.model import FaceClassifier
import requests
from uuid import uuid4
import os


model = FaceClassifier()
app = FastAPI(title='Deepfake Video Classifier', description='Implemented by Louis JZ using Meso Net architecture')


def download_video(video_url, file_path):
    response = requests.get(video_url, stream=True)
    with open(file_path, "wb") as video_file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                video_file.write(chunk)


@app.get("/health", status_code=200, description="Health check endpoint")
async def health():
    return {"health": "ok"}


@app.post("/predict", status_code=200, description="Predict the authenticity of a video")
async def predict(request: Request, response: Response):

    try:
        body = await request.json()
        data = body["instances"][0]
        print("Request Body", data)

        video_url = data["video_url"]
        video_id = str(uuid4())
        video_file_path = f"/tmp/{video_id}.mp4"
        download_video(video_url, video_file_path)

        score = model.predict(video_file_path)
        os.remove(video_file_path)
        print("Prediction Score", score)
        return {'predictions': [{'score': round(score, 4)}]}
    
    except Exception as e:
        response.status_code = 500
        return {"error": str(e)}
