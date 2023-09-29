from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from cnnClassifier.utils.common_functions import decodeImage
from cnnClassifier.components.inference import Inference
from cnnClassifier.config.configuration import ConfigurationManager  
import shutil
from PIL import Image
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="./static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def train_route():
    os.system("python main.py")
    return {"message": "Training done successfully."}

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile):
    config_manager = ConfigurationManager()
    inference_config = config_manager.get_inference_config()
    classifier = Inference(config=inference_config)
    
    image_path = "ImageInput.jpg"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    prediction = classifier.predict_single_image(image_path)
    print(prediction)
    return {"prediction": prediction}

if __name__ == "__main__":
    import os
    os.system("uvicorn app:app --reload")