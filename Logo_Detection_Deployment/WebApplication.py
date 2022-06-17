from typing import List
import io
from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse,FileResponse
import torch
import shutil
from PIL import Image
from io import BytesIO
import uvicorn

app = FastAPI()

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/centos/AITest/Logo_Detection/best_final.pt', force_reload=True)  # or yolov5n - yolov5x6, custom

@app.post('/')
async def main(file:UploadFile = File(...)):

        
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))
    results = model(pil_image)
    # Results
    try:
        shutil.rmtree('/home/centos/AITest/Logo_Detection/runs/detect/exp/')
    except:
        print("result not there")
    results.save("/home/centos/AITest/Logo_Detection/res.jpg")
    return FileResponse(f"/home/centos/AITest/Logo_Detection/runs/detect/exp/image0.jpg")

if __name__ == "__main__":
#     app.run()
    uvicorn.run(app, host="0.0.0.0", port=3000)
    
#     uvicorn main:app --reload
