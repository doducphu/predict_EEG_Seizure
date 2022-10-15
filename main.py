from run_model import runmodel
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
app = FastAPI()
@app.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a txt
    if file.filename.endswith(".txt"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        # Return a JSON object containing the model predictions
        return {
            "Labels": runmodel(file.filename)
        }
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request
        raise HTTPException(status_code=400, detail="Invalid file format. Only txt Files accepted.")


    # uvicorn.run(app='main:app',port=8080)