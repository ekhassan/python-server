from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import face_recognition
import requests
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_image_data(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail="Failed to fetch image from URL.")

def compare_faces(cnic_image_data, profile_image_data):
    cnic_image = face_recognition.load_image_file(io.BytesIO(cnic_image_data))
    profile_image = face_recognition.load_image_file(io.BytesIO(profile_image_data))

    try:
        cnic_encoding = face_recognition.face_encodings(cnic_image)[0]
    except IndexError:
        return "Face not found on CNIC Card."
    try:
        profile_encoding = face_recognition.face_encodings(profile_image)[0]
    except IndexError:
        return "Face not found on Profile Picture."

    # Compare the faces
    results = face_recognition.compare_faces([cnic_encoding], profile_encoding)
    return results[0]

@app.get("/testing/")
async def test():
    return JSONResponse({'message':'Testing'})

@app.post("/compare-faces/")
async def upload_images(
    cnic_image: UploadFile = File(None),
    profile_image: UploadFile = File(None),
    cnic_image_url: str = Form(None),
    profile_image_url: str = Form(None)
):
    # Check for image data in file or URL form
    if cnic_image is not None:
        cnic_image_data = await cnic_image.read()
    elif cnic_image_url is not None:
        cnic_image_data = fetch_image_data(cnic_image_url)
    else:
        raise HTTPException(status_code=400, detail="CNIC image is required in file or URL form.")

    if profile_image is not None:
        profile_image_data = await profile_image.read()
    elif profile_image_url is not None:
        profile_image_data = fetch_image_data(profile_image_url)
    else:
        raise HTTPException(status_code=400, detail="Profile image is required in file or URL form.")

    # Compare the faces
    match_result = compare_faces(cnic_image_data, profile_image_data)

    if match_result == "Face not found on CNIC Card.":
        raise HTTPException(status_code=400, detail="Face not found on CNIC Card.")
    elif match_result == "Face not found on Profile Picture.":
        raise HTTPException(status_code=400, detail="Face not found on Profile Picture.")

    return {"match": bool(match_result)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
