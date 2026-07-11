from fastapi import  FastAPI

app  = FastAPI()

@app.get("/")
def home():
    return {"status:" "govprep api is running"}

@app.get("/hello/{name}")
def hello(name:str):
    return {"message:" f"Hello. {name}!"}
