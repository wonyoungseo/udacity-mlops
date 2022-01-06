from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    message: str


@app.post("/test/{test_id}")
async def exercise_function(test_id: str, query: int, test_message: Message):
    return {
        "path": test_id,
        "query": query,
        "body": test_message
    }