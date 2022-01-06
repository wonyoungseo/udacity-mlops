from fastapi import FastAPI

from typing import Union
from pydantic import BaseModel

class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int



app = FastAPI()


@app.post("/items/")
async def create_item(item: TaggedItem):
    return item