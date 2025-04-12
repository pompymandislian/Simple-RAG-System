from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from typing import Generator
from chroma_chat import chat

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat_stream")
async def chat_stream(query: str = Form(...)):
    def generate() -> Generator[str, None, None]:
        for token in chat(query):
            yield token 

    return StreamingResponse(generate(), media_type="text/event-stream")
