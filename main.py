import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="ChatGPT Proxy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class PromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 10000


class ChatResponse(BaseModel):
    response: str
    model: str
    usage: dict


@app.get("/")
async def root():
    return {
        "message": "ChatGPT Proxy API",
        "endpoints": {
            "/chat": "POST - Send a prompt to ChatGPT",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: PromptRequest):
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

        completion = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "user", "content": request.prompt}
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        response_content = completion.choices[0].message.content
        if response_content is None:
            raise HTTPException(status_code=500, detail="Empty response from OpenAI")

        if completion.usage is None:
            raise HTTPException(status_code=500, detail="No usage data from OpenAI")

        return ChatResponse(
            response=response_content,
            model=completion.model,
            usage={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
