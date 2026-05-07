# app/main.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.routes.ptr_fitting import router as ptr_router

app = FastAPI(title="PTR Fitting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ptr_router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "PTR Fitting API is running"}