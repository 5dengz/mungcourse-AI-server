from fastapi import FastAPI
from app.routes import recommend_route, train_model

app = FastAPI()

app.include_router(train_model.router)
app.include_router(recommend_route.router)

# uvicorn app.main:app --reload