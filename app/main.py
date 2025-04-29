from fastapi import FastAPI
from app.routes import route_gps, route_location, route_walking

app = FastAPI()

app.include_router(route_gps.router)
app.include_router(route_location.router)
app.include_router(route_walking.router)

# uvicorn app.main:app --reload