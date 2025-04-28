from fastapi import APIRouter
from app.models.schema import WalkingRoute

router = APIRouter()

@router.post("/oneway-route")
async def create_walking_route(route: WalkingRoute):
    return route
