from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter() 

class Location(BaseModel):
    place_name: str
    latitude: float
    longitude: float

@router.post("/location")
async def get_location(location: Location):
    return {
        "place_name": location.place_name,
        "latitude": location.latitude,
        "longitude": location.longitude
    }
