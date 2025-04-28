from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class Location(BaseModel): # 이건 예시용 이긴 함
    place_name: str
    latitude: float
    longitude: float

class Gps(BaseModel): # 단일 GPS값
    lat: float
    lon: float

class GpsList(BaseModel): # GPS리스트, 한개의 완전한 루트
    get_list: List[Gps]

class WalkingRoute(BaseModel): # 간단한 라우팅용 함수 : 출발지, 도착지, 경유지, 걸리는시간, 만들어진시간.
    start_location: Location
    end_location: Location
    waypoints: List[Location] = []
    travel_time: int
    created_at: datetime = Field(default_factory=datetime.now)
