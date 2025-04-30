from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from fastapi import UploadFile, File

class Location(BaseModel): 
    latitude: float
    longitude: float

class Gps(BaseModel): # 단일 GPS값
    lat: float
    lon: float

class GpsList(BaseModel): # GPS리스트, 한개의 완전한 루트
    gps_list: List[Gps]
    label: Optional[float] = None  # label 필드 추가, 기본값은 None
