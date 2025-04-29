from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from app.models.schema import GpsList, WalkingRoute, Location
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import pickle
import json
from io import BytesIO
from typing import Optional
from pydantic import BaseModel
import lightgbm as lgb
import math
from geopy.distance import geodesic

router = APIRouter()

def gps_to_route(G, gps_coords):
    """
    GPS 좌표 리스트를 받아 OSMnx 그래프 상의 경로를 생성합니다.
    :param G: OSMnx 그래프
    :param gps_coords: (lat, lon) 형식의 좌표 리스트
    :return: 경로에 포함된 노드 ID 리스트
    """
    # Step 1: 각 GPS 좌표 → 가장 가까운 노드로 매핑
    node_ids = [
        ox.distance.nearest_nodes(G, X=float(coord.lon), Y=float(coord.lat))
        for coord in gps_coords
    ]

    # Step 2: 연속된 노드 쌍 사이 최단 경로 찾기
    full_route = []
    for u, v in zip(node_ids[:-1], node_ids[1:]):
        try:
            path = nx.shortest_path(G, u, v, weight='length')
            if full_route:
                full_route += path[1:]  # 중복 방지
            else:
                full_route += path
        except nx.NetworkXNoPath:
            print(f"경로 없음: {u} → {v}")

    return full_route

def count_name_and_highway(G, route):
    """
    경로 상의 도로명(name)과 도로 유형(highway)을 카운트합니다.
    :param G: OSMnx 그래프
    :param route: 경로 노드 ID 리스트
    :return: (name_counter, highway_counter) - 각각 Counter 객체
    """
    name_counter = Counter()
    highway_counter = Counter()

    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)

        for key, data in edge_data.items():
            name = data.get('name')
            highway = data.get('highway')

            if name:
                if isinstance(name, list):
                    for n in name:
                        if n:
                            name_counter[n.strip()] += 1
                elif isinstance(name, str) and name.strip():
                    name_counter[name.strip()] += 1

            if highway:
                if isinstance(highway, list):
                    for h in highway:
                        if h:
                            highway_counter[h.strip()] += 1
                elif isinstance(highway, str) and highway.strip():
                    highway_counter[highway.strip()] += 1

    return name_counter, highway_counter

def encode_most_common_name(name_counter, name_encoder=None):
    """
    가장 빈도가 높은 도로명을 인코딩합니다.
    기존 인코딩을 유지하면서 새로운 도로명을 추가합니다.
    :param name_counter: 도로명 카운터
    :param name_encoder: 기존 LabelEncoder (없으면 새로 생성)
    :return: (encoded_name, frequency, name_encoder)
    """
    if not name_counter:
        return 0, 0, name_encoder  # unknown name as 0

    most_common_name, freq = name_counter.most_common(1)[0]
    
    if name_encoder is None:
        name_encoder = LabelEncoder()
        name_encoder.fit(list(name_counter.keys()) + [most_common_name])
    else:
        # 기존 인코더에 모든 도로명 추가 (새로운 도로명이 있든 없든)
        existing_classes = set(name_encoder.classes_)
        all_classes = list(existing_classes) + list(name_counter.keys())
        name_encoder.fit(all_classes)
    
    encoded_name = int(name_encoder.transform([most_common_name])[0])
    return encoded_name, freq, name_encoder

def build_feature_row(encoded_name, name_freq, highway_counter, distance, duration, label=None):
    """
    경로의 특징을 DataFrame 행으로 변환합니다.
    :param encoded_name: 인코딩된 도로명
    :param name_freq: 도로명 빈도
    :param highway_counter: 도로 유형 카운터
    :param distance: 총 거리 (m)
    :param duration: 총 소요 시간 (s)
    :param label: 레이블 (선택사항)
    :return: DataFrame
    """
    highway_types = ['busway', 'corridor', 'footway', 'living_street', 'path', 
                    'pedestrian', 'primary', 'primary_link', 'residential', 'secondary', 
                    'secondary_link', 'service', 'services', 'steps', 'tertiary', 
                    'tertiary_link', 'track', 'trunk', 'trunk_link', 'unclassified']

    highway_data = {f'highway_{t}': 0 for t in highway_types}

    for hw_type, count in highway_counter.items():
        key = f'highway_{hw_type}' if hw_type in highway_types else 'highway_unclassified'
        highway_data[key] += count

    row = {
        'distance': distance,
        'duration': duration,
        'name': encoded_name,
        'name_freq': name_freq,
        'label': label
    }
    row.update(highway_data)

    columns = ['distance', 'duration', 'name'] + list(highway_data.keys()) + ['name_freq', 'label']
    return pd.DataFrame([row], columns=columns)

def calculate_distance_and_duration(G, route, default_speed_kph=30):
    """
    경로의 총 거리와 소요 시간을 계산합니다.
    :param G: OSMnx 그래프
    :param route: 경로 노드 ID 리스트
    :param default_speed_kph: 기본 속도 (km/h)
    :return: (distance, duration) - (m, s)
    """
    distance = 0.0
    duration = 0.0

    for u, v in zip(route[:-1], route[1:]):
        edge_datas = G.get_edge_data(u, v)
        if not edge_datas:
            continue

        edge_data = min(edge_datas.values(), key=lambda d: d.get('length', 0))
        length = edge_data.get('length', 0)  # meters
        speed_kph = edge_data.get('speed_kph', default_speed_kph)

        distance += length
        speed_mps = (speed_kph * 1000) / 3600
        if speed_mps > 0:
            duration += length / speed_mps

    return distance, duration

def build_route_feature_dataframe(G, route, label=None, name_encoder=None):
    """
    경로의 모든 특징을 DataFrame으로 변환합니다.
    :param G: OSMnx 그래프
    :param route: 경로 노드 ID 리스트
    :param label: 레이블 (선택사항)
    :param name_encoder: LabelEncoder (선택사항)
    :return: (DataFrame, name_encoder)
    """
    distance, duration = calculate_distance_and_duration(G, route)
    name_counter, highway_counter = count_name_and_highway(G, route)
    encoded_name, name_freq, name_encoder = encode_most_common_name(name_counter, name_encoder)
    df = build_feature_row(encoded_name, name_freq, highway_counter, distance, duration, label)
    return df, name_encoder

def train_model_with_gps(model_data, gps_obj, G):
    """
    gps_obj와 OSMnx 그래프 G를 이용해 DataFrame을 만들고,
    LightGBM 모델에 학습시킨 뒤, pkl bytes로 반환합니다.
    model_data가 dict일 경우 'model' 키에 모델이 있다고 가정합니다.
    """
    # name_encoder 처리
    if isinstance(model_data, dict) and "name_encoder" in model_data:
        name_encoder = model_data["name_encoder"]
    else:
        name_encoder = None

    # 경로 생성 및 특징 추출
    route = gps_to_route(G, gps_obj.get_list)
    if not route:
        raise ValueError("경로를 찾을 수 없습니다.")

    # 특징 DataFrame 생성
    df, name_encoder = build_route_feature_dataframe(G, route, label=gps_obj.label, name_encoder=name_encoder)
    X = df.drop(columns=["label"])
    y = df["label"]

    # 모델 꺼내기
    if isinstance(model_data, dict) and "model" in model_data:
        model = model_data["model"]
    else:
        model = model_data

    # LightGBM Booster 객체 학습
    # 기존 Booster의 파라미터를 가져와서 새로운 Booster 생성
    params = model.params
    train_data = lgb.Dataset(X, label=y)
    
    # 새로운 Booster 생성 및 학습
    new_model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=1,  # 한 번의 부스팅만 수행
        init_model=model,   # 기존 모델을 초기 모델로 사용
    )

    # 항상 dict로 저장
    save_dict = {
        "model": new_model,
        "name_encoder": name_encoder
    }
    new_model_bytes = pickle.dumps(save_dict)
    return new_model_bytes

def generate_circle_from_start_and_direction(start, direction, radius, num_points, clockwise=True):
    direction_angles = {
        "E": 0,
        "N": math.pi / 2,
        "W": math.pi,
        "S": 3 * math.pi / 2,
    }
    if direction not in direction_angles:
        raise ValueError("direction은 'N', 'S', 'E', 'W' 중 하나여야 합니다.")
    base_angle = direction_angles[direction]

    dx = -radius * math.cos(base_angle) / 111000
    dy = -radius * math.sin(base_angle) / 111000
    center_lat = start[0] + dy
    center_lon = start[1] + dx / math.cos(math.radians(start[0]))
    center = (center_lat, center_lon)

    waypoints = []
    for i in range(num_points):
        angle = base_angle + (2 * math.pi * i / num_points) * (-1 if clockwise else 1)
        dx = radius * math.cos(angle) / 111000
        dy = radius * math.sin(angle) / 111000
        lat = center[0] + dy
        lon = center[1] + dx / math.cos(math.radians(center[0]))
        waypoints.append((lat, lon))

    waypoints = rotate_waypoints_to_start_nearest(waypoints, start)
    return waypoints + [waypoints[0]]  # 순환 루프

def rotate_waypoints_to_start_nearest(waypoints, start):
    dists = [((lat - start[0])**2 + (lon - start[1])**2, i) for i, (lat, lon) in enumerate(waypoints)]
    _, start_idx = min(dists)
    return waypoints[start_idx:] + waypoints[:start_idx]

def build_route_with_optional_waypoints(G, start_point, direction, radius, num_points, extra_waypoints=None):
    full_waypoints = generate_circle_from_start_and_direction(start_point, direction, radius, num_points)

    if extra_waypoints:
        full_waypoints += extra_waypoints  # 사용자 지정 경유지 추가

    # 경유지를 가장 가까운 그래프 노드로 매핑
    nodes = [ox.nearest_nodes(G, X=lon, Y=lat) for lat, lon in full_waypoints]

    # 노드들을 순서대로 shortest path 연결
    route = []
    for i in range(len(nodes) - 1):
        segment = nx.shortest_path(G, nodes[i], nodes[i + 1], weight='length')
        if route:
            route.extend(segment[1:])  # 중복 제거
        else:
            route.extend(segment)
    
    return route

def get_max_distance_from_start(p1, p2):
    return geodesic(p1, p2).meters


def fetch_graph_for_radius(start_point, extra_waypoints=None, radius_list=None):
    """
    그래프를 한 번만 생성하고, 경유지가 있으면 그에 맞게 반지름을 조정한 후 그래프를 반환합니다.
    """
    if extra_waypoints:
        # 경유지가 있으면 최대 거리 기준으로 반지름 설정
        d_max = get_max_distance_from_start(start_point, extra_waypoints)
        radius_list = [d_max]  # 경유지가 있으면 단일 반지름
        fetch_radius = d_max + 500  # 그래프 확보 거리
    else:
        # 경유지가 없으면 최대 반지름 기준으로 그래프 확보
        fetch_radius = max(radius_list) + 500  # 최대 반지름 기반으로 그래프 확보
    
    return ox.graph_from_point(start_point, dist=fetch_radius, dist_type="network", network_type="walk")


def generate_all_routes(G, start_point, radius_list, directions, extra_waypoints):
    """
    주어진 반지름과 방향을 사용하여 모든 경로를 생성하여 반환합니다.
    """
    all_routes = []  # 모든 경로 저장
    num_points = 8

    for radius in radius_list:
        for direction in directions:
            route = build_route_with_optional_waypoints(
                G, start_point, direction, radius, num_points, extra_waypoints=extra_waypoints
            )
            all_routes.append(route)  # 저장
            print(f"🌀 반지름: {radius:.1f}m | 방향: {direction} | 경유지 포함 여부: {'O' if extra_waypoints else 'X'}")
    
    return all_routes




@router.post("/train_model")
async def train_model(
    model: UploadFile = File(...),
    json_str: str = Form(...)
):
    """
    GPS 좌표 리스트를 받아 경로를 생성하고, 학습된 모델과 name_encoder가 포함된 pkl 파일을 반환합니다.
    :param model: 업로드된 pkl 파일
    :param json_str: GPS 좌표 리스트가 담긴 JSON 문자열
    :return: 학습된 모델과 name_encoder가 포함된 pkl 파일
    """
    try:
        # 1. pkl 파일 메모리로 읽기
        model_bytes = await model.read()

        # 2. json 파싱 및 GpsList 객체 생성
        gps_data = json.loads(json_str)
        gps_obj = GpsList(**gps_data)

        # 3. pkl 파일에서 모델 로드
        model_data = pickle.loads(model_bytes)
        
        # 4. OSMnx 그래프 생성
        lat, lon = gps_obj.get_list[0].lat, gps_obj.get_list[0].lon
        G = ox.graph_from_point((lat, lon), dist=1000, network_type='walk')

        # 5. 모델 학습 및 새로운 pkl 파일 생성 (name_encoder 포함)
        new_model_bytes = train_model_with_gps(model_data, gps_obj, G)

        # 6. 학습된 모델과 name_encoder가 포함된 pkl 파일 반환
        return StreamingResponse(
            BytesIO(new_model_bytes),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={model.filename}"}
        )

    except json.JSONDecodeError:
        return {"error": "잘못된 JSON 형식입니다."}
    except Exception as e:
        return {"error": f"서버 오류: {str(e)}"}


@router.post("/recommend_route")
async def recommend_route(
    json_str: str = Form(...), 
    model_file: UploadFile = File(...)):

    try:
        route_data = json.loads(json_str)
        
        # start_location을 JSON에서 추출
        start_point = route_data.get("start_location")
        if not start_point:
            return {"error": "start_point is missing in the input JSON"}
        
        latitude = start_point.get("latitude")
        longitude = start_point.get("longitude")

        # print(latitude, longitude) # 시작점 쳌~

        # latitude, longitude 값이 없는 경우 처리
        if latitude is None or longitude is None:
            return {"error": "start_point must contain both 'latitude' and 'longitude'."}
        try:
            # start_point을 tuple로 변환
            start_location = (float(latitude), float(longitude))
        except ValueError:
            return {"error": "Invalid latitude or longitude. Ensure both are valid numbers."}
    

    except Exception as e:
        return {"error": f"Error processing start_location: {e}"}

    try:
        # 모델 파일 로딩
        model_bytes = await model_file.read()
        model = pickle.loads(model_bytes)
        
    except Exception as e:
        return {"error": f"Error loading model: {e}"}
    
    radius_list = [100, 200]
    directions = ['N', 'E', 'S', 'W']
    extra_waypoints = [
        # (37.552882, 126.922893)
    ] 

    # 그래프 생성
    G = fetch_graph_for_radius(start_location, extra_waypoints=extra_waypoints, radius_list=radius_list)

    # 경로 생성
    all_routes = generate_all_routes(G, start_location, radius_list, directions, extra_waypoints)
    
    for idx, route in enumerate(all_routes):
        total_length = 0
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i + 1]
            edge_data = G.get_edge_data(u, v)
            length = edge_data[0]['length'] if isinstance(edge_data, dict) else edge_data['length']
            total_length += length
        print(f"경로 {idx + 1}의 총 거리: {total_length:.2f} meters")


    return all_routes
