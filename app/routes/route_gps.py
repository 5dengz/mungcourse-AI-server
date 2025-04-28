from fastapi import APIRouter
from app.models.schema import GpsList
import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

router = APIRouter()

def gps_to_route(G, gps_coords): # gps list를 받아 완성된 루트를 리턴함, /get_route에다가 쓸거임
    """
    :param G: osmnx로 생성한 그래프
    :param gps_coords: (lat, lon) 형식의 리스트
    :return: 전체 경로에 포함된 노드 리스트
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


# # 지도 G와 route를 받아 해당 route상의 필요한 특징들을 카운트 하는 함수
# def count_name_and_highway(G, route):
#     name_counter = Counter()
#     highway_counter = Counter()

#     for u, v in zip(route[:-1], route[1:]):
#         edge_data = G.get_edge_data(u, v)

#         for key, data in edge_data.items():
#             name = data.get('name')
#             highway = data.get('highway')

#             if name:
#                 if isinstance(name, list):
#                     for n in name:
#                         if n:
#                             name_counter[n.strip()] += 1
#                 elif isinstance(name, str) and name.strip():
#                     name_counter[name.strip()] += 1

#             if highway:
#                 if isinstance(highway, list):
#                     for h in highway:
#                         if h:
#                             highway_counter[h.strip()] += 1
#                 elif isinstance(highway, str) and highway.strip():
#                     highway_counter[highway.strip()] += 1

#     return name_counter, highway_counter


# # name에 쓰일 label encoding 적용 함수인데, 이건 각 사용자 LightGBM .pkl에 함께 저장되어야 한다. 
# def encode_most_common_name(name_counter):
#     if not name_counter:
#         return 0, 0  # unknown name as 0
#     most_common_name, freq = name_counter.most_common(1)[0]
#     encoder = LabelEncoder()
#     encoder.fit(list(name_counter.keys()) + [most_common_name])
#     encoded_name = int(encoder.transform([most_common_name])[0])

#     # with open('model_and_encoder.pkl', 'wb') as f:                        # model과 함께 pkl저장 예시
#         # pickle.dump({'model': model, 'name_encoder': name_encoder}, f)

#     return encoded_name, freq

# # 앞서 만든 값들을 추가해서 df 생성. 이는 학습과 예측에 모두 쓰임
# def build_feature_row(encoded_name, name_freq, highway_counter, distance, duration, label=None):
#     highway_types = ['busway', 'corridor', 'footway', 'living_street', 'path', 
#                     'pedestrian', 'primary', 'primary_link', 'residential', 'secondary', 
#                     'secondary_link', 'service', 'services', 'steps', 'tertiary', 
#                     'tertiary_link', 'track', 'trunk', 'trunk_link', 'unclassified']

#     highway_data = {f'highway_{t}': 0 for t in highway_types}

#     for hw_type, count in highway_counter.items():
#         key = f'highway_{hw_type}' if hw_type in highway_types else 'highway_unclassified'
#         highway_data[key] += count

#     row = {
#         'distance': distance,
#         'duration': duration,
#         'name': encoded_name,
#         'name_freq': name_freq,
#         'label': label
#     }
#     row.update(highway_data)

#     columns = ['distance', 'duration', 'name'] + list(highway_data.keys()) + ['name_freq', 'label']
#     return pd.DataFrame([row], columns=columns)

# def calculate_distance_and_duration(G, route, default_speed_kph=30):
#     distance = 0.0
#     duration = 0.0

#     for u, v in zip(route[:-1], route[1:]):
#         edge_datas = G.get_edge_data(u, v)
#         if not edge_datas:
#             continue  # 간선 정보가 없으면 스킵

#         # 여러 개의 간선이 있을 경우, 가장 짧은 걸 선택
#         edge_data = min(edge_datas.values(), key=lambda d: d.get('length', 0))
#         length = edge_data.get('length', 0)  # meters
#         speed_kph = edge_data.get('speed_kph', default_speed_kph)

#         # 누적 거리
#         distance += length

#         # 속도 → m/s 변환
#         speed_mps = (speed_kph * 1000) / 3600
#         if speed_mps > 0:
#             duration += length / speed_mps

#     return distance, duration

# def build_route_feature_dataframe(G, route, label=None):
#     distance, duration = calculate_distance_and_duration(G, route)
#     name_counter, highway_counter = count_name_and_highway(G, route)
#     encoded_name, name_freq = encode_most_common_name(name_counter)
#     df = build_feature_row(encoded_name, name_freq, highway_counter, distance, duration)

#     if label is not None:
#         df['label'] = label
#     else:
#         df['label'] = pd.NA

#     return df

@router.post("/get_route") # 좌표 날라오면 받아서 루트 생성해서 response함
async def get_route(request_body: GpsList):
    # 먼저 첫번째 좌표값을 받아와서 osmnx의 G를 구해야함 
    lat, lon = 37.5715, 126.9805
    G = ox.graph_from_point((lat, lon), dist=1000, network_type='walk')
    route = gps_to_route(G, request_body.get_list)
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    
    # df = build_route_feature_dataframe(G, route, None)
    
    # for node_id in route:                        # node데이터들 추출용
    #     node_data = G.nodes[node_id]
    #     print(node_data)
    
    # for u, v in zip(route[:-1], route[1:]):      # edge데이터들 추출용
    #     edge_data = G.get_edge_data(u, v)
    #     print(f"{edge_data}")

    return df
