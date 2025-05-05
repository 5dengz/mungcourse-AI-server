from fastapi import APIRouter
from app.routes.train_model import *

router = APIRouter()

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

def insert_waypoints_by_proximity(base_waypoints, extra_waypoints):
    """
    extra_waypoints를 base_waypoints 중 가장 가까운 지점 뒤에 삽입
    """
    updated = base_waypoints[:]
    for wp in extra_waypoints:
        wp_point = Point(wp)
        min_dist = float('inf')
        insert_idx = len(updated)  # 기본은 마지막

        for i, base_wp in enumerate(updated):
            dist = Point(base_wp).distance(wp_point)
            if dist < min_dist:
                min_dist = dist
                insert_idx = i + 1  # 바로 뒤에 삽입

        updated.insert(insert_idx, wp)
    return updated


def build_route_with_optional_waypoints(G, start_point, direction, radius, num_points, extra_waypoints=None):
    full_waypoints = generate_circle_from_start_and_direction(start_point, direction, radius, num_points)

    if extra_waypoints:
        # dict → (lat, lon) 튜플 변환
        converted_waypoints = [
            (wp["latitude"], wp["longitude"]) if isinstance(wp, dict) else wp
            for wp in extra_waypoints
        ]
        # 경유지를 원형 경유지 사이에 자연스럽게 삽입
        full_waypoints = insert_waypoints_by_proximity(full_waypoints, converted_waypoints)

    # 위경도 → 노드 ID 매핑
    nodes = [ox.nearest_nodes(G, X=lon, Y=lat) for lat, lon in full_waypoints]

    # 노드들을 shortest path로 연결
    route = []
    for i in range(len(nodes) - 1):
        segment = nx.shortest_path(G, nodes[i], nodes[i + 1], weight='length')
        if route:
            route.extend(segment[1:])  # 중복 제거
        else:
            route.extend(segment)

    return route

def get_max_distance_from_start(p1, points):
    return max(geodesic(p1, p2).meters for p2 in points)


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

def handle_unknown_values(df: pd.DataFrame, name_encoder: 'LabelEncoder') -> pd.DataFrame:
    """
    주어진 DataFrame에서 name_encoder에 없는 값은 임시로 처리하는 함수.
    
    Args:
        df (pd.DataFrame): 인코딩된 경로 데이터프레임
        name_encoder (LabelEncoder): 기존의 name encoder
    
    Returns:
        pd.DataFrame: name_encoder에 없는 값이 처리된 DataFrame
    """
    # name_encoder에 없는 값을 임시로 -1로 처리
    unknown_values = df[~df.isin(name_encoder.classes_)]
    df[unknown_values.isna()] = -1  # 없으면 -1로 처리
    
    return df

def predict_and_rank_routes(
    G,
    all_routes: list,
    model: Booster,
    name_encoder: 'LabelEncoder',  # 또는 dict
    build_route_feature_dataframe  # 기존 함수
) -> list:
    """
    LightGBM 모델을 사용하여 각 경로의 예측값을 계산하고 점수에 따라 정렬된 경로를 반환합니다.
    
    Args:
        all_routes (list): 경로들의 리스트, 각 경로는 노드 ID 리스트입니다.
        model (Booster): 학습된 LightGBM 회귀 모델
        name_encoder (LabelEncoder): 경로의 노드 ID 인코딩을 위한 인코더
        build_route_feature_dataframe: 경로 특성을 추출하는 함수
    
    Returns:
        list: 예측값(score)에 따라 정렬된 경로 리스트
    """
    routes_with_scores = []
    
    # 각 경로에 대해 feature 데이터프레임 생성 및 예측 수행
    for route in all_routes:
        # build_route_feature_dataframe 호출하여 경로에 대한 특성 데이터프레임 생성
        df, new_name_encoder = build_route_feature_dataframe(G=G, route=route, label=None, name_encoder=name_encoder)
        
        # name_encoder에 없는 값 처리
        df = handle_unknown_values(df, name_encoder)
        
        if "label" in df.columns:
            df = df.drop(columns=["label"])

        # 모델 예측
        preds = model.predict(df)
        
        # 예측 결과 저장
        for pred in preds:
            routes_with_scores.append({'route': route, 'score': pred})
    
    # 점수 기준으로 경로 정렬 (점수가 낮을수록 좋다고 가정)
    ranked_routes = sorted(routes_with_scores, key=lambda x: x['score'])
    
    return ranked_routes

import json

def convert_routes_to_json(routes_with_length):
    """
    [lat, lon] 형식의 좌표를 포함한 route 데이터를 JSON 문자열로 변환

    Parameters:
        routes_with_length (list): [{'route': [[lat, lon], ...], 'route_length': float}]

    Returns:
        str: JSON 문자열 (list of dicts with lat/lng and route_length)
    """
    result = []

    for route_info in routes_with_length:
        route = route_info['route']
        route_length = route_info['route_length']
        latlng_route = []

        for lat, lon in route:
            latlng_route.append({'lat': lat, 'lng': lon})

        result.append({
            'route': latlng_route,
            'route_length': float(route_length)  # np.float64도 처리 가능
        })

    return json.dumps(result, ensure_ascii=False, indent=2)


def calculate_routes_length(routes_with_scores, G):
    """
    각 route의 거리(route_length)를 계산해 반환하는 함수.
    score는 제외되고, route와 route_length만 포함됨.
    
    Parameters:
        routes_with_scores (list): [{'route': [...], 'score': ...}, ...]
        G (networkx.MultiDiGraph): 거리 정보를 포함한 그래프

    Returns:
        list: [{'route': [...], 'route_length': float}, ...]
    """
    updated_routes = []

    for route_obj in routes_with_scores:
        route = route_obj["route"]
        total_length = 0

        for i in range(len(route) - 1):
            u = route[i]
            v = route[i + 1]
            edge_data = G.get_edge_data(u, v)
            length = edge_data[0]['length'] if isinstance(edge_data, dict) and 0 in edge_data else edge_data['length']
            total_length += length

        updated_routes.append({
            "route": route,
            "route_length": total_length
        })

    return updated_routes

def save_route_lat_lon(G, route_list):
    all_routes_lat_lon = []

    for idx, route_data in enumerate(route_list):
        route = route_data['route']
        route_length = route_data['route_length']
        route_lat_lon = []
        route_geometries = []

        # 인접 노드 쌍으로 geometry 생성
        for u, v in zip(route[:-1], route[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data is None:
                continue  # edge가 없으면 skip
            edge = list(edge_data.values())[0]
            if 'geometry' in edge:
                geom = edge['geometry']
            else:
                point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
                point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
                geom = LineString([point_u, point_v])
            route_geometries.append(geom)

        # geometry를 통해 lat, lon 좌표 추출
        for geom in route_geometries:
            for lon, lat in geom.coords:
                route_lat_lon.append([lat, lon])

        # 라우트 결과 저장
        route_result = {
            'route': route_lat_lon,
            'route_length': route_length
        }
        all_routes_lat_lon.append(route_result)

    return all_routes_lat_lon


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

        # wayponit들의 배열인 waypoints JSON에서 추출
        extra_waypoints = route_data.get("waypoints")

        waypoints = []
        for point in extra_waypoints:
            lat = point.get("latitude")
            lon = point.get("longitude")
            if lat is not None and lon is not None:
                waypoints.append((lat, lon))


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
        model_data = pickle.loads(model_bytes)

        model = model_data['model']

        # name_encoder가 포함된 경우만 처리
        name_encoder = model_data.get('name_encoder', None)
        if name_encoder is not None:
            print("\n name_encoder 내용:")
            if hasattr(name_encoder, 'classes_'):
                print("LabelEncoder 클래스 목록:", name_encoder.classes_)
            else:
                print(name_encoder)
        else:
            print("\nname_encoder 없음. 해당 항목은 제공되지 않았습니다.")

    except Exception as e:
        return {"error": f"Error loading model: {e}"}
    
    radius_list = [100, 200]
    directions = ['N', 'E', 'S', 'W']

    # 그래프 생성
    G = fetch_graph_for_radius(start_location, extra_waypoints=waypoints, radius_list=radius_list)

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

    # 루트들 순위매기기
    ranked_routes = predict_and_rank_routes(G, all_routes, model, name_encoder, build_route_feature_dataframe)

    # 각 루트 길이값 추가하기
    ranked_routes_with_length = calculate_routes_length(ranked_routes, G)
    
    # gps로 바꾸면서 루트 이쁘게 보간/조정
    latlon_routes = save_route_lat_lon(G, ranked_routes_with_length)
    
    # json 형태로 바꾸기
    json_routes = convert_routes_to_json(latlon_routes)

    return json_routes 