import os
import json
import numpy as np
import torch
from PIL import Image
import mediapipe as mp
import cv2
import streamlit as st
import matplotlib.pyplot as plt

from model import PoseTransformer

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

@st.cache_resource(show_spinner=False)
def load_transformer_model(model_path: str, device: str = 'cpu'):
    try:
        model = PoseTransformer()
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {str(e)}")
        return None

@st.cache_resource
def load_reference_keypoints(json_dir: str):
    ref_kps = {}
    folders = sorted(
        [d for d in os.listdir(json_dir) if os.path.isdir(os.path.join(json_dir, d))],
        key=lambda x: int(x.split('_')[1]) if x.startswith('class_') and x.split('_')[1].isdigit() else x
    )
    for cls_folder in folders:
        cls_path = os.path.join(json_dir, cls_folder)
        label = int(cls_folder.split('_')[1]) if cls_folder.split('_')[1].isdigit() else cls_folder
        pts_list = []
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith('.json'):
                continue
            full = os.path.join(cls_path, fname)
            data = json.load(open(full, 'r', encoding='utf-8'))
            if isinstance(data, list):
                arr = data
            elif isinstance(data, dict):
                if "keypoints" in data and isinstance(data["keypoints"], list):
                    kps = data["keypoints"]
                    arr = [[kps[i], kps[i+1]] for i in range(0, len(kps), 3)]
                elif "landmarks" in data and isinstance(data["landmarks"], list):
                    arr = data["landmarks"]
                elif all(k.isdigit() for k in data.keys()):
                    arr = [data[str(i)] for i in range(len(data))]
                else:
                    arr = None
                    for v in data.values():
                        if isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 2:
                            arr = v
                            break
                    if arr is None:
                        raise ValueError(f"Unsupported JSON structure: {full}")
            else:
                raise ValueError(f"Unsupported JSON type: {type(data)} in {full}")
            pts_list.append(np.array(arr, dtype=np.float32))
        if pts_list:
            ref_kps[label] = np.mean(np.stack(pts_list, axis=0), axis=0)
    return ref_kps

def extract_keypoints(img: Image.Image) -> np.ndarray:
    img_np = np.array(img.convert("RGB"))
    res = pose.process(image=img_np)
    if not res.pose_landmarks:
        raise ValueError("Pose not detected")
    h, w, _ = img_np.shape
    pts = [[lm.x * w, lm.y * h] for lm in res.pose_landmarks.landmark]
    return np.array(pts, dtype=np.float32)

POSE_PARTS = {
    '팔_왼쪽': [11, 13],
    '팔_오른쪽': [12, 14],
    '다리_왼쪽': [23, 25],
    '다리_오른쪽': [24, 26],
}

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

def compute_distances(user_pts: np.ndarray, ref_pts: np.ndarray):
    all_dists = np.linalg.norm(user_pts - ref_pts, axis=1)
    overall = 100 * (1 - np.mean(all_dists) / np.linalg.norm([user_pts.shape[1], user_pts.shape[0]]))
    part_dists = {part: float(np.mean(np.linalg.norm(user_pts[idxs] - ref_pts[idxs], axis=1))) for part, idxs in POSE_PARTS.items()}
    return overall, all_dists, part_dists

def compute_scores(part_dists: dict, max_dist: float = 100.0):
    return {part: max(0, 100 * (1 - dist / max_dist)) for part, dist in part_dists.items()}

def visualize_keypoints(img: Image.Image, user_pts: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
    vis = np.array(img.convert("RGB"))
    for x, y in user_pts:
        cv2.circle(vis, (int(x), int(y)), 4, (0,255,0), -1)
    for x, y in ref_pts:
        cv2.circle(vis, (int(x), int(y)), 4, (255,0,0), -1)
    return vis

def pad_to_1920x1080_with_keypoint_adjustment(image: np.ndarray, keypoints: np.ndarray, target_w: int = 1920, target_h: int = 1080):
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (resized_w, resized_h))
    padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - resized_w) // 2
    y_offset = (target_h - resized_h) // 2
    padded_img[y_offset:y_offset+resized_h, x_offset:x_offset+resized_w] = resized_img
    adjusted_keypoints = keypoints.copy().astype(np.float32)
    if adjusted_keypoints.max() <= 1.5:
        adjusted_keypoints *= np.array([w, h], dtype=np.float32)
    adjusted_keypoints *= scale
    adjusted_keypoints[:, 0] += x_offset
    adjusted_keypoints[:, 1] += y_offset
    return padded_img, adjusted_keypoints, scale, x_offset, y_offset

# [**여기서부터 아래가 수정된 부분**]
def visualize_user_and_reference(user_image: np.ndarray, user_keypoints: np.ndarray, reference_keypoints: np.ndarray, connections):
    # 입력 데이터 전처리 및 형식 변환
    reference_keypoints = np.array(reference_keypoints, dtype=np.float32)
    user_keypoints = np.array(user_keypoints, dtype=np.float32)
    
    # 배열 형태 표준화 (1, N, 2) -> (N, 2)
    if len(reference_keypoints.shape) == 3 and reference_keypoints.shape[0] == 1:
        reference_keypoints = reference_keypoints[0]
    if len(reference_keypoints.shape) == 1 and reference_keypoints.size % 2 == 0:
        reference_keypoints = reference_keypoints.reshape((-1, 2))
    
    # 디버깅 정보 출력
    print(f"User keypoints shape: {user_keypoints.shape}")
    print(f"Reference keypoints shape: {reference_keypoints.shape}")
    
    image = user_image.copy()
    
    # 사용자 키포인트 연결선 그리기 (녹색)
    for i, j in connections:
        if i < len(user_keypoints) and j < len(user_keypoints):
            pt1 = tuple(map(int, user_keypoints[i]))
            pt2 = tuple(map(int, user_keypoints[j]))
            if np.isfinite(pt1).all() and np.isfinite(pt2).all():
                cv2.line(image, pt1, pt2, (0, 255, 0), 3)
    
    # 참조 키포인트 연결선 그리기 (빨간색)
    for i, j in connections:
        if i < len(reference_keypoints) and j < len(reference_keypoints):
            pt1 = tuple(map(int, reference_keypoints[i]))
            pt2 = tuple(map(int, reference_keypoints[j]))
            if np.isfinite(pt1).all() and np.isfinite(pt2).all():
                cv2.line(image, pt1, pt2, (0, 0, 255), 3)  # 빨간색 연결선
    
    # 사용자 키포인트 점 그리기 (녹색)
    for x, y in user_keypoints:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(image, (int(x), int(y)), 6, (0, 255, 0), -1)
    
    # 참조 키포인트 점 그리기 (빨간색)
    for x, y in reference_keypoints:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(image, (int(x), int(y)), 6, (0, 0, 255), -1)  # 빨간색 점
    
    return image

KEY_JOINTS = {
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28
}

def get_joint_weights(num_joints=33):
    weights = np.ones(num_joints)
    for name, idx in KEY_JOINTS.items():
        if idx < num_joints:
            weights[idx] = 3.0
    return weights

def normalize_pose(keypoints):
    kp = np.array(keypoints).copy()
    # 1D 벡터 (예: (66,)) → (33, 2)
    if len(kp.shape) == 1 and kp.size % 2 == 0:
        kp = kp.reshape((-1, 2))  # 유연하게 reshape
    # (1, 33, 2) → (33, 2)
    if len(kp.shape) == 3 and kp.shape[0] == 1:
        kp = kp[0]
        
    # 적은 수의 키포인트 처리 대응 (수정 부분)
    if kp.shape[0] < 13:
        print(f"Warning: 키포인트 수가 적음 ({kp.shape[0]}). 임시 정규화 방법 사용")
        # 적은 키포인트에 대한 대체 정규화 방법
        if kp.shape[0] >= 2:  # 최소 2개 이상의 키포인트가 있는 경우
            center = np.mean(kp, axis=0)
            kp -= center
            scale = np.max(np.linalg.norm(kp, axis=1)) + 1e-8  # 최대 거리로 정규화
            kp /= scale
            return kp
        else:
            # 키포인트가 너무 적어 정규화가 불가능한 경우
            return kp  # 원본 반환
    
    # 기존 정규화 방법 (키포인트가 충분히 많은 경우)
    center = (kp[11] + kp[12]) / 2  # 어깨 중심
    kp -= center
    scale = np.linalg.norm(kp[11] - kp[12]) + 1e-8  # 어깨 폭
    if scale > 0:
        kp /= scale
    return kp

def extract_class_from_filename(filename):
    if '_' in filename:
        parts = filename.split('_')
        for part in parts:
            if part.isdigit():
                return part
    return None

def create_similarity_heatmap(user_kp, ref_kp):
    user_kp = np.array(user_kp)
    ref_kp = np.array(ref_kp)
    
    # 형태 변환
    if len(user_kp.shape) == 1 and user_kp.size % 2 == 0:
        user_kp = user_kp.reshape((-1, 2))
    if len(ref_kp.shape) == 1 and ref_kp.size % 2 == 0:
        ref_kp = ref_kp.reshape((-1, 2))
    if len(user_kp.shape) == 3 and user_kp.shape[0] == 1:
        user_kp = user_kp[0]
    if len(ref_kp.shape) == 3 and ref_kp.shape[0] == 1:
        ref_kp = ref_kp[0]

    # 키포인트 개수가 부족한 경우 간소화된 히트맵 생성
    if user_kp.shape[0] < 13 or ref_kp.shape[0] < 13:
        print(f"Warning: 키포인트 수가 적음 (사용자: {user_kp.shape[0]}, 참조: {ref_kp.shape[0]})")
        # 사용 가능한 공통 키포인트만 선택
        min_joints = min(user_kp.shape[0], ref_kp.shape[0])
        user_kp = user_kp[:min_joints]
        ref_kp = ref_kp[:min_joints]
        
        # 간소화된 히트맵 정보
        similarities = []
        names = [f"점{i+1}" for i in range(min_joints)]
        
        # 각 포인트별 유사도 계산
        for i in range(min_joints):
            dist = np.linalg.norm(user_kp[i] - ref_kp[i])
            sim = max(0, min(1, 1 - dist / 2)) * 100  # 거리 2 이상은 0% 유사도
            similarities.append(sim)
        
        # 히트맵 생성
        fig, ax = plt.subplots(figsize=(10, 3))
        heatmap = ax.imshow([similarities], cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticks([])
        for i, sim in enumerate(similarities):
            ax.text(i, 0, f"{sim:.1f}%", ha="center", va="center", color="black")
        plt.colorbar(heatmap, ax=ax, label="유사도 (%)")
        plt.title("키포인트별 유사도 (간소화)")
        return fig
    
    # 기존 코드 (충분한 키포인트가 있는 경우)
    user_norm = normalize_pose(user_kp)
    ref_norm = normalize_pose(ref_kp)
    joint_similarities = []
    joint_names = ["머리", "목", "어깨", "팔꿈치", "손목", "골반", "무릎", "발목"]
    joint_indices = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     [11, 12],
                     [13, 14],
                     [15, 16],
                     [17, 18, 19, 20, 21, 22],
                     [23, 24],
                     [25, 26],
                     [27, 28, 29, 30, 31, 32]]
    
    for indices in joint_indices:
        dists = []
        for idx in indices:
            dist = np.linalg.norm(user_norm[idx] - ref_norm[idx])
            dists.append(dist)
        sim = 1 - np.mean(dists)
        joint_similarities.append(max(0, min(1, sim)) * 100)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    heatmap = ax.imshow([joint_similarities], cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(joint_names)))
    ax.set_xticklabels(joint_names)
    ax.set_yticks([])
    for i, sim in enumerate(joint_similarities):
        ax.text(i, 0, f"{sim:.1f}%", ha="center", va="center", color="black")
    plt.colorbar(heatmap, ax=ax, label="유사도 (%)")
    plt.title("관절 부위별 유사도")
    return fig
