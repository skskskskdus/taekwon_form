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

# --- 유사도 함수에 필요한 추가 import ---
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance

# --- DTW 관련 (optional) ---
try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None

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

def visualize_user_and_reference(user_image: np.ndarray, user_keypoints: np.ndarray, reference_keypoints: np.ndarray, connections):
    reference_keypoints = np.array(reference_keypoints, dtype=np.float32)
    user_keypoints = np.array(user_keypoints, dtype=np.float32)
    if len(reference_keypoints.shape) == 3 and reference_keypoints.shape[0] == 1:
        reference_keypoints = reference_keypoints[0]
    if len(reference_keypoints.shape) == 1 and reference_keypoints.size % 2 == 0:
        reference_keypoints = reference_keypoints.reshape((-1, 2))
    image = user_image.copy()
    for i, j in connections:
        if i < len(user_keypoints) and j < len(user_keypoints):
            pt1 = tuple(map(int, user_keypoints[i]))
            pt2 = tuple(map(int, user_keypoints[j]))
            if np.isfinite(pt1).all() and np.isfinite(pt2).all():
                cv2.line(image, pt1, pt2, (0, 255, 0), 3)
    for i, j in connections:
        if i < len(reference_keypoints) and j < len(reference_keypoints):
            pt1 = tuple(map(int, reference_keypoints[i]))
            pt2 = tuple(map(int, reference_keypoints[j]))
            if np.isfinite(pt1).all() and np.isfinite(pt2).all():
                cv2.line(image, pt1, pt2, (0, 0, 255), 3)
    for x, y in user_keypoints:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(image, (int(x), int(y)), 6, (0, 255, 0), -1)
    for x, y in reference_keypoints:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(image, (int(x), int(y)), 6, (0, 0, 255), -1)
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
    if len(kp.shape) == 1 and kp.size % 2 == 0:
        kp = kp.reshape((-1, 2))
    if len(kp.shape) == 3 and kp.shape[0] == 1:
        kp = kp[0]
    if kp.shape[0] < 13:
        if kp.shape[0] >= 2:
            center = np.mean(kp, axis=0)
            kp -= center
            scale = np.max(np.linalg.norm(kp, axis=1)) + 1e-8
            kp /= scale
            return kp
        else:
            return kp
    center = (kp[11] + kp[12]) / 2
    kp -= center
    scale = np.linalg.norm(kp[11] - kp[12]) + 1e-8
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
    if len(user_kp.shape) == 1 and user_kp.size % 2 == 0:
        user_kp = user_kp.reshape((-1, 2))
    if len(ref_kp.shape) == 1 and ref_kp.size % 2 == 0:
        ref_kp = ref_kp.reshape((-1, 2))
    if len(user_kp.shape) == 3 and user_kp.shape[0] == 1:
        user_kp = user_kp[0]
    if len(ref_kp.shape) == 3 and ref_kp.shape[0] == 1:
        ref_kp = ref_kp[0]
    if user_kp.shape[0] < 13 or ref_kp.shape[0] < 13:
        min_joints = min(user_kp.shape[0], ref_kp.shape[0])
        user_kp = user_kp[:min_joints]
        ref_kp = ref_kp[:min_joints]
        similarities = []
        names = [f"점{i+1}" for i in range(min_joints)]
        for i in range(min_joints):
            dist = np.linalg.norm(user_kp[i] - ref_kp[i])
            sim = max(0, min(1, 1 - dist / 2)) * 100
            similarities.append(sim)
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

# -------------------------------
#      유사도 함수 추가 부분
# -------------------------------

# 1. 거리 기반
def similarity_euclidean(user_pts, ref_pts):
    return 1 - np.mean(np.linalg.norm(user_pts - ref_pts, axis=1)) / np.linalg.norm([user_pts.shape[1], user_pts.shape[0]])

def similarity_mahalanobis(user_pts, ref_pts):
    V = np.cov(np.vstack([user_pts, ref_pts]).T)
    if np.linalg.det(V) == 0:
        return similarity_euclidean(user_pts, ref_pts)
    VI = np.linalg.inv(V)
    dists = [np.sqrt((u - r) @ VI @ (u - r)) for u, r in zip(user_pts, ref_pts)]
    max_maha = max(dists) if max(dists) != 0 else 1
    return 1 - np.mean(dists) / max_maha

# 2. 각도 기반
def similarity_joint_angle(user_pts, ref_pts):
    def angle_3pts(a, b, c):
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.clip(cos_angle, -1, 1)
    joint_sets = [[11, 13, 15], [12, 14, 16], [23, 25, 27], [24, 26, 28]]
    user_angles = []
    ref_angles = []
    for s in joint_sets:
        if max(s) < min(len(user_pts), len(ref_pts)):
            user_angles.append(angle_3pts(user_pts[s[0]], user_pts[s[1]], user_pts[s[2]]))
            ref_angles.append(angle_3pts(ref_pts[s[0]], ref_pts[s[1]], ref_pts[s[2]]))
    return 1 - np.mean(np.abs(np.array(user_angles) - np.array(ref_angles))) / np.pi

# 3. 벡터 기반
def similarity_cosine(user_pts, ref_pts):
    v1 = user_pts.flatten()
    v2 = ref_pts.flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

# 4. 정렬 보정
def similarity_procrustes(user_pts, ref_pts):
    mtx1, mtx2, disparity = procrustes(user_pts, ref_pts)
    return 1 - disparity

# 5. 분포 간
def similarity_hausdorff(user_pts, ref_pts):
    d1 = directed_hausdorff(user_pts, ref_pts)[0]
    d2 = directed_hausdorff(ref_pts, user_pts)[0]
    maxd = max(d1, d2)
    return 1 - maxd / (np.linalg.norm([user_pts.shape[1], user_pts.shape[0]]))

def similarity_chamfer(user_pts, ref_pts):
    dists_1 = cdist(user_pts, ref_pts)
    chamfer = np.mean(np.min(dists_1, axis=1)) + np.mean(np.min(dists_1, axis=0))
    norm = np.linalg.norm([user_pts.shape[1], user_pts.shape[0]])
    return 1 - chamfer / (2 * norm)

def similarity_emd(user_pts, ref_pts):
    emd_x = wasserstein_distance(user_pts[:,0], ref_pts[:,0])
    emd_y = wasserstein_distance(user_pts[:,1], ref_pts[:,1])
    return 1 - (emd_x + emd_y) / (2 * np.linalg.norm([user_pts.shape[1], user_pts.shape[0]]))

# 6. 시계열
def similarity_dtw(user_pts, ref_pts):
    if fastdtw is None:
        return np.nan
    dist_x, _ = fastdtw(user_pts[:,0], ref_pts[:,0])
    dist_y, _ = fastdtw(user_pts[:,1], ref_pts[:,1])
    dtw = (dist_x + dist_y) / 2
    return 1 - dtw / (user_pts.shape[0])

def similarity_softdtw(user_pts, ref_pts):
    try:
        from scipy.spatial.distance import soft_dtw
        gamma = 1.0
        D = cdist(user_pts, ref_pts)
        sdtw = soft_dtw(D, gamma=gamma)
        return 1 - sdtw / (user_pts.shape[0])
    except:
        return similarity_dtw(user_pts, ref_pts)

# 7. 앙상블
def similarity_ensemble(user_pts, ref_pts, weights=None):
    methods = [
        ('유클리드', similarity_euclidean),
        ('마할라노비스', similarity_mahalanobis),
        ('관절각', similarity_joint_angle),
        ('코사인', similarity_cosine),
        ('프로크루스테스', similarity_procrustes),
        ('하우스도르프', similarity_hausdorff),
        ('챔퍼', similarity_chamfer),
        ('EMD', similarity_emd),
        ('DTW', similarity_dtw),
        ('Soft-DTW', similarity_softdtw)
    ]
    results = []
    for name, func in methods:
        try:
            score = float(func(user_pts, ref_pts))
        except Exception as e:
            score = float('nan')
        results.append((name, score))
    arr = np.array([v for _, v in results if not np.isnan(v)])
    if arr.size == 0:
        return 0, results, None
    if weights is None:
        weights = np.ones(arr.shape) / len(arr)
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)
    ensemble_score = np.sum(arr * weights[:len(arr)])
    max_idx = np.nanargmax(arr)
    best_method = [name for name, v in results if not np.isnan(v)][max_idx]
    return ensemble_score, results, best_method

# 유사도 함수 사전
SIMILARITY_METHODS = {
    "유클리드": similarity_euclidean,
    "마할라노비스": similarity_mahalanobis,
    "관절각": similarity_joint_angle,
    "코사인": similarity_cosine,
    "프로크루스테스": similarity_procrustes,
    "하우스도르프": similarity_hausdorff,
    "챔퍼": similarity_chamfer,
    "EMD": similarity_emd,
    "DTW": similarity_dtw,
    "Soft-DTW": similarity_softdtw,
    "앙상블": similarity_ensemble
}
