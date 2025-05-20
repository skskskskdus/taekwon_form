# 이미지 폴더 경로: C:/Users/LG/taekwondo_forms/taekwon_form/dataset
# JSOn 폴더 경로: C:/Users/LG/taekwondo_forms/taekwon_form/joint_point_dataset/new_정리된_관절좌표
# test_web2.py
"""
Streamlit 앱: 사용자 입력(TCP 이미지, 웹캠, 비디오)과
클래스별 미리 추출된 MediaPipe 관절 좌표 JSON 비교 →
파트별 거리·점수 → 전체 클래스 유사도 순위 → 상세 피드백
"""
import os
import json
import cv2
import math
import numpy as np
import streamlit as st
from PIL import Image
import mediapipe as mp

# ===============================
# 1) MediaPipe Pose 초기화
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# ===============================
# 2) 포즈 JSON 로드 & 대표 포인트 계산
# ===============================
@st.cache_resource
def load_reference_keypoints(json_dir: str):
    """
    JSON 폴더 구조: json_dir/<class_label>/*.json
    각 클래스별로 모든 JSON 파일을 불러와서 평균 keypoints (33,2) 계산
    반환: {label: mean_pts}
    """
    ref_kps = {}
    for cls in sorted(os.listdir(json_dir)):
        cls_path = os.path.join(json_dir, cls)
        if not os.path.isdir(cls_path): continue
        pts_list = []
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith('.json'): continue
            data = json.load(open(os.path.join(cls_path, fname)))  # {'0':[x,y],...}
            pts = np.array([data[str(i)] for i in range(len(data))], dtype=np.float32)
            pts_list.append(pts)
        if pts_list:
            ref_kps[cls] = np.mean(np.stack(pts_list), axis=0)  # (33,2)
    return ref_kps

# ===============================
# 3) 사용자 keypoints 추출 함수
# ===============================
def extract_keypoints(img: Image.Image) -> np.ndarray:
    img_np = np.array(img.convert("RGB"))
    res = pose.process(img_np)
    if not res.pose_landmarks:
        return None
    pts = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark], dtype=np.float32)
    return pts  # (33,2)

# ===============================
# 4) 파트 정의
# ===============================
POSE_PARTS = {
    'left_arm':  [11,13,15],
    'right_arm': [12,14,16],
    'left_leg':  [23,25,27],
    'right_leg': [24,26,28],
    'torso':     [11,12,23,24]
}

# ===============================
# 5) 거리·점수 계산 함수
# ===============================
def compute_part_metrics(user_pts: np.ndarray, ref_pts: np.ndarray):
    """
    각 파트별 평균 유클리드 거리(px normalized)와 0-100 점수 계산
    거리 단위: normalized coord (0~sqrt(2) 최대)
    score = (1 - (dist / max_dist)) * 100
    max_dist = sqrt(2)
    """
    max_d = math.sqrt(2)
    part_dist, part_score = {}, {}
    for part, idxs in POSE_PARTS.items():
        dists = [np.linalg.norm(user_pts[i] - ref_pts[i]) for i in idxs]
        mean_d = float(np.mean(dists))
        score = max(0.0, (1 - (mean_d / max_d))) * 100
        part_dist[part]  = mean_d
        part_score[part] = score
    return part_dist, part_score

# ===============================
# 6) 클래스별 유사도 순위
# ===============================
def rank_classes(user_pts: np.ndarray, ref_kps: dict):
    """
    모든 클래스에 대해 part metrics 계산 →
    클래스별 전체 score(파트별 score 평균)
    상위 5개 반환
    """
    results = []
    for cls, pts in ref_kps.items():
        dists, scores = compute_part_metrics(user_pts, pts)
        overall = float(np.mean(list(scores.values())))
        results.append((cls, overall, dists, scores))
    # score 내림차순
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ===============================
# 7) Streamlit UI
# ===============================
st.title("🥋 세밀 관절 기반 자세 유사도 분석")
json_dir = st.sidebar.text_input("JSON 폴더 경로", value="taekwon_form_json", key="jdir")
ref_kps = load_reference_keypoints(json_dir)

st.write(f"Loaded {len(ref_kps)} classes from JSON folder: {json_dir}")

# 입력
mode = st.radio("Input Source", ['이미지 업로드','웹캠','비디오'], key="inp_mode")
user_pts = None
if mode == '이미지 업로드':
    file = st.file_uploader("이미지 파일", type=['jpg','png','jpeg'], key="up_img")
    if file:
        img = Image.open(file)
        st.image(img, width=300)
        user_pts = extract_keypoints(img)
elif mode == '웹캠':
    cam = st.camera_input("웹캠 캡처", key="cam")
    if cam:
        img = Image.open(cam)
        st.image(img, width=300)
        user_pts = extract_keypoints(img)
else:
    vid = st.file_uploader("비디오 업로드 (mp4)", type=['mp4'], key="up_vid")
    if vid:
        st.video(vid)
        user_pts = None  # 비디오는 프레임 평균 함수 추가 가능

if user_pts is not None:
    # 클래스 순위 계산
    ranked = rank_classes(user_pts, ref_kps)
    if not ranked:
        st.error("비교할 기준 클래스가 없습니다. JSON 경로와 파일 구성을 확인해주세요.")
    else:
        # 상위 1개
        cls, overall, dists, scores = ranked[0]

        st.header(f"✅ 예측 클래스: {cls}   |   Overall Score: {overall:.1f}%")
        st.subheader("파트별 상세")
        for part in POSE_PARTS:
            st.write(f"- {part}: 거리={dists[part]:.3f}, 점수={scores[part]:.1f}%")

        st.subheader("Top 5 클래스 점수 순위")
        for cls2, sc2, *_ in ranked[:5]:
            st.write(f"{cls2}번: {sc2:.1f}%")

        # (선택) 시각화: 사용자 vs 대표 keypoints
        if st.checkbox('비교 점 찍기', key="vis_kp"):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4,4))
            # 대표 keypoints (빨강), 사용자 (초록)
            ref = ref_kps[cls]
            ax.scatter(ref[:,0], ref[:,1], c='r', label='ref')
            ax.scatter(user_pts[:,0], user_pts[:,1], c='g', label='user')
            ax.invert_yaxis()
            ax.legend()
            st.pyplot(fig)