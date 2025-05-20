# 이미지 폴더 경로: C:/Users/LG/taekwondo_forms/taekwon_form/dataset
# JSON 폴더 경로: C:/Users/LG/taekwondo_forms/taekwon_form/joint_point_dataset/new_정리된_관절좌표
# Transformer 모델 경로: C:/Users/LG/taekwondo_forms/taekwon_form/web/pose_transformer.pth
# Streamlit 앱: 사용자 입력 이미지 ↔ 클래스별 MediaPipe 관절 JSON 비교 →
# 파트별 거리·점수 → Overall 유사도 순위 → 상세 피드백 + 시각화

import streamlit as st
from PIL import Image
import torch
import os
import json
import numpy as np
import cv2
from utils import (
    load_transformer_model,
    load_reference_keypoints,
    extract_keypoints,
    pad_to_1920x1080_with_keypoint_adjustment,
    visualize_user_and_reference,
    POSE_CONNECTIONS
)
from sklearn.metrics.pairwise import cosine_similarity

# KEY JOINTS 매핑
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
    for name in ['left_wrist', 'right_wrist', 'left_elbow', 'right_elbow']:
        weights[KEY_JOINTS[name]] = 3.0
    return weights

def normalize_pose(keypoints):
    kp = keypoints.copy()
    center = (kp[11] + kp[12]) / 2
    kp -= center
    scale = np.linalg.norm(kp[11] - kp[12])
    if scale > 0:
        kp /= scale
    return kp

def calculate_similarity(user_kp, ref_kp_list):
    weights = get_joint_weights(user_kp.shape[0])
    user_norm = normalize_pose(user_kp)
    user_vec = user_norm.flatten()
    weight_vec = np.repeat(weights, 2)
    sims = []
    for ref in ref_kp_list:
        ref_norm = normalize_pose(ref)
        ref_vec = ref_norm.flatten()
        sim = cosine_similarity([user_vec * weight_vec], [ref_vec * weight_vec])[0][0]
        sims.append(sim)
    return max(sims)

@st.cache_resource
def get_models(json_dir: str, pth_path: str):
    ref_kps_dict = load_reference_keypoints(json_dir)
    transformer = load_transformer_model(pth_path)
    return ref_kps_dict, transformer

# UI 설정
st.title('🔍 태권도 품새 유사도 분석')
uploaded = st.file_uploader('사용자 이미지 업로드', type=['png', 'jpg', 'jpeg'])
json_dir = st.text_input('JSON 폴더 경로', 'C:/Users/LG/taekwondo_forms/taekwon_form/joint_point_dataset/new_정리된_관절좌표')
pth_path = st.text_input('Transformer 모델 경로', 'C:/Users/LG/taekwondo_forms/taekwon_form/web/pose_transformer.pth')

if uploaded and json_dir and pth_path:
    # 모델 및 참조 키포인트 불러오기
    ref_kps_dict, transformer = get_models(json_dir, pth_path)

    # 원본 이미지 및 크기 정보
    img_pil = Image.open(uploaded).convert('RGB')
    np_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    orig_h, orig_w = np_img.shape[:2]

    # 사용자 키포인트 추출 (픽셀 단위)
    user_pts = extract_keypoints(img_pil)

    # Transformer 입력용 키포인트 정규화(0~1)
    user_pts_norm = user_pts.copy()
    user_pts_norm[:, 0] /= orig_w
    user_pts_norm[:, 1] /= orig_h
    st.write('🔍 키포인트 정규화 분포:', user_pts_norm.min(), user_pts_norm.max())

    # Transformer 추론
    inp = torch.from_numpy(user_pts_norm).unsqueeze(0).float()
    with torch.no_grad():
        logits = transformer(inp)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item() * 100
    st.subheader(f'🤖 Transformer 예측: 클래스 {pred}, 신뢰도 {conf:.1f}%')

    # JSON 기반 유사도 비교 → 모델링(1).py와 동일한 공식 적용
    class_results = []
    for lbl, ref in ref_kps_dict.items():
        ref_pixel = ref.copy()
        ref_pixel[:, 0] *= orig_w
        ref_pixel[:, 1] *= orig_h
        sim = calculate_similarity(user_pts, [ref_pixel])
        score = sim * 100
        class_results.append((lbl, score))
    ranked = sorted(class_results, key=lambda x: x[1], reverse=True)
    best_lbl, best_score = ranked[0]
    st.header(f'✅ JSON 예측: 클래스 {best_lbl}, 전체 유사도 {best_score:.1f}%')

    # Top 5 클래스 순위
    st.subheader('Top 5 클래스 순위')
    for lbl, score in ranked[:5]:
        st.write(f'클래스 {lbl}: {score:.1f}%')

    # JSON 샘플 단위 최우수 예시
    best_file_score = -1
    best_file_name = None
    for class_folder in os.listdir(json_dir):
        class_path = os.path.join(json_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if not file.lower().endswith('.json'):
                continue
            file_path = os.path.join(class_path, file)
            try:
                data = json.load(open(file_path, 'r', encoding='utf-8'))
            except:
                continue
            arr = None
            # modeling(1).py 형식: data['keypoints']
            if isinstance(data, dict) and 'keypoints' in data:
                arr = np.array(data['keypoints']).reshape((33, 3))[:, :2]
            # MediaPipe 형식: landmarks
            elif isinstance(data, dict) and 'landmarks' in data:
                arr = np.array(data['landmarks'], dtype=np.float32)
            # 리스트로 직접 좌표 제공
            elif isinstance(data, list):
                arr = np.array(data, dtype=np.float32)
            # 숫자 키 인덱스
            elif isinstance(data, dict) and all(k.isdigit() for k in data.keys()):
                arr = np.vstack([data[str(i)] for i in range(len(data))])
            if arr is None:
                continue
            kp = arr.copy()
            if kp.max() <= 1.5:
                kp[:, 0] *= orig_w
                kp[:, 1] *= orig_h
            sim = calculate_similarity(user_pts, [kp])
            score = sim * 100
            if score > best_file_score:
                best_file_score = score
                best_file_name = file
    if best_file_name:
        st.subheader(f'✅ 가장 유사한 유단자 샘플: {best_file_name} ({best_file_score:.1f}%)')

    # 포즈 오버레이 시각화
    if st.checkbox('포즈 오버레이 시각화'):
        padded_img, adjusted_user_kps, scale, x_off, y_off = pad_to_1920x1080_with_keypoint_adjustment(np_img, user_pts)
        ref_arr = np.array(ref_kps_dict[best_lbl], dtype=np.float32)
        if ref_arr.max() <= 1.5:
            ref_arr[:, 0] *= orig_w
            ref_arr[:, 1] *= orig_h
        ref_arr *= scale
        ref_arr[:, 0] += x_off
        ref_arr[:, 1] += y_off
        overlay = visualize_user_and_reference(padded_img, adjusted_user_kps, ref_arr, POSE_CONNECTIONS)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f'클래스 {best_lbl}, 유사도 {best_score:.1f}%', use_container_width=True)
