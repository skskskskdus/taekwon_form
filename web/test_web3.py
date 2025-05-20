# ▶ Streamlit 파일 워처가 PyTorch의 torch.classes 를 잘못 스캔하며 발생하는 RuntimeError를 예방하기 위해
#   스트림릿 파일‑워처 비활성화 + torch.classes 에 더미 __path__ 주입
import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"            # Streamlit ≥1.41 (new 키)
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"      # Streamlit ≤1.40 (old 키)

import streamlit as st
import pandas as pd
from PIL import Image
import torch
import json
import numpy as np
import cv2
import time
from typing import Dict, List, Tuple
import tempfile
import matplotlib.pyplot as plt

# 포즈 오버레이 시각화 함수 추가
from utils import (
    load_transformer_model,
    load_reference_keypoints,
    extract_keypoints,
    pad_to_1920x1080_with_keypoint_adjustment,
    visualize_user_and_reference,
    POSE_CONNECTIONS,
    KEY_JOINTS,
    get_joint_weights,
    normalize_pose,
    extract_class_from_filename,
    create_similarity_heatmap,
)
from sklearn.metrics.pairwise import cosine_similarity

# ────────────────────────────────────────────────────────────────
# 환경 설정
# ────────────────────────────────────────────────────────────────
if not hasattr(torch.classes, "__path__"):
    torch.classes.__path__ = []  # type: ignore[attr-defined]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────
# Streamlit 페이지 설정
# ────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("🥋 태권도 품새 유사도 분석 (Optimized)")

# ─── 사이드바 입력 ───────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://www.pngall.com/wp-content/uploads/2016/04/Taekwondo-Download-PNG.png",
        width=100,
    )
    st.markdown("## 설정")
    json_dir = st.text_input(
        "JSON 폴더 경로",
        "C:/Users/LG/taekwondo_forms/taekwon_form/joint_point_dataset/new_정리된_관절좌표",
    )
    pth_path = st.text_input(
        "Transformer 모델 경로",
        "C:/Users/LG/taekwondo_forms/taekwon_form/web/pose_transformer.pth",
    )

    # 시각화 옵션
    st.markdown("## 시각화 옵션")
    show_overlay = st.checkbox("포즈 오버레이 시각화", value=True)
    show_heatmap = st.checkbox("유사도 히트맵 표시", value=False)

    st.markdown("---")
    st.markdown("### 🔍 개발자 정보")
    st.markdown("태권도 품새 자세 분석 시스템 – Optimized")
    st.markdown("© 2025 Taekwondo AI Team")

# ────────────────────────────────────────────────────────────────
# 참조 키포인트 및 모델 로딩
# ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def get_assets(json_dir: str, pth_path: str):
    """JSON 참조 키포인트 + Transformer 모델 + 전처리된 ref‑bank 로드"""
    # JSON 폴더에서 모든 샘플의 keypoints 불러오기
    ref_kps_dict = load_reference_keypoints(json_dir)
    # 벡터화된 ref bank 생성 (유사도 평가용)
    ref_bank = { lbl: np.stack([normalize_pose(kp).flatten() * np.repeat(get_joint_weights(kp.shape[0]), 2) for kp in ([ref_kps_dict[lbl]] if ref_kps_dict[lbl].ndim==2 else ref_kps_dict[lbl])], axis=0)
                 for lbl in ref_kps_dict }

    model = None
    if os.path.exists(pth_path):
        model = load_transformer_model(pth_path, device=DEVICE.type)
        model.to(DEVICE)
        model.eval()
    return ref_kps_dict, ref_bank, model

# ────────────────────────────────────────────────────────────────
# 결과 화면 함수
# ────────────────────────────────────────────────────────────────
def show_result_page(
    img_pil: Image.Image,
    ranked: List[Tuple[str, float]],
    best_lbl: str,
    best_score: float,
    transformer_results: Tuple[int, float] | None,
    ref_kps_dict: Dict[str, np.ndarray],
    orig_w: int,
    orig_h: int,
    user_pts: np.ndarray,
    show_overlay: bool,
    show_heatmap: bool,
):
    """결과 시각화 페이지"""
    st.markdown("## 🥋 분석 결과 요약")
    # KPI 카드
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("예측 클래스", best_lbl)
    with kpi2:
        st.metric("전체 유사도", f"{best_score:.1f}%")
    if transformer_results:
        with kpi3:
            st.metric("Transformer 예측", f"{transformer_results[0]} ({transformer_results[1]:.1f}% 확신)")

    st.image(img_pil, caption="업로드 이미지", use_container_width=True)

    # 상위 5개 랭킹
    st.subheader("상위 5개 클래스")
    rank_df = pd.DataFrame(ranked[:5], columns=["클래스", "유사도(%)"])
    st.bar_chart(rank_df.set_index("클래스"))

    # 포즈 오버레이 & 히트맵
    if show_overlay or show_heatmap:
        tabs = st.tabs(["포즈 오버레이", "관절별 히트맵"])
        if show_overlay:
            padded_img, adj_user_kps, scale, x_off, y_off = pad_to_1920x1080_with_keypoint_adjustment(
                cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), user_pts
            )
            # 참조 키포인트 배열 가져오기 (전체 샘플 평균 or 단일 배열)
            ref_data = ref_kps_dict[best_lbl]
            # 배열 차원 확인
            if ref_data.ndim == 1:
                # 1D flat 배열인 경우 재구성
                if ref_data.size % 2 == 0:
                    ref_arr = ref_data.reshape((-1, 2)).astype(np.float32)
                else:
                    st.warning("참조 키포인트 형식이 잘못되었습니다. 오버레이를 표시할 수 없습니다.")
                    return
            else:
                ref_arr = ref_data.astype(np.float32)
            # 원본 픽셀 스케일 적용
            if ref_arr.max() <= 1.5:
                ref_arr[:, 0] *= orig_w
                ref_arr[:, 1] *= orig_h
            # 패딩 스케일 및 오프셋 적용
            ref_arr *= scale
            ref_arr[:, 0] += x_off
            ref_arr[:, 1] += y_off
            # 오버레이 시각화
            overlay = visualize_user_and_reference(
                padded_img, adj_user_kps, ref_arr, POSE_CONNECTIONS
            )
            fig = plot_pose_overlay_with_title(
                overlay, pred_label=best_lbl, similarity_score=best_score
            )
            st.pyplot(fig, use_container_width=True)
        if show_heatmap:
            with tabs[1]:
                # 히트맵용 참조 키포인트
                ref_data = ref_kps_dict[best_lbl]
                if ref_data.ndim == 1:
                    if ref_data.size % 2 == 0:
                        ref_data = ref_data.reshape((-1, 2)).astype(np.float32)
                    else:
                        st.warning("참조 키포인트 형식이 잘못되었습니다. 히트맵을 표시할 수 없습니다.")
                        return
                heat = create_similarity_heatmap(user_pts, ref_data)
                st.pyplot(heat)

# ────────────────────────────────────────────────────────────────
# 메인 처리
# ────────────────────────────────────────────────────────────────

uploaded = None
mode = None
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    mode = st.radio("입력 소스 선택", ['이미지 업로드', '웹캠', '비디오'], key="inp_mode")
    if mode == '이미지 업로드':
        uploaded = st.file_uploader("📷 사용자 이미지 업로드", type=["png", "jpg", "jpeg"] )
    elif mode == '웹캠':
        uploaded = st.camera_input("📷 웹캠 캡처")
    else:
        uploaded = st.file_uploader("🎥 비디오 업로드", type=["mp4"])
        if uploaded:
            st.video(uploaded)

if uploaded and json_dir and pth_path:
    with st.spinner("모델과 참조 데이터를 로드하는 중 …"):
        ref_kps_dict, ref_bank, transformer = get_assets(json_dir, pth_path)

    try:
        # 비디오 처리
        if mode == '비디오':
            frames, keypoints_list = process_video_frames(uploaded)
            if not keypoints_list:
                st.error("비디오에서 키포인트를 추출할 수 없습니다.")
                st.stop()
            img_pil = Image.fromarray.frames[0]
            user_pts = keypoints_list[0]
            np_img = frames[0]
            orig_h, orig_w = np_img.shape[:2]
            st.video(uploaded)
        else:
            img_pil = Image.open(uploaded).convert("RGB")
            np_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            orig_h, orig_w = np_img.shape[:2]
            user_pts = extract_keypoints(img_pil)

        if user_pts is None or len(user_pts) == 0:
            st.error("이미지에서 키포인트를 추출할 수 없습니다. 다른 이미지를 시도해보세요.")
            st.stop()

        # 유사도 계산 및 결과 표시
        weight_vec = np.repeat(get_joint_weights(user_pts.shape[0]), 2)
        user_vec = normalize_pose(user_pts).flatten() * weight_vec
        results = [(lbl, cosine_max_similarity(user_vec, ref_bank[lbl]) * 100) for lbl in ref_bank]
        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        best_lbl, best_score = ranked[0]

        transformer_out = None
        if transformer is not None:
            inp = torch.from_numpy(user_pts.copy()).unsqueeze(0).float().to(DEVICE)
            inp[:, :, 0] /= orig_w
            inp[:, :, 1] /= orig_h
            with torch.no_grad():
                probs = torch.softmax(transformer(inp), dim=1)[0]
            pred = int(torch.argmax(probs).item())
            transformer_out = (pred, float(probs[pred] * 100))

        show_result_page(
            img_pil,
            ranked,
            best_lbl,
            best_score,
            transformer_out,
            ref_kps_dict,
            orig_w,
            orig_h,
            user_pts,
            show_overlay,
            show_heatmap,
        )
    except Exception as exc:
        st.error(f"🚨 처리 중 오류 발생: {exc}")
        st.exception(exc)
else:
    st.markdown("### 📋 사용 방법")
    st.markdown("1️⃣ 왼쪽 사이드바에서 **JSON 폴더**와 **Transformer 모델** 경로를 설정합니다.")
    st.markdown("2️⃣ 위에서 이미지 또는 비디오를 업로드하세요.")
    st.markdown("3️⃣ 잠시만 기다리면 분석 결과가 표시됩니다.")
