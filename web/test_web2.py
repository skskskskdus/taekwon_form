import os
# ▶ Streamlit 파일 워처가 PyTorch의 torch.classes 를 잘못 스캔하며 발생하는 RuntimeError를 예방하기 위해
#   스트림릿 파일‑워처 비활성화 + torch.classes 에 더미 __path__ 주입
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

import matplotlib.pyplot as plt
# 포즈 오버레이 시각화 함수 추가
def plot_pose_overlay_with_title(overlay_img, pred_label, similarity_score):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(
        f"예측 동작: {pred_label}\n유사도: {similarity_score:.2f}점",
        fontsize=17, fontweight="bold", color="#003366", loc="center", pad=25
    )
    plt.tight_layout()
    return fig

# ╭─────────────────────────────────────────────────────────╮
# │  RuntimeError("Tried to instantiate class '__path__._path' …")  │
# ╰─────────────────────────────────────────────────────────╯
# torch.classes 는 동적 C++ 래퍼로, 일반적인 __path__ 속성이 없습니다. Streamlit 의
# local_sources_watcher 가 __path__ 를 강제로 탐색하면서 예외가 발생하므로 더미 리스트를
# 주입하여 무해화합니다.
if not hasattr(torch.classes, "__path__"):
    torch.classes.__path__ = []  # type: ignore[attr-defined]

# ─── 런타임 디바이스 설정 ───────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 프로젝트 유틸 함수 로드 ───────────────────────────────────────
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
# Streamlit 페이지 설정
# ────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title("🥋 태권도 품새 유사도 분석 (Optimized)")

# ─── 사이드바 입력 ────────────────────────────────────────────────
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
# 헬퍼: 벡터 기반 유사도 계산 (완전 벡터화 → ~40‑50× 속도 개선)
# ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=1)  # cache across reruns
def _prepare_ref_bank(ref_dict: Dict[str, List[np.ndarray]]):
    """각 클래스별로 (N, 34) 평탄화+정규화+가중치 벡터 스택을 미리 만들어 둡니다."""
    ref_bank: Dict[str, np.ndarray] = {}
    # 공통 가중치 벡터 (x,y 한 쌍마다 동일 가중치)
    any_kp = next(iter(iter(ref_dict.values())))
    weight_vec = np.repeat(get_joint_weights(any_kp.shape[0]), 2)

    for lbl, kp_list in ref_dict.items():
        vecs = []
        for kp in kp_list:
            vec = normalize_pose(kp).flatten() * weight_vec
            vecs.append(vec)
        ref_bank[lbl] = np.stack(vecs, axis=0)  # (M, 34)
    return ref_bank


def cosine_max_similarity(user_vec: np.ndarray, ref_vecs: np.ndarray) -> float:
    """numpy 벡터 연산으로 ref_vecs 행렬과 user_vec 간 최대 코사인 유사도 반환"""
    dot = ref_vecs @ user_vec
    denom = (np.linalg.norm(ref_vecs, axis=1) * np.linalg.norm(user_vec) + 1e-8)
    sims = dot / denom
    return float(np.max(sims))

@st.cache_data(show_spinner=False, max_entries=1)
def _prepare_ref_bank(ref_dict: Dict[str, np.ndarray]):
    """각 클래스별로 평탄화+정규화+가중치 벡터를 미리 만들어 둡니다."""
    ref_bank: Dict[str, np.ndarray] = {}
    
    # 디버깅 정보 출력
    print(f"Reference data keys: {list(ref_dict.keys())}")
    any_key = next(iter(ref_dict.keys()))
    any_kp = ref_dict[any_key]
    print(f"Sample keypoint shape: {any_kp.shape}")
    
    # 안전하게 가중치 벡터 생성
    num_joints = any_kp.shape[0]
    weight_vec = np.repeat(get_joint_weights(num_joints), 2)

    for lbl, kp in ref_dict.items():
        try:
            # 데이터가 단일 샘플인 경우 리스트로 변환
            if len(kp.shape) == 2:  # (joints, 2) 형태
                kp_list = [kp]
            else:
                # 다중 샘플인 경우 (이 부분은 ref_dict 구조에 따라 달라질 수 있음)
                kp_list = kp
                
            vecs = []
            for k in kp_list:
                vec = normalize_pose(k).flatten() * weight_vec
                vecs.append(vec)
                
            if vecs:
                ref_bank[lbl] = np.stack(vecs, axis=0)
        except Exception as e:
            print(f"Error processing keypoints for label {lbl}: {e}")
            continue
            
    return ref_bank

# ────────────────────────────────────────────────────────────────
# 모델 & 데이터 로딩 (한 번만 실행) ───────────────────────────────
# ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=True)
def get_assets(json_dir: str, pth_path: str):
    """JSON 참조 키포인트 + Transformer 모델 + 전처리된 ref‑bank 로드"""
    ref_kps_dict = load_reference_keypoints(json_dir)
    ref_bank = _prepare_ref_bank(ref_kps_dict)

    model = None
    if os.path.exists(pth_path):
        model = load_transformer_model(pth_path)
        model.to(DEVICE)
        model.eval()
    return ref_kps_dict, ref_bank, model


# ────────────────────────────────────────────────────────────────
# 메인 UI 요소 – 파일 업로더
# ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded = st.file_uploader(
        "📷 사용자 이미지 업로드",
        type=["png", "jpg", "jpeg"],
        help="분석할 태권도 동작 이미지를 업로드하세요",
    )

# ────────────────────────────────────────────────────────────────
# 보조 함수: 결과 페이지 (UI 분리)
# ────────────────────────────────────────────────────────────────

def show_result_page(
    img_pil: Image.Image,
    ranked: List[Tuple[str, float]],
    best_lbl: str,
    best_score: float,
    transformer_results: Tuple[int, float] | None,
    ref_kps_dict: Dict[str, List[np.ndarray]],
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

    # 상위 5개 랭킹 표
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
             # 참조 키포인트 배열 가져오기
            ref_data = ref_kps_dict[best_lbl][0]
            
             # 배열 차원 확인 및 적절히 처리
            print(f"Debug: ref_data shape = {ref_data.shape}")
            
            # 1차원 배열인 경우 2차원으로 reshape
            if len(ref_data.shape) == 1:
                # 길이가 짝수인지 확인
                if len(ref_data) % 2 == 0:
                    # x, y 쌍으로 reshape
                    num_points = len(ref_data) // 2
                    ref_arr = ref_data.reshape(num_points, 2)
                else:
                    # 오류 처리
                    st.warning("참조 키포인트 형식이 잘못되었습니다. 오버레이를 표시할 수 없습니다.")
                    return
            else:
                ref_arr = np.array(ref_data, dtype=np.float32)
                
            # 나머지 코드 계속 진행
            if ref_arr.max() <= 1.5:
                ref_arr[:, 0] *= orig_w
                ref_arr[:, 1] *= orig_h
                
            ref_arr *= scale
            ref_arr[:, 0] += x_off
            ref_arr[:, 1] += y_off
            overlay = visualize_user_and_reference(
                padded_img, adj_user_kps, ref_arr, POSE_CONNECTIONS
            )
            fig = plot_pose_overlay_with_title(
            overlay, pred_label=best_lbl, similarity_score=best_score
        )
        st.pyplot(fig, use_container_width=True)
            #st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if show_heatmap:
            with tabs[1]:
                ref_data = ref_kps_dict[best_lbl][0]

                # 배열 차원 확인 및 적절히 처리
                if len(ref_data.shape) == 1:
                    if len(ref_data) % 2 == 0:
                        # x, y 쌍으로 reshape
                        num_points = len(ref_data) // 2
                        ref_data = ref_data.reshape(num_points, 2)
                    else:
                        st.warning("참조 키포인트 형식이 잘못되었습니다. 히트맵을 표시할 수 없습니다.")
                        return

                heat = create_similarity_heatmap(user_pts, ref_data)
                st.pyplot(heat)


# ────────────────────────────────────────────────────────────────
# 메인 프로세스
# ────────────────────────────────────────────────────────────────

if uploaded and json_dir and pth_path:
    with st.spinner("모델과 참조 데이터를 로드하는 중 …"):
        ref_kps_dict, ref_bank, transformer = get_assets(json_dir, pth_path)

    try:
        img_pil = Image.open(uploaded).convert("RGB")
        np_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = np_img.shape[:2]

        # ─── 사용자 키포인트 추출
        user_pts = extract_keypoints(img_pil)
        if user_pts is None or len(user_pts) == 0:
            st.error("이미지에서 키포인트를 추출할 수 없습니다. 다른 이미지를 시도해보세요.")
            st.stop()

        # ─── 사용자 벡터 계산 (정규화·가중치 적용)
        weight_vec = np.repeat(get_joint_weights(user_pts.shape[0]), 2)
        user_vec = normalize_pose(user_pts).flatten() * weight_vec

        # ─── 클래스별 최대 유사도 계산 (벡터화)
        results: List[Tuple[str, float]] = []
        for lbl, ref_vecs in ref_bank.items():
            sim = cosine_max_similarity(user_vec, ref_vecs)
            results.append((lbl, sim * 100))
        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        best_lbl, best_score = ranked[0]

        # ─── Transformer 예측 (선택 사항)
        transformer_out: Tuple[int, float] | None = None
        if transformer is not None:
            inp = torch.from_numpy(user_pts.copy()).unsqueeze(0).float().to(DEVICE)
            inp[:, :, 0] /= orig_w
            inp[:, :, 1] /= orig_h
            with torch.no_grad():
                probs = torch.softmax(transformer(inp), dim=1)[0]
            pred = int(torch.argmax(probs).item())
            transformer_out = (pred, float(probs[pred] * 100))

        # ─── 결과 페이지 호출
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
    st.markdown("2️⃣ 위 파일 업로더에 분석할 이미지를 드래그‑앤‑드롭합니다.")
    st.markdown("3️⃣ 잠시만 기다리면 분석 결과가 아래에 표시됩니다.")
