# 이미지 폴더 경로: C:/Users/LG/taekwondo_forms/taekwon_form/dataset
# JSON 폴더 경로: C:/Users/LG/taekwondo_forms/taekwon_form/joint_point_dataset/new_정리된_관절좌표
# Transformer 모델 경로: C:/Users/LG/taekwondo_forms/taekwon_form/web/pose_transformer.pth
# Streamlit 앱: 사용자 입력 이미지 ↔ 클래스별 MediaPipe 관절 JSON 비교 →
# 파트별 거리·점수 → Overall 유사도 순위 → 상세 피드백 + 시각화

import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import streamlit as st
import pandas as pd
from PIL import Image
import torch
import json
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
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
    create_similarity_heatmap
)
from sklearn.metrics.pairwise import cosine_similarity

# 페이지 설정
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('🥋 태권도 품새 유사도 분석')
# 사이드바 설정
with st.sidebar:
    st.image("https://www.pngall.com/wp-content/uploads/2016/04/Taekwondo-Download-PNG.png", width=100)
    st.markdown("## 설정")
    json_dir = st.text_input('JSON 폴더 경로', 'C:/Users/LG/taekwondo_forms/taekwon_form/joint_point_dataset/new_정리된_관절좌표')
    pth_path = st.text_input('Transformer 모델 경로', 'C:/Users/LG/taekwondo_forms/taekwon_form/web/pose_transformer.pth')
    
    # 시각화 옵션
    st.markdown("## 시각화 옵션")
    show_overlay = st.checkbox('포즈 오버레이 시각화', value=True)
    show_heatmap = st.checkbox('유사도 히트맵 표시', value=False)
    st.markdown("---")
    st.markdown("### 🔍 개발자 정보")
    st.markdown("태권도 품새 자세 분석 시스템")
    st.markdown("© 2025 Taekwondo AI Team")

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
    try:
        ref_kps_dict = load_reference_keypoints(json_dir)
        if os.path.exists(pth_path):
            transformer = load_transformer_model(pth_path)
        else:
            st.warning(f"모델 파일을 찾을 수 없습니다: {pth_path}")
            transformer = None
        return ref_kps_dict, transformer
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        return None, None


# 파일 업로더를 메인 컨테이너의 가운데에 위치
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded = st.file_uploader('📷 사용자 이미지 업로드', type=['png', 'jpg', 'jpeg'], 
                              help="분석할 태권도 동작 이미지를 업로드하세요")

# JSON 폴더 경로가 입력되면 폴더 구조 탐색
if json_dir:
    if os.path.exists(json_dir):
        with st.spinner("데이터 폴더 분석 중..."):
            json_files = []
            for root, dirs, files in os.walk(json_dir):
                for file in files:
                    if file.lower().endswith('.json'):
                        json_files.append(os.path.join(root, file))
            
            if json_files:
                st.sidebar.success(f"{len(json_files)}개의 JSON 파일을 찾았습니다")
                
                # JSON 파일 클래스 분포 계산
                classes = []
                for f in json_files:
                    cls = extract_class_from_filename(os.path.basename(f))
                    if cls:
                        classes.append(cls)
                
                if classes:
                    class_counts = pd.Series(classes).value_counts()
                    
                    with st.sidebar.expander("클래스 분포 확인"):
                        # 간단한 막대 차트
                        st.bar_chart(class_counts)
                        
                        # 데이터프레임으로 표시
                        df = pd.DataFrame({
                            '클래스': class_counts.index,
                            '파일 수': class_counts.values
                        })
                        st.dataframe(df, hide_index=True)
            else:
                st.sidebar.warning("입력한 폴더에서 JSON 파일을 찾을 수 없습니다")
    else:
        st.sidebar.error(f"입력한 JSON 폴더 경로가 존재하지 않습니다: {json_dir}")

def show_result_page(img_pil,
                     results,
                     ranked,
                     best_lbl,
                     best_score,
                     transformer_results,
                     ref_kps_dict,
                     orig_w,
                     orig_h,
                     user_pts,
                     show_overlay,
                     show_heatmap,
                     best_file_path,
                     best_file_score,
                     best_file_class):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import cv2
    import os

    np_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    st.markdown("""
        <style>
        .result-card {
            background: #f8f9fa;
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a237e;
            margin-bottom: 0.5rem;
        }
        .sub-title {
            font-size: 1.2rem;
            color: #3949ab;
            margin-bottom: 1rem;
        }
        .score-badge {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            border-radius: 12px;
            padding: 0.3rem 1.2rem;
            font-size: 1.1rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }
        .best-class {
            color: #388e3c;
            font-weight: 700;
            font-size: 1.3rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🥋 태권도 품새 유사도 분석 결과</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">업로드한 이미지와 데이터셋 내 표준 포즈와의 유사도 분석 결과를 한눈에 확인하세요.</div>', unsafe_allow_html=True)

    # 상단 카드: index, count, best class, best score
    col1, col2, col3, col4 = st.columns([1,1,2,2])
    with col1:
        st.markdown(f'<div class="result-card"><span class="score-badge">클래스 index</span><br><span style="font-size:1.5rem;">{best_lbl}</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="result-card"><span class="score-badge">총 비교 샘플</span><br><span style="font-size:1.5rem;">{len(results)}</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="result-card"><span class="score-badge">최적 일치 클래스</span><br><span class="best-class">{best_lbl}</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="result-card"><span class="score-badge">전체 유사도</span><br><span style="font-size:2rem;color:#1976d2;font-weight:700;">{best_score:.1f}%</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # 중간: 업로드 이미지 & 상위 5개 순위
    col_img, col_rank = st.columns([1.2, 1.8])
    with col_img:
        st.image(img_pil, caption="업로드된 이미지", use_container_width=True)
        if best_file_path:
            st.markdown(f"**가장 유사한 샘플 파일:** `{os.path.basename(best_file_path)}`")
    with col_rank:
        st.markdown("#### 🏆 상위 5개 클래스 유사도 순위")
        top5_df = pd.DataFrame(
            [(f"클래스 {lbl}", f"{score:.1f}%") for lbl, score in ranked[:5]],
            columns=["클래스", "유사도"]
        )
        st.dataframe(top5_df, hide_index=True, use_container_width=True)
        st.bar_chart(pd.DataFrame(
            [(f"클래스 {lbl}", score) for lbl, score in ranked[:5]],
            columns=["클래스", "유사도"]
        ), x="클래스", y="유사도", use_container_width=True)

    st.markdown("---")

    # 하단: 포즈 오버레이 & 히트맵 탭
    tab1, tab2 = st.tabs(["포즈 오버레이", "관절별 유사도 히트맵"])
    with tab1:
        if show_overlay:
            st.markdown("#### 👤 사용자 vs 표준 포즈 오버레이")
            padded_img, adjusted_user_kps, scale, x_off, y_off = pad_to_1920x1080_with_keypoint_adjustment(np_img, user_pts)
            ref_arr = np.array(ref_kps_dict[best_lbl], dtype=np.float32)
            if ref_arr.max() <= 1.5:
                ref_arr[:, 0] *= orig_w
                ref_arr[:, 1] *= orig_h
            ref_arr *= scale
            ref_arr[:, 0] += x_off
            ref_arr[:, 1] += y_off
            overlay = visualize_user_and_reference(padded_img, adjusted_user_kps, ref_arr, POSE_CONNECTIONS)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="포즈 오버레이", use_container_width=True)
    with tab2:
        if show_heatmap:
            st.markdown("#### 🔥 관절별 유사도 히트맵")
            heatmap = create_similarity_heatmap(user_pts, ref_kps_dict[best_lbl])
            st.pyplot(heatmap)

    st.markdown("---")
    st.markdown("#### 💡 품새 분석 예시")
    example_cols = st.columns(3)
    with example_cols[0]:
        st.image(
            "https://www.pngitem.com/pimgs/m/248-2482211_taekwondo-stance-png-transparent-png.png",
            caption="전방 기립 자세",
            width=200
        )
    with example_cols[1]:
        st.image(
            "https://www.pngkey.com/png/detail/366-3667870_taekwondo-stance-taekwondo-fighting-stance.png",
            caption="앞차기 준비 자세",
            width=200
        )
    with example_cols[2]:
        st.image(
            "https://e7.pngegg.com/pngimages/972/911/png-clipart-taekwondo-karate-martial-arts-tang-soo-do-mma-taekwondo-game-sports.png",
            caption="측면 차기 자세",
            width=200
        )

if uploaded and json_dir and pth_path:
    progress_container = st.empty()
    with progress_container.container():
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.write("[디버그] 1단계 진입")
        status_text.text("1/5 단계: 모델 및 참조 키포인트 로딩 중...")
        ref_kps_dict, transformer = get_models(json_dir, pth_path)
        progress_bar.progress(20)
        if ref_kps_dict is None:
            st.error("참조 키포인트를 로드할 수 없습니다. JSON 폴더 경로를 확인해주세요.")
            st.stop()
        st.write("[디버그] 2단계 진입")
        try:
            status_text.text("2/5 단계: 이미지 로드 및 키포인트 추출 중...")
            img_pil = Image.open(uploaded).convert('RGB')
            np_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            orig_h, orig_w = np_img.shape[:2]
            user_pts = extract_keypoints(img_pil)
            progress_bar.progress(40)
            if user_pts is None or len(user_pts) == 0:
                st.error("이미지에서 키포인트를 추출할 수 없습니다. 다른 이미지를 시도해보세요.")
                st.stop()
            st.write("[디버그] 3단계 진입")
            status_text.text("3/5 단계: 유사도 계산 중...")
            user_pts_norm = user_pts.copy()
            user_pts_norm[:, 0] /= orig_w
            user_pts_norm[:, 1] /= orig_h
            transformer_results = None
            if transformer is not None:
                inp = torch.from_numpy(user_pts_norm).unsqueeze(0).float()
                with torch.no_grad():
                    logits = transformer(inp)
                    probs = torch.softmax(logits, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    conf = probs[0, pred].item() * 100
                    transformer_results = (pred, conf)
            results = []
            for lbl, ref in ref_kps_dict.items():
                ref_pixel = ref.copy()
                ref_pixel[:, 0] *= orig_w
                ref_pixel[:, 1] *= orig_h
                sim = calculate_similarity(user_pts, [ref_pixel])
                score = sim * 100
                results.append((lbl, score))
            progress_bar.progress(60)
            if not results:
                st.warning("참조 데이터와의 비교 중 오류가 발생했습니다.")
                st.stop()
            ranked = sorted(results, key=lambda x: x[1], reverse=True)
            best_lbl, best_score = ranked[0]
            st.write("[디버그] 4단계 진입")
            status_text.text("4/5 단계: 가장 유사한 유단자 샘플 검색 중...")
            best_file_score = float('-inf')
            best_file_path = None
            best_file_class = None
            for root, dirs, files in os.walk(json_dir):
                for fname in files:
                    if not fname.lower().endswith('.json'):
                        continue
                    file_path = os.path.join(root, fname)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        arr = None
                        if isinstance(data, list):
                            arr = data
                        elif isinstance(data, dict):
                            if 'landmarks' in data and isinstance(data['landmarks'], list):
                                arr = data['landmarks']
                            elif all(k.isdigit() for k in data.keys()):
                                arr = [data[str(i)] for i in range(len(data))]
                            else:
                                for v in data.values():
                                    if isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 2:
                                        arr = v
                                        break
                        if arr is None:
                            continue
                        pts = np.array(arr, dtype=np.float32)
                        if pts.max() <= 1.5:
                            pts[:, 0] *= orig_w
                            pts[:, 1] *= orig_h
                        dist = np.mean(np.linalg.norm(user_pts - pts, axis=1))
                        score = 100 * (1 - dist / np.linalg.norm([orig_w, orig_h]))
                        filename = os.path.basename(file_path)
                        file_class = extract_class_from_filename(filename)
                        if score > best_file_score:
                            best_file_score = score
                            best_file_path = file_path
                            best_file_class = file_class
                    except Exception as e:
                        continue
            progress_bar.progress(80)
            st.write("[디버그] 5단계 진입")
            status_text.text("5/5 단계: 결과 분석 및 시각화 중...")
            time.sleep(0.5)
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_container.empty()
            st.write("[디버그] 결과 페이지 호출")
            show_result_page(img_pil, results, ranked, best_lbl, best_score, transformer_results, ref_kps_dict, orig_w, orig_h, user_pts, show_overlay, show_heatmap, best_file_path, best_file_score, best_file_class)
        except Exception as e:
            st.error(f"처리 중 오류가 발생했습니다: {e}")
            import traceback
            st.error(traceback.format_exc())
else:
    # 안내 메시지 표시
    st.markdown("📋 사용 방법")
    st.markdown("1. **사이드바**에서 **JSON 폴더 경로**와 **Transformer 모델 경로**를 입력하세요")
    st.markdown("2. 위의 파일 업로더를 통해 **태권도 자세 이미지**를 업로드하세요")
    st.markdown("3. 시스템이 자동으로 이미지를 분석하고 **품새 자세 유사도**를 평가합니다")
    st.markdown("4. 결과에 따라 **포즈 오버레이** 및 **관절별 유사도**를 확인할 수 있습니다")
    
    # 예시 이미지 표시
    st.markdown("💡 품새 분석 예시")
    example_cols = st.columns(3)
    with example_cols[0]:
        st.image("https://www.pngitem.com/pimgs/m/248-2482211_taekwondo-stance-png-transparent-png.png", 
                 caption="전방 기립 자세", width=200)
    with example_cols[1]:
        st.image("https://www.pngkey.com/png/detail/366-3667870_taekwondo-stance-taekwondo-fighting-stance.png", 
                 caption="앞차기 준비 자세", width=200)
    with example_cols[2]:
        st.image("https://e7.pngegg.com/pngimages/972/911/png-clipart-taekwondo-karate-martial-arts-tang-soo-do-mma-taekwondo-game-sports.png", 
                 caption="측면 차기 자세", width=200)