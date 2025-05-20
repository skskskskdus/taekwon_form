import streamlit as st
import pandas as pd
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="태권도 폼 분석",
    page_icon="🥋",
    layout="wide"
)

# 헤더
st.title("태권도 폼 분석 시스템")
st.markdown("---")

# 사이드바
st.sidebar.header("메뉴")
menu = st.sidebar.selectbox(
    "선택하세요",
    ["홈", "폼 분석", "통계", "설정"]
)

# 메인 컨텐츠
if menu == "홈":
    st.header("환영합니다!")
    st.write("이 시스템은 태권도 폼을 분석하고 평가하는 도구입니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("주요 기능")
        st.write("- 실시간 폼 분석")
        st.write("- 자세 평가")
        st.write("- 통계 데이터")
        
    with col2:
        st.subheader("사용 방법")
        st.write("1. 사이드바에서 원하는 메뉴를 선택하세요")
        st.write("2. 분석을 시작하세요")
        st.write("3. 결과를 확인하세요")

elif menu == "폼 분석":
    st.header("폼 분석")
    uploaded_file = st.file_uploader("동영상을 업로드하세요", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        # 분석 버튼
        if st.button("분석 시작"):
            with st.spinner("분석 중..."):
                # 여기에 실제 분석 코드를 추가할 수 있습니다
                st.success("분석 완료!")
                
                # 결과 표시
                st.subheader("분석 결과")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("정확도: 85%")
                    st.write("개선이 필요한 부분:")
                    st.write("- 발차기 높이")
                    st.write("- 균형")
                    
                with col2:
                    # 간단한 차트 예시
                    data = pd.DataFrame({
                        '점수': [85, 75, 90],
                        '항목': ['정확도', '속도', '균형']
                    })
                    st.bar_chart(data.set_index('항목'))

elif menu == "통계":
    st.header("통계")
    # 샘플 데이터 생성
    data = pd.DataFrame({
        '날짜': pd.date_range(start='2023-01-01', periods=30),
        '점수': np.random.randint(60, 100, 30)
    })
    
    st.line_chart(data.set_index('날짜'))
    
    # 통계 정보
    st.subheader("기본 통계")
    st.write(f"평균 점수: {data['점수'].mean():.2f}")
    st.write(f"최고 점수: {data['점수'].max()}")
    st.write(f"최저 점수: {data['점수'].min()}")

else:
    st.header("설정")
    st.write("시스템 설정을 변경할 수 있습니다.")
    
    # 설정 옵션
    st.subheader("일반 설정")
    theme = st.selectbox("테마", ["라이트", "다크"])
    language = st.selectbox("언어", ["한국어", "영어"])
    
    st.subheader("알림 설정")
    email_notification = st.checkbox("이메일 알림 받기")
    if email_notification:
        email = st.text_input("이메일 주소")
    
    if st.button("설정 저장"):
        st.success("설정이 저장되었습니다!") 