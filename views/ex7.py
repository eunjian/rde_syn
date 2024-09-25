import streamlit as st
import pandas as pd
# from utils import 

def createPage():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        prject_name = st.selectbox(
            '프로젝트 선택',
            [0, 1, 2],
            format_func=lambda x: ['프로젝트1', '프로젝트2', '프로젝트3'][x]
        )

    with col4:
        # data = pd.read_csv
        # st.download_button(
        #     label="결과데이터 다운로드",
        #     data=data,
        #     file_name=f"{data}.csv",
        #     mime="text/csv",
        # )
        a = st.button('download')
    if a:
        print(prject_name)

    t1, t2 = st.tabs(['지표 비교 및 데이터 정보', '사각화 비교'])

    with t1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div style="background-color: #E2E2E2; border-radius: 10px; padding: 10px;"><h5 style="color: black">JSD</h5><h2 style="color: black; text-align: center;">3.5</h2></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                '<div style="background-color: #E2E2E2; border-radius: 10px; padding: 10px;"><h5 style="color: black">pMSE</h5><h2 style="color: black; text-align: center;">2.7</h2></div>',
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                '<div style="background-color: #E2E2E2; border-radius: 10px; padding: 10px;"><h5 style="color: black">Corr.Diff</h5><h2 style="color: black; text-align: center;">1.5</h2></div>',
                unsafe_allow_html=True
            )
        st.markdown('<br>', unsafe_allow_html=True)
        summary = pd.DataFrame(
            [
                ['Real', '50', '180', '5.9MB', '10', "40"],
                ['Fake', '50', '200', '5.9MB', '10', "40"],
            ],
            columns=["구분", "전체 컬럼수", "데이터 count", "파일크기", "Category", "Integer"],
        )
        st.data_editor(summary, hide_index=True)
    
    with t2:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            column_name = st.selectbox(
                '컬럼선택',
                ['컬럼1', '컬럼2','컬럼3'],
            )
        with col4:
            chart_name = st.selectbox(
                '그래프선택',
                ['히스토그램', '...'] if True else ['바차트', '파이차트'],
            )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write('차트')
        with col2:
            st.write('통계')

    return True