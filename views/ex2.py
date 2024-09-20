import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

def createPage():
    st.header('학습 모델 조회')
    container = st.container(border=True)
    c1, c2, c3 = container.columns(3)
    with c1:
        st.text_input('데이터셋명')
        st.text_input('학습명')
        
    with c2:
        st.selectbox('카테고리', ['전체'])
        st.selectbox('학습상태', ['전체'])
        
    with c3:
        st.selectbox('알고리즘', ['전체'])
        _,b= st.columns([5,1])
        b.button('조회', type='primary')
        
    st.subheader('학습 모델 목록')
    container2 = st.container(border=True)
    # c4, c5, c6, c7, c8, c9, c10, c11, c12 = container2.columns(9)
    # popup_button = st.button('상세 팝업')
    df = pd.DataFrame(
        [{"번호": 4, "카테고리": '기업CB합성', "알고리즘": 'Transformer', "학습명":'기업CB_합성데이터셋생성 v1.2', "데이터셋명":'Corp_CB_Dataset_2024', "등록자": "admin", "수행시작일자": "2019-01-17 02:14:20", "수행종료일시": "2019-01-17 02:21:52", "결과(Accuracy)": 0.8059, "상태": "정상종료", "공개범위": "전체공개", "액션": '상세팝업'},
        {"번호": 2, "카테고리": '기업CB합성', "알고리즘": 'CTAB_GAN+', "학습명":'기업CB_합성데이터셋생성 v1.1', "데이터셋명":'Corp_CB_Dataset_2024', "등록자": "admin", "수행시작일자": "2019-01-17 00:19:35", "수행종료일시": "2019-01-17 00:19:48", "결과(Accuracy)": 0.8025, "상태": "정상종료", "공개범위": "전체공개", "액션": '상세팝업'},
        {"번호": 1, "카테고리": '기업CB합성', "알고리즘": 'CART', "학습명":'기업CB_합성데이터셋생성 v0.1', "데이터셋명":'Corp_CB_Dataset_2024', "등록자": "system", "수행시작일자": "2019-01-28 20:26:31", "수행종료일시": "2019-01-28 20:28:55", "결과(Accuracy)": 0.9820, "상태": "정상종료", "공개범위": "전체공개", "액션": '상세팝업'}
        ])
    edited_df = st.data_editor(df, hide_index=True)
    
    return True