import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from views import ex3
from utils import id03_01_01, id01_03_01

def createPage():
    st.header('학습 모델 조회')
    # container = st.container()
    container = st.container(border=True)
    datas= id03_01_01()
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
        click = b.button('조회', type='primary')
        
    if click:
        ex3.createPage()
        
    st.subheader('학습 모델 목록')
    # container2 = st.container()
    container2 = st.container(border=True)
    # c4, c5, c6, c7, c8, c9, c10, c11, c12 = container2.columns(9)
    # popup_button = st.button('상세 팝업')
    df = pd.DataFrame(
        [{
            '번호': data['번호'],
            '카테고리': data['카테고리'],
            '알고리즘': data['알고리즘'],
            '학습명': data['학습명'],
            '데이터셋명': {i['ID'] : i['데이터셋명'] for i in id01_03_01()}[data['데이터셋ID']],
            '등록자': data['등록자'],
            '수행시작일자': data['수행시작일자'],
            '수행종료일시': data['수행종료일시'],
            '결과(Accuracy)': data['결과(Accuracy)'],
            '상태': data['상태'],
            '공개범위': data['공개범위'],
            '액션': '상세팝업',
        } for data in datas]
    )
    edited_df = st.data_editor(df, hide_index=True)
    
    return True