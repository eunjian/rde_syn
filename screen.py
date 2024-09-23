import streamlit as st
from streamlit_option_menu import option_menu
from views import ex1, ex2, ex4

st.set_page_config(layout="wide")

with st.sidebar:
    st.image('photo_2024-09-20_16-27-37.jpg')
    choice = option_menu("합성데이터 솔루션", ["Home", "데이터수집", "전처리", "후처리", "학습 모델 설계", "학습 모델 조회"],
                         icons=['house','database-add','file-earmark-bar-graph', 'file-earmark-bar-graph-fill', 'bi bi-robot', 'search-heart'],
                         menu_icon="clipboard2-data", default_index=0,
                         styles={
        "container": {"padding": "4!important", "background-color": "(#F0F0F0"},
        "icon": {"color": "black", "font-size": "15px"},
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#FF4500"},
    }
    )

if choice == '학습 모델 설계':
  ex1.createPage()
  
if choice == '학습 모델 조회':
  ex2.createPage()
  
if choice == '데이터수집':
  ex4.createPage()