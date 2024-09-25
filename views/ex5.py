import streamlit as st
from utils import id02_01_01, id02_03_01

def createPage():
    # t1, t2 = st.tabs(['pre-built 전처리', 'user-defined 전처리'])
    # 정렬 및 기준일자 표 생성
    # 날짜타입 데이터 연도만 추출
    # 상장경과연수 계산
    # 기타 자산 생성
    # 이자보상배율산출용이자비용 산출
    # f = open('./preprocess_list.txt', 'a')
    # form = st.form(key='preprocessing')
    # selects = form.multiselect('처리할 전처리 과정을 모두 선택하세요.',['기준일자 표 생성', '날짜 데이터 연도 추출', '상장경과연수 계산', '기타 자산 생성', '이자보상배율산출용이자비용 산출'])
    # name = form.text_input('전처리 과정 저장명')
    # submit = form.form_submit_button('저장', type='primary')
    # if submit:
    #     f.write(name+'\n')
    # f.close()
    selects = st.multiselect('처리할 전처리 과정을 모두 선택하세요.',['기준일자 표 생성', '날짜 데이터 연도 추출', '상장경과연수 계산', '기타 자산 생성', '이자보상배율산출용이자비용 산출'])
    name = st.text_input('전처리 과정 저장명')
    submit = st.button('저장', type='primary')
    if submit:
        id02_01_01(selects, name)
    
    # container = st.container()
    container = st.container(border=True)
    container.subheader('저장된 전처리 리스트')
    # preprocess_list = open('./preprocess_list.txt','r')
    # lines = preprocess_list.readlines()
    datas = id02_03_01()
    for data in datas:
        container.write(data['전처리명'])
    return True