import streamlit as st
from views import ex4
from utils import id04_01_01, id04_01_02

def createPage():
    st.header('후처리')
    
    # form = st.form(key='derivation')
    # form.selectbox('칼럼을 선택하세요', [])
    # expression = form.text_input('파생항목 생성을 위한 계산식을 입력하세요')
    # name = form.text_input('파생항목 저장명')
    # submit = form.form_submit_button('저장')

    # st.write('해당 파생항목 저장명을 입력 후 저장하세요.')

    # if submit:
    #     st.write(f'hello {name}')

    # 기준년월 -> 정확히 뭘하는건지..?
    # 상장폐지일자 재처리
    # 연도 -> 년월일
    # 주소지시군구 -> 정확히 뭘하는건지2222
    # 기타항목 삭제
    # 가명식별자 랜덤 생성
    f = open('./postprocess_list.txt','a')
    
    # form = st.form(key='postprocessing')
    # selects = form.multiselect('처리할 후처리 과정을 모두 선택하세요.',['날짜 데이터 형식 복원', '상장폐지일자 재처리', '기타 항목 삭제', '가명식별자 랜덤 생성'])
    # name = form.text_input('후처리 과정 저장명')
    # submit = form.form_submit_button('저장', type='primary')
    selects = st.multiselect('처리할 후처리 과정을 모두 선택하세요.',['날짜 데이터 형식 복원', '상장폐지일자 재처리', '기타 항목 삭제', '가명식별자 랜덤 생성'])
    name = st.text_input('후처리 과정 저장명')
    submit = st.button('저장', type='primary')
    if submit:
        id04_01_02(selects, name)
    
    container = st.container()
    # container = st.container(border=True)
    container.subheader('저장된 후처리 리스트')
    # postprocess_list = open('./postprocess_list.txt','r')
    # lines = postprocess_list.readlines()
    
    datas = id04_01_01()
    for data in datas:
        st.write(data['후처리명'])
    
    return True