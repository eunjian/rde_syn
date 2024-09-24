import streamlit as st
from views import ex4

def createPage():
    st.header('후처리')
    
    form = st.form(key='derivation')
    form.selectbox('칼럼을 선택하세요', [])
    expression = form.text_input('파생항목 생성을 위한 계산식을 입력하세요')
    name = form.text_input('파생항목 저장명')
    submit = form.form_submit_button('저장')

    # st.write('해당 파생항목 저장명을 입력 후 저장하세요.')

    # if submit:
    #     st.write(f'hello {name}')
    
    container = st.container(border=True)
    container.subheader('저장된 파생항목 리스트')
    
    
    
    
    return True