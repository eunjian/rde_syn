import streamlit as st
import pandas as pd

    
def createPage():
    ds_dict = {}
    t1, t2 = st.tabs(['원천데이터 업로드', '데이터셋 관리'])    
    f = open('./dataset_list.txt','a')
    with t1:    
        uploaded_file = st.file_uploader('원천 데이터를 업로드해주세요.',['csv','parquet'])
        
        if uploaded_file is not None:
            c1, c2 = st.columns([5,1])
            ds_name = ''
            with c1:
                ds_name = st.text_input('데이터셋 이름을 지정해주세요')
            with c2:
                st.write('')
                st.write('')
                if st.button('저장', type='primary'):
                    c1.write(f'데이터셋의 이름: {ds_name}')
                    f.write(ds_name+'\n')
                    f.close()
            df = pd.read_csv(uploaded_file)
            col, row = df.shape
            st.write(f'데이터셋의 크기: {col}열, {row}행')
            st.write(df.head(20))
        else:
            st.info('☝️ 파일을 업로드하세요')
    with t2:
        container = st.container(border=True)
        container.subheader('지금까지 업로드된 데이터셋 목록')
        ds_list = open('./dataset_list.txt','r')
        lines = ds_list.readlines()
        for line in lines:
            container.write(line)
    return True