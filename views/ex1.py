import streamlit as st

def createPage():
    st.header('학습 모델 설계')
    model_container = st.container(border=True)
    c1, c2, c3 = model_container.columns([0.5, 0.5, 0.1])

    with c1:
        model_name = st.text_input('학습명')
        ds_name = st.selectbox('데이터셋명', ['Corp_CB_Dataset_2024'])
        st.write('데이터 분할')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('항목')
            st.write('학습')
            st.write('테스트')
        with col2:
            st.write('데이터 소스')
            st.selectbox('',['Corp_CB_Training'], label_visibility="collapsed")
            st.selectbox('',['Corp_CB_Testing'], label_visibility="collapsed")
        with col3:
            st.write('데이터 수')
            st.number_input('데이터수1', step=1, label_visibility="collapsed")
            st.number_input('데이터수2', step=1, label_visibility="collapsed")

        previous_train = st.text_input('이전 학습 불러오기')
        
    with c2:
        st.selectbox('알고리즘', ['CTAB_GAN_For_Corp_CB_V12', 'CART', 'CT_GAN_Transformer_ensemble_For_Corp_CB', 'Transformer_for_Credit_card', 'Diffusion_For_syntheis'])
        st.selectbox('공개범위', ['전체공개', '나만보기'])

    with c3:
        st.write('')
        st.write('')
        st.button('설명', type='primary')
        
    t1, t2, t3 = st.tabs(['공통 파라미터', '알고리즘 파라미터', 'Neural Network'])
    with t1:
        st.selectbox('초기화방법', ['Xavier uniform'])
        st.selectbox('최적화방법', ['Adam'])
        st.number_input('자동저장주기', 50, step=1)
        st.number_input('배치사이즈',step=1)
        st.slider('Learning rate', 0.000, 1.0)
        st.slider('Dropout ratio', 0.0, 1.0)
        st.number_input('학습수행횟수(epoch)', step=1)

    st.button('저장', type='primary')
    
    return True