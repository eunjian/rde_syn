import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from utils import (
    # 조회 관련
    id01_03_01, # 데이터
    id02_03_01, # 전처리
    id04_01_01, # 후처리
    # 모델 등록
    id03_01_02,
)

def createPage():
    st.header('학습 모델 설계')
    # model_container = st.container()
    model_container = st.container(border=True)
    c1, c2, c3 = model_container.columns([0.5, 0.5, 0.1])

    #
    DATA_LIST = id01_03_01()
    PREPROCESS_LIST = id02_03_01()
    POSTPROCESS_LIST = id04_01_01()

    with c1:
        model_name = st.text_input('학습명')
        ds_name = st.selectbox(
            '데이터셋명',
            [data['ID'] for data in DATA_LIST],
            format_func=lambda x: {data['ID']: data['데이터셋명'] for data in DATA_LIST}[x]
        )
        preprocess_name = st.selectbox(
            '전처리명',
            [data['ID'] for data in PREPROCESS_LIST],
            format_func=lambda x: {data['ID']: data['전처리명'] for data in PREPROCESS_LIST}[x]
        )
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
            train_length = st.number_input('데이터수1', step=1, label_visibility="collapsed")
            test_length = st.number_input('데이터수2', step=1, label_visibility="collapsed")

        previous_train = st.text_input('이전 학습 불러오기')
        
    with c2:
        algorithm = st.selectbox('알고리즘', ['CTAB_GAN_For_Corp_CB_V12', 'CART', 'CT_GAN_Transformer_ensemble_For_Corp_CB', 'Transformer_for_Credit_card', 'Diffusion_For_syntheis'])
        privacy_policy = st.selectbox('공개범위', ['전체공개', '나만보기'])
        postprocess_name = st.selectbox(
            '후처리명',
            [data['ID'] for data in POSTPROCESS_LIST],
            format_func=lambda x: {data['ID']: data['후처리명'] for data in POSTPROCESS_LIST}[x]
        )

    with c3:
        st.write('')
        st.write('')
        st.button('설명', type='primary')
        
    t1, t2, t3 = st.tabs(['공통 파라미터', '알고리즘 파라미터', 'Neural Network'])
    with t1:
        initialization_method = st.selectbox('초기화방법', ['Xavier uniform'])
        optimizer_method = st.selectbox('최적화방법', ['Adam'])
        saving_term = st.number_input('자동저장주기', 50, step=1)
        batch_size = st.number_input('배치사이즈',step=1)
        learning_rate = st.slider('Learning rate', 0.000, 1.0)
        dropout = st.slider('Dropout ratio', 0.0, 1.0)
        epochs = st.number_input('학습수행횟수(epoch)', step=1)
    
    with t2:
        ########################
        # csv 에 한함 추후 수정 #
        ########################
        def read_data(ds_id):
            data_path = {data['ID']: data['데이터경로'] for data in DATA_LIST}[ds_id] 
            data = pd.read_csv(data_path)
            return list(data.columns)
        ########################
        ########################
        try:
            cols = read_data(ds_name)
            float_cols, category_cols = list(), list()
            integer_cols = st.multiselect("정수 형식 컬럼", [col for col in cols if col not in float_cols + category_cols])
            float_cols = st.multiselect("실수 형식 컬럼", [col for col in cols if col not in integer_cols + category_cols])
            category_cols = st.multiselect("카테고리 형식 컬럼", [col for col in cols if col not in integer_cols + float_cols])
        except:
            pass

    submit_clicked = st.button('학습', type='primary')
    
    
    if submit_clicked:
        # if set(cols) - set(integer_cols) - set(float_cols) - set(category_cols) != set():
        if False:
            st.warning('컬럼타입 미지정', icon="⚠️")
        else:
            id03_01_02(
                model_name, # 학습명
                ds_name, # 데이터셋명
                train_length, # 학습 길이 ##
                test_length, # 테스트 길이 ##
                previous_train, # 이전 학습 ##
                algorithm, # 알고리즘
                privacy_policy, # 공개범위
                initialization_method, # 초기화방법 ##
                optimizer_method, # 최적화방법 ##
                saving_term, # 자동저장주기 ##
                batch_size, # 배치사이즈 ##
                learning_rate, # learning-rate ##
                dropout, # dropout ##
                epochs, # 에폭스 ##
                preprocess_name,
                postprocess_name,
                integer_cols,
                float_cols,
                category_cols,
            )
    
    return True