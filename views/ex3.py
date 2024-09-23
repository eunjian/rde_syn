import streamlit as st

def createPage():
    st.header('학습 모델 상세정보')
    
    container = st.container(border=True)
    container.subheader('학습 기본 정보')
    c1, c2 = container.columns(2)
    with c1:
        st.selectbox('학습명',['기업 CB 합성데이터셋 생성 v1.2'])
        st.selectbox('데이터셋명',['corp_CB_Dataset_2024'])
    with c2:
        st.selectbox('알고리즘',['Transformer'])
        sub_c1, _, sub_c2 = st.columns([3,5,2])
        with sub_c1:
            st.selectbox('공개범위',['전체공개'])
        with sub_c2:
            st.button('변경', type='primary')
    
    container2 = st.container(border=True)
    container2.subheader('학습 모델 정보')
    t1,t2, t3 = container2.tabs(['공통 파라미터', 'Neural Network', '학습상태'])
    
    _,col1,col2,col3,col4,col5,col6 = st.columns([5,1,1,1,1,1,1])
    with col1:
        st.button('목록으로',type='primary')
    with col2:
        st.button('텐서보드',type='primary')
    with col3:
        action = st.button('실행',type='primary')
    with col4:
        st.button('중단',type='primary')
    with col5:
        st.button('수정',type='primary')
    with col6:
        st.button('삭제',type='primary')
        
    if action:
        st.toast('학습이 완료되었습니다.')
    return True