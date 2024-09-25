import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt



def C_PRE_OP_001(df):
    '''
    if 영업이자보상배율 != 0 and 영업이자보상배율 != -77777777777777
    영업이익이자보상배율산출용이자비용 = 영업손익 / 영업이자보상배율
    else if 영업이자보상배율 = -77777777777777
    영업이익이자보상배율산출용이자비용 = 0
    '''
    df['이자보상배율산출용이자비용'] = df.apply(
        lambda row: row['영업손익'] / row['영업이익이자보상배율']
        if row['영업이익이자보상배율'] != 0 and row['영업이익이자보상배율'] != -77777777777777
        else 0 if row['영업이익이자보상배율'] == -77777777777777
        else None,
        axis=1
    )
    return df


def C_PRE_OP_002(df):
    '''
    기타비유동자산 = 비유동자산 - 유형자산 - 무형자산 - 투자자산
    '''
    df['기타비유동자산'] = df.apply(
        lambda row: row['비유동자산'] - row['유형자산'] - row['무형자산'] - row['투자자산'],
        axis=1
    )
    return df


def C_PRE_OP_003(df):
    '''
    기타유동부채 = 유동부채 - 단기차입금 - 매입채무
    '''
    df['기타유동부채'] = df.apply(
        lambda row: row['유동부채'] - row['단기차입금'] - row['매입채무'],
        axis=1
    )
    return df


def C_PRE_OP_004(df):
    '''
    기타비유동부채 = 비유동부채 - 차입금
    '''
    df['기타비유동부채'] = df.apply(
        lambda row: row['비유동부채'] - row['차입금'],
        axis=1
    )
    return df


def C_PRE_OP_005(df):
    '''
    전기유동자산 = (100 * 유동자산) / (재무비율_유동자산증가율 + 100)
    '''
    df['전기유동자산'] = df.apply(
        lambda row: (100 * row['유동자산']) / (row['재무비율_유동자산증가율'] + 100)
        if (row['재무비율_유동자산증가율'] + 100) != 0 else 0,
        axis=1
    )
    return df


def C_PRE_OP_006(df):
    '''
    이자수익 = 당기순이익  + 이자비용 + 법인세 - EBIT
    '''
    df['이자수익'] = df.apply(
        lambda row: row['당기순이익'] + row['이자비용'] + row['법인세'] - row['EBIT'],
        axis=1
    )
    return df


def C_PRE_OP_007(df):
    '''
    상각비 = EBITDA - EBIT
    '''
    df['상각비'] = df.apply(
        lambda row: row['EBITDA'] - row['EBIT'],
        axis=1
    )
    return df


def C_PRE_OP_008(df):
    '''
    전기유형자산 = (100 * 유형자산) /(재무비율_유형자산증가율 + 100)
    '''
    df['전기유형자산'] = df.apply(
        lambda row: (100 * row['유형자산']) / (row['재무비율_유형자산증가율'] + 100)
        if (row['재무비율_유형자산증가율'] + 100) != 0 else 0,
        axis=1
    )
    return df


def C_PRE_OP_009(df):
    '''
    총차입금 = 재무비율_차입금의존도 * 자본총계 / 100
    '''
    df['총차입금'] = df.apply(
        lambda row: row['재무비율_차입금의존도'] * row['자본총계'] / 100,
        axis=1
    )
    return df




# def C_PRE_OP_002(df):
#     '''
#     df[금융비용] 에 대한 식
#     if df['EBITDA/금융비용'] = -77777777777777   then df['금융비용'] = 0
#     else if df['EBITDA/금융비용'] != 0  then df['금융비용'] = df['EBITDA']/df['EBITDA/금융비용']
#     '''
#     df['금융비용'] = df.apply(
#         lambda row: 0 if row['EBITDA/금융비용'] == -77777777777777
#         else row['EBITDA'] / row['EBITDA/금융비용'] if row['EBITDA/금융비용'] != 0
#         else None,
#         axis=1
#     )
#     return df



def C_OP_000(df):
    """외감구분이 2인 (비외감) row의 법인세, 계속사업이익, 중단산업손익은 0으로 update """
    idx = df['외감구분'] == 2
    df.loc[idx, '법인세'] = 0
    df.loc[idx, '중단산업손익'] = 0
    return df

def C_OP_001(df): #검증완료 by eschoi
    """유동자산 = 당좌자산 + 재고자산"""
    df["유동자산"] = df.eval("당좌자산 + 재고자산")
    return df


def C_OP_002(df):  #검증완료 by eschoi
    """비유동자산 = 유형자산 + 무형자산 + 투자자산 + 기타비유동자산"""
    try:
        df["비유동자산"] = df.eval("유형자산 + 무형자산 + 투자자산 + 기타비유동자산")
    except:
        df["비유동자산"] = df.eval("유형자산 + 무형자산 + 투자자산")
    return df


def C_OP_003(df):  #검증완료 by eschoi
    """자산총계 = 유동자산 + 비유동자산"""
    df["자산총계"] = df.eval("유동자산 + 비유동자산")
    return df


def C_OP_004(df):
    """유동부채 = 단기차입금 + 매입채무 + 기타유동부채"""
    try:
        df["유동부채"] = df.eval("단기차입금 + 매입채무 + 기타유동부채")
    except:
        df["유동부채"] = df.eval("단기차입금 + 매입채무")
    return df


def C_OP_005(df):
    """비유동부채 = 차입금 + 기타비유동부채"""
    df["비유동부채"] = df.eval("차입금 + 기타비유동부채")
    return df


def C_OP_006(df):
    """납입자본 = 자기자본(납입자본금) + 자본잉여금"""
    df["납입자본"] = df["자기자본(납입자본금)"] + df["자본잉여금"]
    return df


def C_OP_007(df):
    """부채총계 = 유동부채 + 비유동부채"""
    df["부채총계"] = df.eval("유동부채 + 비유동부채")
    return df


def C_OP_008(df):
    """자본총계 = 자기자본(납입자본금) + 자본잉여금 + 이익잉여금 + 자본조정 + 기타포괄손익누계액"""
    df["자본총계"] = df["자기자본(납입자본금)"] + df.eval("자본잉여금 + 이익잉여금 + 자본조정 + 기타포괄손익누계액")
    return df


def C_OP_009(df):
    """유보금 = 자본잉여금 + 이익잉여금"""
    df["유보금"] = df.eval("자본잉여금 + 이익잉여금")
    return df


def C_OP_010(df):
    """
    적립금비율 = (자본잉여금 + 자본조정 + 기타포괄손익누계액 + 이익잉여금) / 자기자본(납입자본금) * 100
    """

    conditions = [
        (df.eval("자본잉여금 + 자본조정 + 기타포괄손익누계액 + 이익잉여금") == 0),   # 분자가 0인 경우
        (df['자기자본(납입자본금)'] == 0)   # 분모가 0인 경우
    ]
    choices = [-77777777777777, 0]

    df['적립금비율'] = np.select(conditions, choices, default=None)  # 조건에 맞춰서 값을 선택

    # 조건에 해당하지 않는 경우에 대한 적립금비율 계산
    mask = ~df['적립금비율'].notna()
    df.loc[mask, '적립금비율'] = (df.loc[mask, '자본잉여금'] +
                                   df.loc[mask, '자본조정'] +
                                   df.loc[mask, '기타포괄손익누계액'] +
                                   df.loc[mask, '이익잉여금']) / df.loc[mask, '자기자본(납입자본금)'] * 100

    return df


def C_OP_011(df):
    """매출총이익 = 매출액 - 매출원가"""
    df["매출총이익"] = df.eval("매출액 - 매출원가")
    return df


def C_OP_012(df):
    """법인세비용차감전순이익 = 매출총이익 - 판매비와관리비"""
    df["법인세비용차감전순이익"] = df.eval("매출총이익 - 판매비와관리비")
    return df


def C_OP_013(df):
    """영업손익 = 법인세비용차감전순이익"""
    df["영업손익"] = df["법인세비용차감전순이익"]
    return df


def C_OP_014(df):
    """법인세차감전순이익 = 영업손익 + 영업외수익 - 영업외비용"""
    df["법인세차감전순이익"] = df.eval("영업손익 + 영업외수익 - 영업외비용")
    return df


def C_OP_015(df):
    """계속사업이익 = 법인세차감전순이익 - 법인세"""
    df["계속사업이익"] = df["법인세차감전순이익"] - df["법인세"]
    df["계속사업이익"][df['외감구분'] == 2] = 0
    return df


def C_OP_016(df):
    """당기순이익 = 계속사업이익 - 중단산업손익"""
    result = pd.Series(np.full((df.shape[0]), np.nan))
    result[df["외감구분"] == 1] = df.eval("계속사업이익 - 중단산업손익")[df["외감구분"] == 1]
    df["당기순이익"] = result
    return df


def C_OP_017(df):
    """현금흐름 = 영업활동현금흐름 + 투자활동현금흐름 + 재무활동현금흐름"""
    df["현금흐름"] = df.eval("영업활동현금흐름 + 투자활동현금흐름 + 재무활동현금흐름")
    return df


def C_OP_018(df):
    """이자보상배율 = 영업손익 / 이자비용"""
    mask = df["이자보상배율산출용이자비용"] == 0
    result = df.eval("이자보상배율산출용이자비용 / 이자비용")
    result[mask] = result[mask].map(lambda x :  -77777777777777)
    df["이자보상배율"] = result
    return df


def C_OP_019(df):  # 수정완료 by eschoi 1115 
    """부채상환계수 = (이자비용 + 매출채권처분손실(당기))/(단기차입금 + 이자비용 + 매출채권처분손실(당기)) * 100"""
    
    mask = (df["단기차입금"] + df["이자비용"]+ df["매출채권처분손실(당기)"]) == 0
    result = (
        (df["이자비용"] + df["매출채권처분손실(당기)"])
        / (df["단기차입금"] + df["이자비용"] + df["매출채권처분손실(당기)"])
        * 100
    )
    result[mask] = result[mask].map(lambda x : -77777777777777.00)
    df["부채상환계수"] = result
    return df


def C_OP_020(df):  # 수정완료 by eschoi 1115 
    """청산가치율 = (유동자산 + 유형자산)/자산총계 * 100"""
    mask = df["자산총계"] == 0
    result = df.eval("(유동자산 + 유형자산)/자산총계 * 100")
    result[mask] = result[mask].map(lambda x : -77777777777777)
    df["청산가치율"] = result
    return df


def C_OP_021(df):
    """청산가치 = 유동자산 + 유형자산"""
    df["청산가치"] = df.eval("유동자산 + 유형자산")
    return df


def C_OP_022(df):
    """순운전자본 = 유동자산 - 유동부채"""
    df["순운전자본"] = df.eval("유동자산 - 유동부채")
    return df


def C_OP_023(df):
    """재무비율_부채비율 = 부채총계 / 자본총계 * 100"""
    mask = df["자본총계"] == 0
    result = df.eval("부채총계 / 자본총계 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_부채비율"] = result
    return df


def C_OP_024(df):
    """재무비율_자기자본비율 = 자본총계 / 자산총계 * 100"""
    mask = df["자산총계"] == 0
    result = df.eval("자본총계 / 자산총계 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_자기자본비율"] = result
    return df


def C_OP_025(df):
    """재무비율_유동비율 = 유동자산 / 유동부채 * 100"""
    mask = df["유동부채"] == 0
    result = df.eval("유동자산 / 유동부채 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_유동비율"] = result
    return df


def C_OP_026(df):
    """재무비율_당기순이익율 = 당기순이익 / 매출액 * 100"""
    mask = df["매출액"] == 0
    result = df.eval("당기순이익 / 매출액 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_당기순이익율"] = result
    return df


def C_OP_027(df):
    """재무비율_매출원가율 = 매출원가 / 매출액 * 100"""
    mask = df["매출액"] == 0
    result = df.eval("매출원가 / 매출액 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_매출원가율"] = result
    return df


def C_OP_028(df):
    """재무비율_판관비율 = 판매비와관리비 / 매출액 * 100"""
    mask = df["매출액"] == 0
    result = df.eval("판매비와관리비 / 매출액 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_판관비율"] = result
    return df


def C_OP_029(df):
    """재무비율_총자산순이익률 = 당기순이익 / 자산총계 * 100"""
    mask = df["자산총계"] == 0
    result = df.eval("당기순이익 / 자산총계 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_총자산순이익률"] = result
    return df


def C_OP_030(df):
    """재무비율_영업이익율 = 영업손익 / 매출액 * 100"""
    mask = df["매출액"] == 0
    result = df.eval("영업손익 / 매출액 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_영업이익율"] = result
    return df


def C_OP_031(df):
    """단기차입금의존도 = 단기차입금 / 자산총계 * 100"""
    mask = df["자산총계"] == 0
    result = df.eval("단기차입금 / 자산총계 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["단기차입금의존도"] = result
    return df


def C_OP_032(df):
    """당좌비율 = 당좌자산 / 유동부채 * 100"""
    mask = df["유동부채"] == 0
    result = df.eval("당좌자산 / 유동부채 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["당좌비율"] = result
    return df


def C_OP_033(df):
    """순차입금비율 = 순차입금 / 자본총계 * 100"""
    mask = df["자본총계"] == 0
    result = df.eval("순차입금 / 자본총계 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["순차입금비율"] = result
    return df


def C_OP_034(df):
    """순운전자본회전율 = 매출액 / 순운전자본"""
    mask = df["순운전자본"] == 0
    result = df.eval("매출액 / 순운전자본")
    result[mask] = result[mask].map(lambda x : 0)
    df["순운전자본회전율"] = result
    return df


def C_OP_035(df):
    """총자본회전율 = 매출액 / 자본총계"""
    mask = df["자본총계"] == 0
    result = df.eval("매출액 / 자본총계")
    result[mask] = result[mask].map(lambda x : 0)
    df["총자본회전율"] = result
    return df


def C_OP_036(df):
    """매출총이익율 = 매출총이익 / 매출액 * 100"""
    mask = df["매출액"] == 0
    result = df.eval("매출총이익 / 매출액 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["매출총이익율"] = result
    return df


def C_OP_037(df):
    """EBITDA마진율 = (법인세비용차감전순이익 + 금융비용) / 매출액 * 100"""
    mask = df["매출액"] == 0
    result = (df["법인세비용차감전순이익"] + df["금융비용"]) / df["매출액"] * 100
    result[mask] = result[mask].map(lambda x : 0)
    df["EBITDA마진율"] = result
    return df


def C_OP_038(df):
    """OCF/매출액비용 = 영업활동현금흐름 / 매출액 * 100"""
    mask = df["매출액"] == 0
    result = df.eval("영업활동현금흐름 / 매출액 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["OCF/매출액비용"] = result
    return df


def C_OP_039(df):
    """소유건축물권리침해여부 = 사업장권리침해여부"""
    df["소유건축물권리침해여부"] = df["사업장권리침해여부"]
    return df


def C_OP_040(df): # 수정완료 by eschoi 1115 
    """부채상환계수.1 = (이자비용 + 매출채권처분손실(당기))/(단기차입금 + 이자비용 + 매출채권처분손실(당기)) * 100"""
    
    # mask = (df["단기차입금"] + df["이자비용"]+ df["매출채권처분손실(당기)"]) == 0
    result = (
        (df["이자비용"] + df["매출채권처분손실(당기)"])
        / (df["단기차입금"] + df["이자비용"] + df["매출채권처분손실(당기)"])
        * 100
    )
    # result[mask] = result[mask].map(lambda x : -77777777777777.00)
    df["부채상환계수.1"] = result 
    return df





def method_op_0(df_orig, df_syn):
    """
    실제 데이터를 기반으로 매출채권(전기) 값을 예측하는 과정을 수행.
    RandomForestRegressor를 사용하여 모델을 학습하고,
    합성 데이터에 대한 예측 결과를 df_syn에 추가하여 반환.
    """

    # NaN 체크 및 처리
    df_orig.dropna(subset=["매출채권","매출채권(전기)"], inplace=True)

    # 학습 데이터 준비
    X = df_orig["매출채권"].values.reshape(-1, 1)
    y = df_orig["매출채권(전기)"]

    # 학습/테스트 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 모델 학습
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 실제 데이터에 대한 예측
    y_pred = rf.predict(X_test)

    # 합성 데이터에 대한 예측
    X_synthetic = df_syn["매출채권"].values.reshape(-1, 1)
    y_synthetic_pred = rf.predict(X_synthetic)

    # 예측된 값을 합성 데이터에 추가
    df_syn["매출채권(전기)"] = y_synthetic_pred

    return df_syn  # 합성 데이터에 예측 결과 추가하여 반환


def method_op_1(df_orig, df_syn):
    '''자산총계(전기) = f(자산총계)'''
    # NaN 체크 및 처리
    df_orig.dropna(subset=["자산총계","자산총계(전기)"], inplace=True)

    # 학습 데이터 준비
    X = df_orig["자산총계"].values.reshape(-1, 1)
    y = df_orig["자산총계(전기)"]

    # 학습/테스트 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 모델 학습
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 실제 데이터에 대한 예측
    y_pred = rf.predict(X_test)

    # 합성 데이터에 대한 예측
    X_synthetic = df_syn["자산총계"].values.reshape(-1, 1)
    y_synthetic_pred = rf.predict(X_synthetic)

    # 예측된 값을 합성 데이터에 추가
    df_syn["자산총계(전기)"] = y_synthetic_pred

    return df_syn  # 합성 데이터에 예측 결과 추가하여 반환


def method_op_2(df_orig, df_syn):
    """전기자본총계 = f(자본총계)"""

    # NaN 체크 및 처리
    df_orig.dropna(subset=["자본총계", "전기자본총계"], inplace=True)

    # 학습 데이터 준비
    X = df_orig["자본총계"].values.reshape(-1, 1)
    y = df_orig["전기자본총계"]

    # 학습/테스트 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 모델 학습
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 실제 데이터에 대한 예측
    y_pred = rf.predict(X_test)

    # 합성 데이터에 대한 예측
    X_synthetic = df_syn["자본총계"].values.reshape(-1, 1)
    y_synthetic_pred = rf.predict(X_synthetic)

    # 예측된 값을 합성 데이터에 추가
    df_syn["전기자본총계"] = y_synthetic_pred

    return df_syn  # 합성 데이터에 예측 결과 추가하여 반환


def method_op_3(df_orig, df_syn):
    """전기매출액 = f(매출액)"""

    # NaN 체크 및 처리
    df_orig.dropna(subset=["매출액", "전기매출액"], inplace=True)

    # 학습 데이터 준비
    X = df_orig["매출액"].values.reshape(-1, 1)
    y = df_orig["전기매출액"]

    # 학습/테스트 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 모델 학습
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 실제 데이터에 대한 예측
    y_pred = rf.predict(X_test)

    # 합성 데이터에 대한 예측
    X_synthetic = df_syn["매출액"].values.reshape(-1, 1)
    y_synthetic_pred = rf.predict(X_synthetic)

    # 예측된 값을 합성 데이터에 추가
    df_syn["전기매출액"] = y_synthetic_pred

    return df_syn  # 합성 데이터에 예측 결과 추가하여 반환


def method_op_4(df_orig, df_syn):
    """전기법인세차감전순이익 = f(법인세차감전순이익)"""

    # NaN 체크 및 처리
    df_orig.dropna(subset=["법인세차감전순이익", "전기법인세차감전순이익"], inplace=True)

    # 학습 데이터 준비
    X = df_orig["법인세차감전순이익"].values.reshape(-1, 1)
    y = df_orig["전기법인세차감전순이익"]

    # 학습/테스트 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 모델 학습
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 실제 데이터에 대한 예측
    y_pred = rf.predict(X_test)

    # 합성 데이터에 대한 예측
    X_synthetic = df_syn["법인세차감전순이익"].values.reshape(-1, 1)
    y_synthetic_pred = rf.predict(X_synthetic)

    # 예측된 값을 합성 데이터에 추가
    df_syn["전기법인세차감전순이익"] = y_synthetic_pred

    return df_syn  # 합성 데이터에 예측 결과 추가하여 반환

def method_op_5(df_orig, df_syn):
    """당기순이익 = f(법인세차감전순이익) 외감 2 case"""
    # NaN 체크 및 처리
    df_orig[df_orig['외감구분'] == 2].dropna(subset=["법인세차감전순이익", "당기순이익"], inplace=True)

    # 학습 데이터 준비
    X = df_orig["법인세차감전순이익"].values.reshape(-1, 1)
    y = df_orig["당기순이익"]

    # 학습/테스트 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 모델 학습
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 실제 데이터에 대한 예측
    y_pred = rf.predict(X_test)

    idx = df_syn[df_syn['외감구분'] == 2].index
    # 합성 데이터에 대한 예측
    X_synthetic = df_syn.loc[idx, "법인세차감전순이익"].values.reshape(-1, 1)
    y_synthetic_pred = rf.predict(X_synthetic)

    # 예측된 값을 합성 데이터에 추가
    df_syn.loc[idx, "당기순이익"] = y_synthetic_pred.reshape(-1, 1)
    return df_syn

def method_op_6(df_orig, df_syn):
    """당기순이익(전기) = f(당기순이익)"""

    # NaN 체크 및 처리
    df_orig.dropna(subset=["당기순이익", "당기순이익(전기)"], inplace=True)

    # 학습 데이터 준비
    X = df_orig["당기순이익"].values.reshape(-1, 1)
    y = df_orig["당기순이익(전기)"]

    # 학습/테스트 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 모델 학습
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 실제 데이터에 대한 예측
    y_pred = rf.predict(X_test)

    # 합성 데이터에 대한 예측
    X_synthetic = df_syn["당기순이익"].values.reshape(-1, 1)
    y_synthetic_pred = rf.predict(X_synthetic)

    # 예측된 값을 합성 데이터에 추가
    df_syn["당기순이익(전기)"] = y_synthetic_pred

    return df_syn  # 합성 데이터에 예측 결과 추가하여 반환





def C_POST_OP_001(df):
    """전기영업이익 = 전기법인세차감전순이익"""
    df["전기영업이익"] = df["전기법인세차감전순이익"]
    return df


def C_POST_OP_002(df):
    """재무비율_총자산증가율 = (자산총계 - 자산총계(전기)) / 자산총계 * 100"""
    mask = df["자산총계"] == 0
    result = (df["자산총계"] - df["자산총계(전기)"]) / df["자산총계"] * 100
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_총자산증가율"] = result
    return df


def C_POST_OP_003(df):
    """당기순이익증가율 = (당기순이익 / 당기순이익(전기)) * 100 - 100"""
    mask = df["당기순이익(전기)"] == 0
    result = df["당기순이익"] / df["당기순이익(전기)"] * 100 - 100
    result[mask] = result[mask].map(lambda x : 0)
    df["당기순이익증가율"] = result
    return df


def C_POST_OP_004(df):
    """영업이익증가율 = (영업손익 / 전기영업이익) * 100 - 100"""
    mask = df["전기영업이익"] == 0
    result = df.eval("(영업손익 / 전기영업이익) * 100 - 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["영업이익증가율"] = result
    return df


def C_POST_OP_005(df):
    """자기자본순이익율 = 당기순이익 / ((자본총계 + 전기자본총계) / 2) * 100"""
    mask = df.eval("자본총계 + 전기자본총계") == 0
    result = df.eval("당기순이익 / ((자본총계 + 전기자본총계) / 2) * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["자기자본순이익율"] = result
    return df


def C_POST_OP_006(df):
    """재무비율_매출액증가율 = (매출액 - 전기매출액) / 전기매출액 * 100"""
    mask = df["전기매출액"] == 0
    result = df.eval("(매출액 - 전기매출액) / 전기매출액 * 100")
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_매출액증가율"] = result
    return df


def C_POST_OP_007(df):
    """재무비율_매출채권회전율 = 매출액/((매출채권 + 매출채권(전기))/2)"""
    mask = (df["매출채권"] + df["매출채권(전기)"]) == 0
    result = df["매출액"] / ((df["매출채권"] + df["매출채권(전기)"]) / 2)
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_매출채권회전율"] = result
    return df


def C_POST_OP_008(df):
    """재무비율_총자산회전율 = 매출액 / ((자산총계 + 자산총계(전기)) / 2)"""
    mask = (df["자산총계"] + df["자산총계(전기)"]) == 0
    result = df["매출액"] / ((df["자산총계"] + df["자산총계(전기)"]) / 2)
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_총자산회전율"] = result
    return df


def C_POST_OP_009(df): # 수정
    """재무비율_유동자산증가율 =  유동자산 / 전기유동자산 * 100 - 100"""
    mask = df["전기유동자산"] == 0
    result = df["유동자산"] / (df["전기유동자산"] * 100 - 100)
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_유동자산증가율"] = result
    return df


def C_POST_OP_010(df): # 수정
    """EBIT =   당기순이익 - 이자수익 + 이자비용 + 법인세비용"""
    result = df.eval('당기순이익 - 이자수익 + 이자비용 + 법인세')
    df["EBIT"] = result
    return df


def C_POST_OP_011(df): # 수정
    """EBITDA = EBIT + 상각비"""
    result = df.eval('EBIT + 상각비')
    df["EBITDA"] = result
    return df


def C_POST_OP_012(df): # 수정
    """재무비율_재고자산회전율 = 매출원가 / 재고자산"""
    mask = df["재고자산"] == 0
    result = df.eval('매출원가 / 재고자산')
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_재고자산회전율"] = result
    return df


def C_POST_OP_013(df): # 수정
    """EBITDA/금융비용 = EBITDA / 금융비용"""
    mask = df["금융비용"] == 0
    result = df.eval('EBITDA / 금융비용')
    result[mask] = result[mask].map(lambda x : 0)
    df["EBITDA/금융비용"] = result
    return df


def C_POST_OP_014(df): # 수정
    """재무비율_유형자산증가율 = 유형자산 / 전기유형자산 * 100 - 100"""
    mask = df["전기유형자산"] == 0
    result = df.eval('유형자산 / 전기유형자산 * 100 - 100')
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_유형자산증가율"] = result
    return df


def C_POST_OP_015(df): # 수정
    """재무비율_자기자본이익률(ROE)  = 당기순이익 / 자본총계"""
    mask = df["자본총계"] == 0
    result = df.eval('당기순이익 / 자본총계')
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_자기자본이익률(ROE)"] = result
    return df


def C_POST_OP_016(df): # 수정
    """재무비율_차입금의존도 =  총차입금 / 자본총계 * 100"""
    mask = df["자본총계"] == 0
    result = df.eval('총차입금 / 자본총계 * 100')
    result[mask] = result[mask].map(lambda x : 0)
    df["재무비율_차입금의존도"] = result
    return df


def C_POST_OP_017(df): # 수정
    """영업이익이자보상배율 = 이자보상배율"""
    result = df.eval('이자보상배율')
    df["영업이익이자보상배율"] = result
    return df