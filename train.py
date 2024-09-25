from synthpop import Synthpop
import pandas as pd
import os
import random
import numpy as np
import yaml
import math
from process.op_rules import *
from process.preprocess import Preprocess
from process.postprocess import Postprocess
import argparse
import warnings
import shutil
import json

warnings.filterwarnings('ignore')

def main():
    
    parser = argparse.ArgumentParser(prog=__name__, description='기업 CB 데이터 CART 기법을 이용한 합성')
    parser.add_argument('-project_id', '--project-id', type=str, required=True, help='테스트명')

    args = parser.parse_args()

    project_id = args.project_id
    
    os.mkdir(f'./projects/{project_id}')
    os.mkdir(f'./projects/{project_id}/original_data')
    os.mkdir(f'./projects/{project_id}/preprocessed_data')
    os.mkdir(f'./projects/{project_id}/generated_data')
    os.mkdir(f'./projects/{project_id}/complete_data')
    
    with open(f'model_config/{project_id}/config.yaml') as file:
        project_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(f'original_data/{project_config["데이터셋ID"]}/config.yaml') as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(f'preprocess_config/{project_config["전처리ID"]}.yaml') as file:
        preprocess_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(f'postprocess_config/{project_config["후처리ID"]}.yaml') as file:
        postprocess_config = yaml.load(file, Loader=yaml.FullLoader)

    project_config['상태'] = '전처리중'
    with open(f'model_config/{project_id}/config.yaml', 'w', encoding='utf-8') as file:
        print(project_id)
        yaml.dump(project_config, file)

    shutil.copy(
        data_config['데이터경로'],
        os.path.join(f'./projects/{project_id}/original_data', os.path.split(data_config['데이터경로'])[-1])
    )

    f_info_path = './meta/convert_f_rates.json'
    meta_path = './meta/meta_기업CB.json'
    cart_types = dict()
    for i in project_config['integer_cols']:
        cart_types[i] = 'integer'
    for i in project_config['float_cols']:
        cart_types[i] = 'float'
    for i in project_config['category_cols']:
        cart_types[i] = 'category'
    with open('./meta/cart_types.json') as f:
        cart_types = json.load(f)

    # 연도 컬럼 리스트
    date_col_list = ['설립일자', '상장일자', '상장폐지일자']
    
    # 전처리
    preprocess = Preprocess(original_data_path = data_config['데이터경로'], f_rate_path = f_info_path, meta_path=meta_path)

    preprocess.convert_eng_code_to_kor_code()
    preprocess.C_PRE_001()
    preprocess.C_PRE_002(sort_by_cols=['자산총계', '설립일자', '소유건축물실거래가합계', '소유건축물건수'], key_cols=['기준년월', '가명식별자'])
    preprocess.C_PRE_003(date_key_col='기준년월', df_ordernum_bsdt_path=f'./projects/{project_id}/preprocessed_data/order_bsdt.csv')
    preprocess.C_PRE_004(date_cols=date_col_list)
    preprocess.C_PRE_005(start_date_col = '상장일자', end_date_col = '상장폐지일자')
    preprocess.C_PRE_006()
    preprocess.C_PRE_007(nan_categorical_cols = ['주소지시군구'])
    preprocess.df_train.to_csv(f'./projects/{project_id}/preprocessed_data/preprocess.csv', index=False)
        
    project_config['상태'] = '학습중'
    with open(f'model_config/{project_id}/config.yaml', 'w', encoding='utf-8') as file:
        yaml.load(project_config, file)

    # 학습
    cart = Synthpop()
    cart_apply_types = dict()
    for i, j in cart_types.items():
        if i in preprocess.df_train.columns:
            cart_apply_types[i] = j
    cart.fit(preprocess.df_train.copy(), cart_apply_types)
    
    # 생성 부분
    df_fake_unique = cart.generate(len(preprocess.df_train))

    # difussion-factor
    diffussion_list = list()
    for i, j in cart_types.items():
        if j != 'category' and i not in date_col_list:
            diffussion_list.append(i)
        if i == '소유건축물실거래가합계':
            break
    for i in diffussion_list:
        idx_diff_full = df_fake_unique.index
        idx_diff_choice = np.random.choice(idx_diff_full, size=int(len(idx_diff_full)/10*7), replace=False) # 대략 70퍼센트 추출
        df_fake_unique.loc[idx_diff_choice, i] = df_fake_unique.loc[idx_diff_choice, i].map(lambda x : int((0.95 + random.random()*0.1) * x) 
                                                                                            if x not in [k for k in range(-1000, 1000)]+[-99999999] and pd.isna(x) == False else x)

    # 자기자본(납입자본금) floor
    df_fake_unique['자기자본(납입자본금)'] = df_fake_unique['자기자본(납입자본금)'].map(lambda x : math.floor(x/1000)*1000) # 1000단위

    df_fake_unique.to_csv(f'./projects/{project_id}/generated_data/generate.csv', index = False)

    project_config['상태'] = '후처리중'
    with open(f'model_config/{project_id}/config.yaml', 'w', encoding='utf-8') as file:
        yaml.load(project_config, file)

    # 연산
    df_fake_unique_op = C_OP_000(df_fake_unique).copy()
    
    cnt = 1
    while True:
        try:
            func_ = "C_OP_" + "0" * (3 - len(str(cnt))) + str(cnt)
            df_fake_unique_op = globals()[func_](df_fake_unique_op)
            cnt += 1
        except:
            break
    df_fake_unique_op = df_fake_unique_op.replace([np.inf, -np.inf], 0)

    forest_apply_list = ["매출채권(전기)", "자산총계(전기)", "전기자본총계", "전기매출액", "전기법인세차감전순이익", '당기순이익', "당기순이익(전기)"]
    df_fake_unique_rf = df_fake_unique_op.copy()
    
    cnt = 0
    while True:
        try:
            func_ = "method_op_" + str(cnt)
            df_fake_unique_rf = globals()[func_](preprocess.df_original, df_fake_unique_rf)
            cnt += 1
        except:
            break
    df_fake_unique_rf[forest_apply_list] = df_fake_unique_rf[forest_apply_list].applymap(math.floor)
    df_fake_unique_rf[forest_apply_list] = df_fake_unique_rf[forest_apply_list].replace(np.nan, None).astype(int)

    df_fake_unique_post_op = df_fake_unique_rf.copy()
    
    cnt = 1
    while True:
        try:
            func_ = "C_POST_OP_" + "0" * (3 - len(str(cnt))) + str(cnt)
            df_fake_unique_post_op = globals()[func_](df_fake_unique_post_op)
            cnt += 1
        except:
            break
    df_fake_unique_post_op = df_fake_unique_post_op.replace([np.inf, -np.inf], 0)
    
    # 후처리
    postprocess = Postprocess(synthetic_data_path=df_fake_unique_post_op, df_ordernum_bsdt_path=f'./projects/{project_id}/preprocessed_data/order_bsdt.csv',
                          f_rate_path=f_info_path, meta_path=meta_path)

    postprocess.C_POST_001(sort_by_cols=['자산총계','설립일자','소유건축물실거래가합계','소유건축물건수'])
    postprocess.C_POST_002(date_key_col='기준년월')
    postprocess.C_POST_003(start_date_col='상장일자', end_date_col='상장폐지일자')
    postprocess.C_POST_004(date_cols=['설립일자', '상장일자', '상장폐지일자'])
    postprocess.C_POST_005(nan_categorical_cols=['주소지시군구'])
    postprocess.C_POST_006()
    postprocess.C_POST_007()
    postprocess.C_POST_008()
    postprocess.df_fake_completed.to_csv(f'./projects/{project_id}/complete_data/complete.csv', index=False)
    
    project_config['상태'] = '정상종료'
    with open(f'model_config/{project_id}/config.yaml', 'w', encoding='utf-8') as file:
        yaml.load(project_config, file)

if __name__ == '__main__':
    main()