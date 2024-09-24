import os
import pandas as pd
import argparse

from eval import calc_eval
from meta import Column_type


parser = argparse.ArgumentParser(
    prog=__name__,
    description='기업 CB 합성데이터 평가지표 산출'
)
parser.add_argument(
    '-original_data_path',
    '--original-data-path',
    type=str,
    required=True,
    help='원본데이터 저장 경로'
)
parser.add_argument(
    '-fake_datas_dir',
    '--fake-datas-dir',
    type=str,
    required=True,
    help='합성데이터들이 저장된 폴더 경로'
)

args = parser.parse_args()

orig_data_path = args.original_data_path
syn_datas_path = args.fake_datas_dir


KEY_COL = Column_type.KEY_COL
BASE_YM_COL = Column_type.BASE_YM_COL
baseym_list = Column_type.BASE_YM_LIST

categorical_columns = Column_type.evaluation_config['categorical_columns']
float_columns = Column_type.evaluation_config['float_columns']
int_columns = Column_type.evaluation_config['integer_columns']


dfReal = pd.read_csv(orig_data_path)
high_cardinality_cols = [col for col in categorical_columns if len(dfReal[col].unique()) >= 20]

for syn_data_path in os.listdir(syn_datas_path):
    if os.path.splitext(orig_data_path)[1] == ".parquet":
        df_orig = pd.read_parquet(orig_data_path).reset_index(drop=True)
    else:
        df_orig = pd.read_csv(orig_data_path)

    if os.path.splitext(syn_data_path)[1] == ".parquet":
        df_syn = pd.read_parquet(f'{syn_datas_path}/{syn_data_path}').reset_index(drop=True)
    else:
        df_syn = pd.read_csv(f'{syn_datas_path}/{syn_data_path}')
    df_syn[[BASE_YM_COL, KEY_COL]] = df_syn[[BASE_YM_COL, KEY_COL]].astype(str)
    df_orig[[BASE_YM_COL, KEY_COL]] = df_orig[[BASE_YM_COL, KEY_COL]].astype(str)
    print(f"파일 데이터 모양: [df_orig]: {df_orig.shape}, [df_syn]: {df_syn.shape}")

    syn_baseym_list = sorted(df_syn[BASE_YM_COL].unique())
    if syn_baseym_list != baseym_list:
        print(f"합성데이터에 존재하는 기준년월 ({syn_baseym_list})만 사용해 지표를 산출합니다.")
        df_orig = df_orig[df_orig[BASE_YM_COL].isin(syn_baseym_list)].reset_index(drop=True)
        baseym_list = syn_baseym_list

    curr_high_cardinality_cols = list(
        set(df_orig.columns).intersection(high_cardinality_cols)
    )

    unseened_cols = list(set(df_orig.columns).difference(df_syn.columns))
    if len(unseened_cols) > 0:
        print("원천데이터와 합성데이터의 컬럼 구성이 다릅니다. 원천데이터에 존재하는 컬럼만 사용해 지표를 산출합니다.")
    df_syn = df_syn[df_orig.columns]

    print(f"최종 데이터 모양: [df_orig]: {df_orig.shape}, [df_syn]: {df_syn.shape}")

    cate_cols = [col for col in df_orig.columns if col in categorical_columns]
    num_cols = [col for col in df_orig.columns if col in float_columns]
    int_cols = [col for col in df_orig.columns if col not in categorical_columns + float_columns]
    int_nan_cols_value_count = df_orig[int_cols].isna().sum()
    int_nan_cols = int_nan_cols_value_count[int_nan_cols_value_count > 0]
    change_dtypes = dict()
    for col in df_orig.columns:
        if col in cate_cols:
            change_dtypes[col] = "str"
        elif col in num_cols or col in int_nan_cols:
            change_dtypes[col] = "float"
        elif col in [BASE_YM_COL, KEY_COL]:
            pass
        else:
            change_dtypes[col] = "int"

    df_orig[cate_cols] = df_orig[cate_cols].fillna('empty')
    df_orig = df_orig.astype(change_dtypes)

    df_syn[cate_cols] = df_syn[cate_cols].fillna('empty')
    df_syn = df_syn.astype(change_dtypes)
    assert all(df_orig.dtypes.values == df_syn.dtypes.values), '데이터 타입이 일치하지 않습니다.'

    # 원본과 합성 결합
    df_orig = pd.concat(
        [df_orig, pd.Series([0] * len(df_orig), name='is_syn')],
        axis=1
    )
    df_syn = pd.concat(
        [df_syn, pd.Series([1] * len(df_syn), name='is_syn')],
        axis=1
    )
    df_merge_all = pd.concat(
        [df_orig, df_syn], ignore_index=True
    )
    print(f"df_merge_all: {df_merge_all.shape}")

    # categorical type으로 변경
    print("set categorical dtypes")
    for col in cate_cols:
        df_merge_all[col] = df_merge_all[col].astype(
            pd.CategoricalDtype(categories=sorted(df_merge_all[col].unique()))
        )  # plot axis 순서 설정용

    res_all = dict()

    for baseym in baseym_list:

        df_merge = (
            df_merge_all[df_merge_all[BASE_YM_COL] == baseym]
            .drop(columns=BASE_YM_COL)
            .reset_index(drop=True)
        )
        res_dict = calc_eval(
            df_merge=df_merge, # df
            KEY_COL=KEY_COL, # cust_rnno
            PARTITION_COL='is_syn', # is_syn
            curr_high_cardinality_cols=curr_high_cardinality_cols,
            cate_cols=cate_cols,
            num_cols=num_cols,
            int_cols=int_cols,
        )
        res_all[baseym] = res_dict
        print(f'{baseym}의 결과 : {res_dict}')

    ## 테스트 결과 저장
    # df_summary.to_csv(save_path, index=False)