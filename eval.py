import pandas as pd
import numpy as np
from typing import List, Union
import copy
from pandas.api.types import is_numeric_dtype
from scipy.special import rel_entr
from dython.nominal import associations
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.linear_model import LogisticRegression

def _scale_points(points, min_range, max_range):
    """지점들을 지정된 최소와 최대값에 따라 스케일링

    Args:
        points (ndarray): 1D array containing data with numerical type
        min_range (float): 스케일링 기준이 될 최소값
        max_range (float): 스케일링 기준이 될 최대값

    Returns:
        ndarray: 1D array containing data with numerical type
    """
    if max_range != min_range:
        eps = 1E-6  # floating point 오류 방지
        min_range = min_range - eps
        max_range = max_range + eps
        
        points += -(np.min(points))
        points = points / (np.max(points) / (max_range - min_range))
        points += min_range
    else:  # 단일값
        points = np.array([float(min_range) for _ in points])

    return np.round(points, 10)


def _edge_points_to_bin_ranges(points):
    """구간 분할 지점들로 구간별 좌측/우측 값 산출

    Args:
        points (ndarray): 1D array containing data with numerical type

    Returns:
        pd.DataFrame: 구간별 범위 정보 (좌측/우측)
    """
    df_bin_ranges = (
        pd.concat(
            [
                pd.Series(points[:-1], name="bin_left"),
                pd.Series(points[1:], name="bin_right"),
            ],
            axis=1,
        )
        .reset_index()
        .rename(columns={"index": "bin_index"})
    )
    df_bin_ranges.loc[:, "bin_value"] = None

    return df_bin_ranges


def _bin_ranges_to_edge_points(bin_left_vals, bin_right_vals):
    """구간의 좌측/우측 값으로 구간 분할 지점들을 산출

    Args:
        bin_left_vals (ndarray): 1D array containing data with numerical type
        bin_right_vals (ndarray): 1D array containing data with numerical type

    Returns:
        ndarray: 1D array containing data with numerical type
    """
    return np.append(bin_left_vals, bin_right_vals[-1])


def get_cate_frequencies(arr):
    """범주형 데이터에 대해 데이터 구간별 빈도 산출

    Args:
        arr (ndarray): 1D array containing data with categorical type

    Returns:
        tuple: 구간별 빈도, 구간 정보
    """
    df_cate_frequencies = (
        pd.value_counts(arr, dropna=False).rename("frequency").reset_index()
    )
    df_cate_frequencies.columns = ["bin_value", "frequency"]

    df_cate_frequencies.loc[:, "bin_left"] = np.nan
    df_cate_frequencies.loc[:, "bin_right"] = np.nan
    df_cate_frequencies = df_cate_frequencies.reset_index().rename(
        columns={"index": "bin_index"}
    )

    return df_cate_frequencies["frequency"].values, df_cate_frequencies[
        ["bin_index", "bin_left", "bin_right", "bin_value"]
    ].reset_index(drop=True)


def get_bin_frequencies(arr, bin_edges=None, bin_method="equal_width", n_bins=10):
    """수치형 데이터에 대해 데이터 구간별 빈도 산출

    구간 분할 지점이 제공된 경우 이를 사용해 분할하고, 제공되지 않은 경우
    다음 중 지정된 구간화 방식을 사용해 데이터를 `n_bins`개로 구간화함:
    - `equal_width` (등간격):
        각 구간이 균등한 너비를 가지도록 구간화
    - `equal_freq` (등빈도):
        각 구간이 균등한 빈도를 가지도록 구간화. 완전히 동일하진 않을 수 있음
    (데이터가 제공된 구간 범위를 벗어날 경우 가장 끝의 구간으로 분류됨)

    Args:
        arr (ndarray): 1D array containing data with numerical type
        bin_edges (ndarray): 1D array containing data with numerical type
        bin_method (str): 구간화 방식 {`equal_width`, `equal_freq`}
        n_bins (int): 구간 개수

    Returns:
        tuple:
            ndarray (1D array containing data with numerical type),
            ndarray (1D array containing data with numerical type)
    """
    if bin_edges is None:  # 구간 분할 지점 지정되지 않음
        # 고유값 수가 구간 개수보다 적은 경우 구간 개수를 고유값 수로 설정
        n_unique_vals = arr.nunique()
        if n_unique_vals < n_bins:
            n_bins = n_unique_vals

        bin_edges = np.arange(0, n_bins + 1) / (n_bins) * 100

        if bin_method == "equal_width":  # 등간격
            bin_edges = _scale_points(bin_edges, np.nanmin(arr), np.nanmax(arr))

        elif bin_method == "equal_freq":  # 등빈도
            bin_edges = np.stack([np.nanpercentile(arr, p) for p in bin_edges])

    else:  # 구간 분할 지점 지정됨
        if min(arr) < bin_edges[0]:  # 최소값이 최소 구간을 벗어남
            bin_edges[0] = min(arr)

        if max(arr) > bin_edges[-1]:  # 최대값이 최대 구간을 벗어남
            bin_edges[-1] = max(arr)

    # 지정된 구간 분할 지점으로 히스토그램 생성
    frequencies, bin_edges = np.histogram(a=arr, bins=bin_edges)

    return frequencies, bin_edges

def generate_freq_by_feat(
    data, bin_method="equal_width", n_bins=10, feat_index_col="feat_index"
):
    """변수별 데이터 구간 및 분포 정보 생성

    Args:
        data (ndarray): 2D array containing data with numerical type
        bin_method (str): 구간화 방식 {`equal_width`, `equal_freq`}
        n_bins (int): 구간 개수
        feat_index_col (str): 변수 인덱스 컬럼명
    Returns:
        pd.DataFrame: 변수별 데이터 구간 및 분포 정보
    """
    df_freq_by_feat = []
    for i in range(data.shape[1]):
        if is_numeric_dtype(data.iloc[:, i]):
            frequencies, bin_edges = get_bin_frequencies(
                data.iloc[:, i], bin_method=bin_method, n_bins=n_bins
            )
            df_bin_info = _edge_points_to_bin_ranges(bin_edges)
        else:
            frequencies, df_bin_info = get_cate_frequencies(data.iloc[:, i])

        df_freq_by_feat.append(
            pd.concat(
                [
                    pd.Series([i] * len(frequencies), name="feat_index"),
                    df_bin_info,
                    pd.Series(frequencies, name="base_freq"),
                ],
                axis=1,
            )
        )

    df_freq_by_feat = pd.concat(df_freq_by_feat).reset_index(drop=True)

    # 구간별 비율 산출
    df_freq_by_feat["base_pct"] = df_freq_by_feat[
        "base_freq"
    ] / df_freq_by_feat.groupby(feat_index_col)["base_freq"].transform("sum")

    return df_freq_by_feat

def calc_jsd_by_feat(df_all_freq_by_feat, feat_index_col="feat_index"):
    """기준 및 대상 데이터 분포 정보 사용해 변수별 JS Divergence 산출

    Args:
       df_all_freq_by_feat (pd.DataFrame): 변수별 기준 및 대상 데이터 구간 및 분포
       feat_index_col (str): 변수 인덱스 컬럼명

    Returns:
       pd.DataFrame: JS Divergence 계산 테이블
    """
    df_jsd_by_feat = copy.deepcopy(df_all_freq_by_feat)

    # P(x)와 Q(x)의 평균
    df_jsd_by_feat["mean_pct"] = df_jsd_by_feat[["base_pct", "target_pct"]].mean(axis=1)

    # 구간별 JS divergence 산출
    df_jsd_by_feat["jsd_i"] = (
        df_jsd_by_feat.groupby(feat_index_col)
        .apply(
            lambda x: 0.5 * (
                # D_KL(P||M)  # P(x)  # M
                # D_KL(Q||M)  # Q(x)  # M
                rel_entr(x["target_pct"], x["mean_pct"]) + rel_entr(x["base_pct"], x["mean_pct"])
            ) / np.log(2)
        ).squeeze().reset_index(drop=True)
    )

    return df_jsd_by_feat

def calc_target_freq_by_feat(
    df_base_freq_by_feat, target_data, feat_index_col="feat_index"
):
    """기준 데이터 구간 사용해 계산한 대상 분포 정보 추가

    Args:
       df_base_freq_by_feat (pd.DataFrame): 변수별 기준 데이터 구간 및 분포
       target_data (ndarray): 2D array containing data with numerical type
       feat_index_col (str): 변수 인덱스 컬럼명

    Returns:
       pd.DataFrame: 기준 및 대상 분포 정보
    """
    # 변수 Index 목록
    feat_indices = sorted(df_base_freq_by_feat[feat_index_col].unique())

    df_all_freq_by_feat = []
    for i, feat_index in enumerate(feat_indices):
        if is_numeric_dtype(target_data.iloc[:, i]):  # 수치형 변수
            # 해당 변수의 기준 데이터 구간별 빈도
            df_base_freqs = df_base_freq_by_feat[
                df_base_freq_by_feat[feat_index_col] == feat_index
            ].reset_index(drop=True)
            base_frequencies = df_base_freqs["base_freq"].values

            # 기준 데이터의 구간 분할 지점 가져오기
            base_bin_edges = _bin_ranges_to_edge_points(
                df_base_freq_by_feat[
                    df_base_freq_by_feat[feat_index_col] == feat_index
                ]["bin_left"].values,
                df_base_freq_by_feat[
                    df_base_freq_by_feat[feat_index_col] == feat_index
                ]["bin_right"].values,
            )
            # 기준 데이터의 구간 분할 지점으로 대상 데이터 빈도 산출
            target_frequencies, target_bin_edges = get_bin_frequencies(
                target_data.iloc[:, i], bin_edges=base_bin_edges
            )

            # 기준 데이터와 대상 데이터 구간별 빈도 정보 결합
            df_full = pd.concat(
                [df_base_freqs, pd.Series(target_frequencies, name="target_freq")],
                axis=1,
            )

        else:  # 범주형 변수
            # 해당 변수의 기준 데이터 구간별 빈도
            df_base_freqs = df_base_freq_by_feat[
                df_base_freq_by_feat[feat_index_col] == feat_index
            ].reset_index(drop=True)

            # 대상 데이터 빈도 산출
            target_frequencies, df_bin_info = get_cate_frequencies(
                target_data.iloc[:, i]
            )
            df_target_freqs = pd.concat(
                [df_bin_info, pd.Series(target_frequencies, name="target_freq")], axis=1
            )

            # 기준 데이터와 대상 데이터 구간별 빈도 정보를 범주 값으로 join
            df_full = df_base_freqs.merge(
                df_target_freqs[["bin_value", "target_freq"]],
                how="outer",
                on=["bin_value"],
            )
            df_full[feat_index_col] = feat_index

            # 구간 빈도 결측값 대체 (기준 데이터 또는 대상 데이터에 존재하지 않는 범주)
            df_full["base_freq"] = df_full["base_freq"].fillna(0)
            df_full["target_freq"] = df_full["target_freq"].fillna(0)

            # 구간 Index 결측값을 알수없는 범주로 대체 (기준 데이터에는 존재하나 대상 데이터에 존재하지 않는 범주)
            UNKNOWN_CATE_FLAG = -1
            df_full["bin_index"] = df_full["bin_index"].fillna(UNKNOWN_CATE_FLAG)

            # 알수없는 범주 빈도 총합 집계
            df_unknown = df_full[df_full["bin_index"] == UNKNOWN_CATE_FLAG]
            df_unknown = (
                df_unknown.groupby(by=[feat_index_col, "bin_index"])[
                    ["base_freq", "target_freq"]
                ]
                .sum()
                .reset_index()
            )

            # 알수없는 범주의 집계된 빈도로 업데이트
            df_full = df_full[df_full["bin_index"] != UNKNOWN_CATE_FLAG]
            df_full = pd.concat([df_full, df_unknown], ignore_index=True)

        df_all_freq_by_feat.append(df_full)

    # 변수별 구간별 빈도 정보 결합
    df_all_freq_by_feat = pd.concat(df_all_freq_by_feat).reset_index(drop=True)

    # 구간별 비율 산출
    df_all_freq_by_feat["base_pct"] = df_all_freq_by_feat[
        "base_freq"
    ] / df_all_freq_by_feat.groupby(feat_index_col)["base_freq"].transform("sum")
    df_all_freq_by_feat["target_pct"] = df_all_freq_by_feat[
        "target_freq"
    ] / df_all_freq_by_feat.groupby(feat_index_col)["target_freq"].transform("sum")

    # 범주 값 컬럼의 NaN을 None으로 대체
    df_all_freq_by_feat["bin_value"] = np.where(
        pd.isnull(df_all_freq_by_feat["bin_value"]),
        None,
        df_all_freq_by_feat["bin_value"],
    )

    # 최종 데이터 타입으로 변환
    df_all_freq_by_feat[feat_index_col] = df_all_freq_by_feat[feat_index_col].astype(
        int
    )
    df_all_freq_by_feat["bin_index"] = df_all_freq_by_feat["bin_index"].astype(int)
    df_all_freq_by_feat["base_freq"] = df_all_freq_by_feat["base_freq"].astype(int)
    df_all_freq_by_feat["target_freq"] = df_all_freq_by_feat["target_freq"].astype(int)

    return df_all_freq_by_feat

def calc_max_jsd_f2r(df, key_col="cust_rnno", partition_col="is_syn", n_bins=20):
    """첫번째 파티션을 원본으로 보고, 나머지 파티션 별 데이터의 분포와 비교 (First to Rest)"""
    partitions = sorted(df[partition_col].unique())
    assert (
        len(partitions) > 1
    ), f"There must be at least 2 partitions for `{partition_col}` (Current: {len(partitions)})."
    df_by_part = [
        df[df[partition_col] == part]
        .drop(columns=[key_col, partition_col])
        .reset_index(drop=True)
        for part in partitions
    ]

    # 값이 아예 생성되지 않은 컬럼 제외
    null_cols = []
    for df_part in df_by_part[1:]:
        null_cols = null_cols + [
            x[0]
            for x in df_by_part[0].columns
            if x[0] not in [y[0] for y in df_part.columns]
        ]
    for i, df_part in enumerate(df_by_part):
        for col in null_cols:
            if col in df_part:
                df_part = df_part.drop(columns=col)
        df_by_part[i] = df_part

    # 기준 분포 정보 생성
    df_base_dist = generate_freq_by_feat(
        df_by_part[0], bin_method="equal_width", n_bins=n_bins
    )

    # 대상 분포별 JSD 계산
    df_jsd = pd.concat(
        [
            calc_jsd_by_feat(calc_target_freq_by_feat(df_base_dist, x))
            .groupby("feat_index")["jsd_i"]
            .sum()
            .rename(partitions[i])
            for i, x in enumerate(df_by_part)
        ],
        axis=1,
    )
    df_jsd.index = df_by_part[0].columns

    return df_jsd.max(axis=1).to_dict()


def get_jsd_by_col(
    df_to_plot,
    key_col="발급회원번호",
    partition_col="is_syn",
    n_bins=20,
):
    """컬럼별 JSD 계산
    Args:
        df_to_plot (pd.DataFrame): 산출 대상 컬럼 데이터
    """
    try:
        jsd_by_col = calc_max_jsd_f2r(
            df_to_plot, key_col=key_col, partition_col=partition_col, n_bins=n_bins
        )
        for col in [x for x in df_to_plot.columns if x not in [key_col, partition_col]]:
            if col not in jsd_by_col:  # 산출되지 않은 컬럼의 jsd는 -1로 설정
                print(f"jsd for {col} not created")
                jsd_by_col[col] = -1
    except Exception as e:
        print('jsd 산출 에러.')
        jsd_by_col = {  # 오류 발생 시 모든 컬럼의 jsd를 -1로 설정
            x: -1 for x in df_to_plot.columns if x not in [key_col, partition_col]
        }
    return jsd_by_col

def calc_pmse(probs, y):
    N = len(y)
    r = y.sum() / N

    return ((probs - r) ** 2).sum() / N

def get_pmse(
    df_merge: pd.DataFrame,
    high_cardinality_cols: List[str] = None,
    partition_col: str = "is_syn",
    cate_cols: List[str] = None,
) -> float:
    """pMSE를 계산하기 위한 샘플링부터 전처리까지 전반적인 작업 수행
    Args:
        df_merge: 'partition_col' = [0,1] 로 구분 가능한 원본과 합성이 병합된 데이터
        high_cardinality_cols: label enoding 수행할 매우 많은 범주수를 갖는 컬럼명들
        partition_col: 원본과 합성의 구분 컬럼명
    Returns:
        float: pMSE값
    """
    global X_ppc, y
    if high_cardinality_cols is None:
        high_cardinality_cols = []
    if cate_cols is None:
        cate_cols = []

    ## 동일 크기로 샘플링
    n1 = (df_merge[partition_col] == 0).sum()
    n2 = (df_merge[partition_col] == 1).sum()

    if n1 == n2:  # 동일 사이즈
        pass
    elif n1 > n2:  # 합성이 더 적음
        df_merge = pd.concat(
            [
                df_merge[df_merge[partition_col] == 0].sample(n=n2, random_state=0),
                df_merge[df_merge[partition_col] == 1],
            ]
        )
    else:  # 합성이 더 많음
        df_merge = pd.concat(
            [
                df_merge[df_merge[partition_col] == 0],
                df_merge[df_merge[partition_col] == 1].sample(n=n1, random_state=0),
            ]
        )
    

    X = df_merge.drop([partition_col], axis=1)
    y = df_merge[partition_col]

    # 범주형 처리
    enc = LabelEncoder()
    for col in high_cardinality_cols:
        X[col] = enc.fit_transform(X[col])

    X_ppc = X.copy()

    # 수치형 변수 처리
    # - 결측치를 평균값으로 대체
    # - Min-Max scale
    for col in X_ppc.select_dtypes(include=[np.number]).columns:
        X_ppc[col].fillna(X_ppc[col].mean(), inplace=True)
        X_ppc[col] = minmax_scale(X_ppc[col])

    # 범주형 변수 더미화
    X_ppc = pd.get_dummies(X_ppc)
    print(f"X_ppc: {X_ppc.shape}")

    print("start lr")
    model = LogisticRegression(max_iter=5000, random_state=0, n_jobs=1)
    model.fit(X_ppc, y)
    pred_probs = model.predict_proba(X_ppc)[:, 1]
    pmse = calc_pmse(pred_probs, y)
    
    print(f"pmse: {pmse}")

    return pmse



def get_corrdiff(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    categorical_columns: Union[List[str], str] = "auto",
    sample_size=None
) -> float:
    """corr_diff의 계산하여 모든 컬럼쌍의 평균값을 출력
    수치형은 피어슨 상관계수, 범주형은 Theil'U 값을 계산
    Args:
        real: 원본데이터
        fake: 합성데이터
        categorical_columns: 범주형 컬럼명 리스트
        condition (str): 세부 데이터명
        sample_size (int): 계산에 사용할 표본 크기
    Returns:
        float: corr_diff의 평균값
    """
    if sample_size is not None and len(real) > sample_size:
        real = real.sample(sample_size, random_state=0)
        
    real_corr = associations(
        real,
        nominal_columns=categorical_columns,
        nom_nom_assoc="theil",
        compute_only=True,
        multiprocessing=True,
        max_cpu_cores=16
    )
    
    if sample_size is not None and len(fake) > sample_size:
        fake = fake.sample(sample_size, random_state=0)
    
    fake_corr = associations(
        fake,
        nominal_columns=categorical_columns,
        nom_nom_assoc="theil",
        compute_only=True,
        multiprocessing=True,
        max_cpu_cores=16
    )

    corr_dist = real_corr["corr"] - fake_corr["corr"]
    corr_dist = np.abs(corr_dist.values).mean()

    return corr_dist

def calc_eval(
    df_merge: pd.DataFrame,
    KEY_COL: str = None,
    PARTITION_COL: str = None,
    curr_high_cardinality_cols: List[str] = None,
    cate_cols: List[str] = None,
    num_cols: List[str] = None,
    int_cols: List[str] = None,
):

    cond_cols = list(df_merge.columns)
    cond_cate_cols = [
        x for x in list(set(cate_cols).intersection(cond_cols)) if x != KEY_COL
    ]
    cond_num_cols = list(set(num_cols).intersection(cond_cols))
    cond_int_cols = list(set(int_cols).intersection(cond_cols))
    print(
        f"total: {len(cond_cols)}, cate: {len(cond_cate_cols)}, num: {len(cond_num_cols)}, int: {len(cond_int_cols)}"
    )
    cond_curr_high_cardinality_cols = list(
        set(curr_high_cardinality_cols).intersection(cond_cols)
    )
    
    # 하위 디렉토리 생성

    ### JSD 계산
    print("start jsd")
    jsd_by_col = get_jsd_by_col(
        df_merge,
        key_col=KEY_COL,
        partition_col=PARTITION_COL,
        n_bins=20,
    )
    jsd = np.mean(list(jsd_by_col.values()))
    print(f"jsd: {jsd}")

    ### pMSE 계산
    print("start pmse")
    pmse = get_pmse(
        df_merge.drop(columns=[KEY_COL]),
        high_cardinality_cols=cond_curr_high_cardinality_cols,
        partition_col=PARTITION_COL,
        cate_cols=cond_cate_cols,
    )

    ### Corr.diff 계산
    print("start corr. diff")
    df_orig = df_merge[df_merge[PARTITION_COL] == 0].reset_index(drop=True)
    df_syn = df_merge[df_merge[PARTITION_COL] == 1].reset_index(drop=True)

    corr_diff = get_corrdiff(
        real=df_orig.drop(columns=[KEY_COL, PARTITION_COL]),
        fake=df_syn.drop(columns=[KEY_COL, PARTITION_COL]),
        categorical_columns=cond_cate_cols,
    )
    print(f"corr_diff: {corr_diff}")

    # 결과 취합
    res = {
        "js_divergence": jsd,
        "pmse": pmse,
        "corr_diff": corr_diff,
    }

    return res