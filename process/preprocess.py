import pandas as pd
import numpy as np
import json
from process.op_rules import (
    C_PRE_OP_001, C_PRE_OP_002, C_PRE_OP_003, C_PRE_OP_004,
    C_PRE_OP_005, C_PRE_OP_006, C_PRE_OP_007, C_PRE_OP_008,
    C_PRE_OP_009
)

def change_f(f_json, data):
    """
    continuos한 value를 categorical한 value로 반환
    
    Parameters
    ----------
    f_json : dict
        {continuos 칼럼의 값 : 등장 확률}
    data : int
        적용할 데이터의 각 요소
    
    Examples
    --------
    >>> f_json = {1: 0.7, 2: 0.2, 3: 0.1}
    >>> data = 1
    >>> x = change_f(f_json, data)
    >>> x
    F
    
    >>> f_json = {1: 0.7, 2: 0.2, 3: 0.1}
    >>> data = 0
    >>> x = change_f(f_json, data)
    >>> x
    0
    """
    if str(data) in f_json.keys():
        return "F"
    else:
        return data

class Preprocess():
    """
    데이터 전처리를 관리
    
    Parameters
    ----------
    orig_data_path : str
        원천데이터의 경로 지정.
        판다스 데이터 프레임으로 로드
    f_rate_path : str
        contious 변수를 categorical 변수로 바꾸기 위한 메타데이터의 경로 지정.
    dict_cart_items_path : str
        CART 적용 방식을 저장한 json 경로.
    meta_path : str
        기업 CB 실제 데이터 형식, 영문코드 저장한 json 경로.
    """
    def __init__(self, original_data_path=None, f_rate_path='meta/convert_f_rates.json',
                 meta_path='meta/meta_기업CB.json'):
        self.df_original = pd.read_csv(original_data_path)
        self.col_sort_list = self.df_original.columns.to_list()
        with open(f_rate_path, "r") as f:
            self.f_info = json.load(f)
            f.close()
        with open(meta_path) as f:
            self.meta_items = json.load(f)
            f.close()
        super().__init__()
    
    def convert_eng_code_to_kor_code(self):
        """
        편의상 영문코드로 되어있는 컬럼명을 한글로 변환
        """
        rename_dict = {self.meta_items[col]['코드'] : col for col in self.meta_items}
        self.df_original = self.df_original.rename(columns = rename_dict)
        self.df_original = self.df_original[list(self.meta_items.keys())]
    
    # continuos_col_to_categorical_col
    def C_PRE_001(self):
        """
        change_f 함수의 적용.
        """
        self.df_original_f = self.df_original.copy()
        for col in self.f_info:
            self.df_original_f[col] = self.df_original_f[col].apply(lambda x: change_f(self.f_info[col], x) if x != 0 else x)
            self.df_original_f[col] = self.df_original_f[col].astype(str)
    
    # sort_orig_data
    def C_PRE_002(self, sort_by_cols=['자산총계','설립일자','소유건축물실거래가합계','소유건축물건수'],
                  key_cols=['기준년월', '가명식별자']):
        """
        데이터 정렬 및 동일한 회사 그룹화
        
        Parameters
        ----------
        sort_by_cols : list
            sorting 기준 컬럼.
            데이터에서 회사별 그룹화의 역활 동시 수행
        key_cols : list
            데이터 key 값으로 작용하는 기준일자, 가명식별자 컬럼명.
        df_sorted_path : str
            sorting 완료데이터 저장 위치.
        """
        self.df_sorted = self.df_original_f.sort_values(by = sort_by_cols, ascending = False)
        self.bins = (self.df_sorted.drop(columns = key_cols).duplicated() == False).to_numpy()
    
    # sorted_data_to_unique_data
    def C_PRE_003(self, date_key_col='기준년월',
                  df_ordernum_bsdt_path='output/df_ordernum_bsdt.csv'):
        """
        그룹화 적용한 데이터 중복 제거
        
        Parameters
        ----------
        date_key_col : str
            기준년월 컬럼명.
        df_real_unique_path : str
            중복제거 데이터 저장 위치.
        df_ordernum_bsdt_path : str
            정렬된 기준년월 표 저장 위치.
        """
        self.df_real_unique = self.df_sorted[self.bins].iloc[:, 2:]
        self.df_ordernum_BSDT = pd.concat([pd.DataFrame(self.bins.cumsum(), columns = ['order_num']), self.df_sorted.reset_index()[date_key_col]], axis = 1)
        self.df_ordernum_BSDT.to_csv(df_ordernum_bsdt_path, index = False, encoding = 'utf-8-sig')
    
    # date_to_year
    def C_PRE_004(self, date_cols=['설립일자', '상장일자', '상장폐지일자']):
        """
        날짜타입데이터를 연도만 추출하여 데이터프레임에 할당
        
        Parameters
        ----------
        date_cols : list
            날짜 타입 데이터 컬럼명 리스트.
        """
        for col in date_cols:
            self.df_real_unique[col] = self.df_real_unique[col].map(lambda x : int(x//10000) if pd.isna(x) == False else x)
    
    # year_to_num
    def C_PRE_005(self, start_date_col='상장일자', end_date_col='상장폐지일자'):
        """
        상장폐지일자의 경우 상장일자로부터 경과 연수로 변환함.
        
        Parameters
        ----------
        listd_dt : str
            상장일자 데이터 컬럼명.
        listd_abol_dt : str
            상장폐지일자 데이터 컬럼명.
        """
        end_date_col_notnan = self.df_real_unique[self.df_real_unique[end_date_col].isna() == False][end_date_col]
        start_date_col_notnan = self.df_real_unique[self.df_real_unique[end_date_col].isna() == False][start_date_col]
        self.df_real_unique[self.df_real_unique[end_date_col].isna() == False][end_date_col] = end_date_col_notnan - start_date_col_notnan
    
    # pre_calculate
    def C_PRE_006(self):
        """
        산출항목을 계산하기 위한 요소 중 누락된 컬럼을 추가.
        """
        self.df_train = self.df_real_unique.copy()
        self.df_train = C_PRE_OP_001(self.df_train)
        self.df_train = C_PRE_OP_002(self.df_train)
        self.df_train = C_PRE_OP_003(self.df_train)
        self.df_train = C_PRE_OP_004(self.df_train)
        self.df_train = C_PRE_OP_005(self.df_train)
        self.df_train = C_PRE_OP_006(self.df_train)
        self.df_train = C_PRE_OP_007(self.df_train)
        self.df_train = C_PRE_OP_008(self.df_train)
        self.df_train = C_PRE_OP_009(self.df_train)
    
    # fill_na_categorical_col
    def C_PRE_007(self, nan_categorical_cols=['주소지시군구']):
        """
        categorical 컬럼 중 결측치가 있는 컬럼의 처리.
        
        Parameters
        ----------
        nan_categorical_cols : list
            결측치가 있는 categorical 컬럼명 리스트.
        """
        for col in nan_categorical_cols:
            assert (self.df_train[col] == '-').sum() == 0, f"checking symbol of {col}"
            self.df_train[col] = self.df_train[col].replace(np.nan, '-').map(lambda x : str(x))