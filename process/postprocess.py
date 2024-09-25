import pandas as pd
import json
import datetime
import random
import math

pre_calculated_col_list = [
    "이자보상배율산출용이자비용", "기타비유동자산", "기타유동부채",
    "기타비유동부채", "전기유동자산", "이자수익","상각비",
    "전기유형자산", "총차입금"]


class Postprocess():
    """
    데이터 후처리 관리 모듈.
    df_fake  -> df_fake_unique_sorted -> df_fake_augmented              -> df_fake_completed 순 진행
    합성완료  -> 합성 후 정렬된 데이터   -> 원천데이터만큼 row 길이 복원완료 -> 후처리 process 완료
    
    Parameters
    ----------
    synthetic_data_path : str
        합성 완료 데이터 저장 경로.
    df_ordernum_bsdt_path : str
        초반 중복제거된 데이터의 기준년월을 지시하는 테이블 저장 경로.
    f_rate_path : str
        contious 변수를 categorical 변수로 바꾸기 위한 메타데이터의 경로.
    meta_path : str
        기업 CB 실제 데이터 형식, 영문코드 저장한 json 경로.
    """
    def __init__(self, synthetic_data_path='output/df_fake_unique_post_op.csv', df_ordernum_bsdt_path='output/df_ordernum_bsdt.csv',
                 f_rate_path='meta/convert_f_rates.json', meta_path='meta/meta_기업CB.json'):
        # self.df_fake = pd.read_csv(synthetic_data_path)
        self.df_fake = synthetic_data_path
        self.df_ordernum_bsdt = pd.read_csv(df_ordernum_bsdt_path)
        with open(f_rate_path, "r") as f:
            self.f_info = json.load(f)
            f.close()
        with open(meta_path) as f:
            self.meta_items = json.load(f)
            f.close()
        super().__init__()
        
    def C_POST_001(self, sort_by_cols=['자산총계','설립일자','소유건축물실거래가합계','소유건축물건수']):
        """
        합성 완료 데이터 정렬.
        
        Parameters
        ----------
        sort_by_cols : list
            정렬기준 리스트.
            리스트의 역순으로 순서대로 정렬.
        df_fake_unique_sorted_path : str
            정렬된 합성데이터 저장 경로.
        """
        self.df_fake_unique_sorted = self.df_fake.sort_values(by = sort_by_cols, ascending = False).reset_index(drop = True)
    
    def C_POST_002(self, date_key_col='기준년월'):
        """
        원천데이터의 row 길이만큼 합성데이터의 길이 복원 및 기준년월 부여.
        
        Parameters
        ----------
        date_key_col : str
            기준년월 컬럼명 지정.
        df_fake_augmented_path : str
            복원 완료된 합성데이터 저장 경로.
        """
        self.df_fake_unique_sorted = self.df_fake_unique_sorted.iloc[(self.df_ordernum_bsdt['order_num'] - 1).to_list(), :].reset_index(drop = True)
        self.df_fake_unique_sorted[date_key_col] = self.df_ordernum_bsdt[date_key_col]
        self.df_fake_augmented = self.df_fake_unique_sorted.copy()
        
    def C_POST_003(self, start_date_col='상장일자', end_date_col='상장폐지일자'):
        """
        경과연수로 표현한 상장폐지일자를 다시 연도 형식으로 복원
        
        Parameters
        ----------
        start_date_col : str
            타깃 컬럼에 + 연산할 컬럼명.
        end_date_col : str
            타깃 컬럼명.
        """
        end_date_col_notnan = self.df_fake_augmented[self.df_fake_augmented[end_date_col].isna() == False][end_date_col]
        start_date_col_notnan = self.df_fake_augmented[self.df_fake_augmented[end_date_col].isna() == False][start_date_col]
        self.df_fake_augmented[self.df_fake_augmented[end_date_col].isna() == False][end_date_col] = end_date_col_notnan - start_date_col_notnan
    
    def C_POST_004(self, date_cols=['설립일자', '상장일자', '상장폐지일자']):
        """
        연도만 추출한 날짜 형식의 컬럼 복원.
        월, 일은 1 ~ 365 중 난수 생성하여 복원.
        
        Parameters
        ----------
        date_cols : list
            날짜 형식의 데이터 컬럼명.
        """
        def f(x):
            return (datetime.datetime(1,1,1) + datetime.timedelta(x * 365 + random.randrange(0, 365, 1))).strftime('%Y%m%d')
        self.df_fake_augmented[date_cols] = self.df_fake_augmented[date_cols].applymap(lambda x : f(x) if pd.isna(x) == False else x)
    
    def C_POST_005(self, nan_categorical_cols=['주소지시군구']):
        """
        C_PRE_007에서 결측치 처리하였던 컬럼 복원.
        
        Parameters
        ----------
        nan_categorical_cols : list
            원천데이터에 결측치가 있는 categorical 컬럼명 리스트.
        """
        for col in nan_categorical_cols:
            self.df_fake_augmented[col] = self.df_fake_augmented[col].map(lambda x : None if x == '-' else x)
    
    def C_POST_006(self):
        """
        산출항목을 계산하기 위한 추가하였던 컬럼 제거
        """
        self.df_fake_augmented = self.df_fake_augmented.drop(columns = pre_calculated_col_list)
        
    def C_POST_007(self):
        """
        C_PRE_001에서 F로 바꾼 요소 복원.
        원천데이터의 등장 빈도와 거의 동일할 수 있도록 복원.
        """
        for col in self.f_info:
            prop_repository = []
            rates = self.f_info[col]
            f_total = (self.df_fake_augmented[col] == "F").sum()
            print(f"F-total : {f_total}")
            for val, rate in rates.items():
                prop_repository += [val] * math.ceil(f_total * rate)
            random.shuffle(prop_repository)
            self.df_fake_augmented[col] = self.df_fake_augmented[col].apply(lambda x: prop_repository.pop() if x == "F" else x)
            self.df_fake_augmented[col] = self.df_fake_augmented[col].astype(int)
    
    def C_POST_008(self):
        """
        가명식별자(3배수의 데이터가 생성완료된 후 복원)를 제외한 데이터의 컬럼순서 정렬 및 저장.
        
        Parameters
        ----------
        df_fake_completed_path : str
            후처리까지 모두 완료한 합성데이터 저장 경로.
        """
        self.df_fake_completed = self.df_fake_augmented.copy()
        col_sort_list = list(self.meta_items.keys())
        # col_sort_list.remove("가명식별자")
        self.df_fake_completed["가명식별자"] = list(self.df_fake_completed.index)
        self.df_fake_completed = self.df_fake_completed[col_sort_list]
        
        # rename_dict = {col : self.meta_items[col]['code'] for col in self.meta_items}
        # self.df_fake_completed = self.df_fake_augmented.rename(rename_dict)
        # self.df_fake_completed.to_csv(df_fake_completed_path, index = False, encoding = 'utf-8')
        