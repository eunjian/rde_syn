import yaml
import random
import glob
import os
import sys
import shutil
import pandas as pd
from process.preprocess import Preprocess
from process.postprocess import Postprocess
from synthpop import Synthpop

# 초기화
CURRENT_PATH = os.curdir + '/'
ORIGINAL_PATH = 'original_data'
PREPROCESS_PATH = 'preprocess_config'
MODEL_PATH = 'model_config'
POSTPROCESS_PATH = 'postprocess_config'
PROJECT_PATH = 'projects'

if os.path.isdir(ORIGINAL_PATH) == False:
    os.mkdir(ORIGINAL_PATH)
if os.path.isdir(PREPROCESS_PATH) == False:
    os.mkdir(PREPROCESS_PATH)
if os.path.isdir(MODEL_PATH) == False:
    os.mkdir(MODEL_PATH)
if os.path.isdir(POSTPROCESS_PATH) == False:
    os.mkdir(POSTPROCESS_PATH)
if os.path.isdir(PROJECT_PATH) == False:
    os.mkdir(PROJECT_PATH)

F_INFO_PATH = 'meta/convert_f_rates.json'
CART_TYPE_PATH = 'meta/cart_types.json'
META_PATH = 'meta/meta_기업CB.json'

# 디렉토리/파일명 생성
def randstrings():
    ascii_list = list(range(48,58)) + list(range(97, 123)) # ascii 0~9 / a~z
    ascii_list = list(map(chr, ascii_list))
    new_name =  ''.join(random.choices(ascii_list, k=32))
    return new_name

# 원천데이터 셋 등록
def id01_01_01(
    original_data_path:str,
    original_data_name:str,
):
    while True:
        folder_name = randstrings()
        if folder_name not in os.listdir(ORIGINAL_PATH):
            break
    os.mkdir(ORIGINAL_PATH + '/' + folder_name)
    shutil.move(original_data_path, ORIGINAL_PATH + '/' + folder_name)
    config_yaml = dict()
    config_yaml['ID'] = folder_name
    config_yaml['데이터경로'] = os.path.split(original_data_path)[-1]
    config_yaml['데이터셋명'] = original_data_name
    with open(ORIGINAL_PATH + '/' + folder_name + '/config.yaml', 'w') as file:
        yaml.dump(config_yaml, file)

def id01_02_01():
    pass

def id01_02_02():
    pass

def id01_02_03():
    pass

def id01_02_04():
    pass

def id01_02_05():
    pass

# 원천데이터셋 조회
def id01_03_01():
    original_list = os.listdir(ORIGINAL_PATH)
    original_config_list = list()
    for folder in original_list:
        try:
            with open(os.path.join(CURRENT_PATH, ORIGINAL_PATH, folder, 'config.yaml')) as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
            original_config_list.append(data)
        except:
            pass
    return original_config_list

def id01_03_02():
    pass

def id01_03_03():
    pass

# 전처리 함수 저장
def id02_01_01(
    args,
    preprocess_name:str=None,
):
    while True:
        file_name = randstrings() + '.yaml'
        if file_name not in os.listdir(PREPROCESS_PATH):
            break
    os.mkdir(PREPROCESS_PATH + '/' + file_name)
    config_yaml = dict()
    config_yaml['ID'] = file_name.rstrip('.yaml')
    if preprocess_name:
        config_yaml['전처리명'] = preprocess_name
    else:
        config_yaml['전처리명'] = '전처리_' + str(len(os.listdir(PREPROCESS_PATH)) + 1)
    for process, columns in args.items():
        config_yaml[process] = columns
    with open(PREPROCESS_PATH + file_name, 'w') as file:
        yaml.dump(config_yaml, file)

def id02_01_02():
    pass

def id02_02_01():
    pass

def id02_02_02():
    pass

# 전처리 목록 조회
def id02_03_01():
    config_path_list = glob.glob()
    preprocess_name_list = list()
    for config_path in config_path_list:
        with open(config_path) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            preprocess_name_list.append(data['전처리명'])
    return preprocess_name_list

def id02_03_02():
    pass

def id02_03_03():
    pass

def id02_03_04():
    pass

# 학습 목록 조회
def id03_01_01():
    model_list = os.listdir(MODEL_PATH)
    model_config_list = list()
    for folder in model_list:
        try:
            with open(os.path.join(CURRENT_PATH, MODEL_PATH, folder, 'config.yaml')) as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
            model_config_list.append(data)
        except:
            pass
    return model_config_list

# 학습 등록
def id03_01_02(
        model_name:str,
        ds_name:str,
        train_length:str,
        test_length:str,
        previous_train:str,
        algorithm:str,
        privacy_policy:str,
        initialization_method:str,
        optimizer_method:str,
        saving_term:str,
        batch_size:str,
        learning_rate:str,
        dropout:str,
        epochs:str,
        category:str='기업CB합성',
        user:str='admin',
        start_time:str='-',
        end_time:str='-',
        accuracy:str='-',
        status:str='등록완료',
    ):
    while True:
        folder_name = randstrings()
        if folder_name not in os.listdir(MODEL_PATH):
            break
    os.mkdir(MODEL_PATH + '/' + folder_name)
    config_yaml = dict()
    config_yaml['ID'] = folder_name
    config_yaml['번호'] = len(id03_01_01())
    config_yaml['카테고리'] = category
    config_yaml['알고리즘'] = algorithm
    config_yaml['학습명'] = model_name
    config_yaml['데이터셋명'] = ds_name
    config_yaml['학습셋길이'] = train_length
    config_yaml['테스트셋길이'] = test_length
    config_yaml['이전학습'] = previous_train
    config_yaml['등록자'] = user
    config_yaml['수행시작일자'] = start_time
    config_yaml['수행종료일시'] = end_time
    config_yaml['결과(Accuracy)'] = accuracy
    config_yaml['상태'] = status
    config_yaml['공개범위'] = privacy_policy
    config_yaml['초기화방법'] = initialization_method
    config_yaml['최적화방법'] = optimizer_method
    config_yaml['자동저장주기'] = saving_term
    config_yaml['배치사이즈'] = batch_size
    config_yaml['learning-rate'] = learning_rate
    config_yaml['dropout'] = dropout
    config_yaml['epochs'] = epochs
    with open(MODEL_PATH + '/' + folder_name + '/config.yaml', 'w') as file:
        yaml.dump(config_yaml, file)
    return True

def id03_01_03():
    pass

def id03_01_04(model_name:str):
    pass

def id04_01_01():
    pass

def id04_01_02():
    pass

def id04_01_03():
    pass

# 프로젝트 목록 조회
def id05_01_01():
    config_file_list = glob.glob(PROJECT_PATH + '/*/*.yaml')
    project_list = list()
    for file_path in config_file_list:
        with open(file_path) as file:
            data = yaml.load(file)
        project_list.append(data)
    return project_list

# 프로젝트 상세 조회
####################
def id05_01_02(
    project_id:str
):
    config_file = glob.glob(PROJECT_PATH + f'/{project_id}/*.yaml')[0]
    return config_file

# 프로젝트 등록
def id05_01_03(
    project_name:str,
):
    while True:
        folder_name = randstrings()
        if folder_name not in os.listdir(PROJECT_PATH):
            break
    os.mkdirs(PROJECT_PATH + '/' + folder_name + '/' + 'preprocessed_data')
    os.mkdirs(PROJECT_PATH + '/' + folder_name + '/' + 'generated_data')
    os.mkdirs(PROJECT_PATH + '/' + folder_name + '/' + 'complete_data')
    # config file
    project_config = dict()
    project_config['ID'] = folder_name
    project_config['프로젝트명'] = project_name
    project_config['상태'] = '대기중'
    project_config['데이터셋'] = ''
    project_config['전처리'] = ''
    project_config['합성'] = ''
    project_config['후처리'] = ''

# 프로젝트 수정
def id05_01_04(
    change_config_dict:dict,
):
    
    pass

# 프로젝트 삭제
def id05_01_05():
    pass

def id05_02_01():
    pass

# 전처리 수행
def id05_03_01(
    original_data_id:str,
    preprocess_id:str,
    project_id:str,
):
    with open(os.path.join(ORIGINAL_PATH, original_data_id, 'config.yaml')) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        original_data_path = data['데이터경로']
    with open(os.path.join(PREPROCESS_PATH, preprocess_id + '.yaml')) as file:
        preprocess_config = yaml.load(file, Loader=yaml.FullLoader)
    preprocess = Preprocess(
        original_data_path = original_data_path,
        f_rate_path = F_INFO_PATH,
        dict_cart_items_path = CART_TYPE_PATH,
        meta_path=META_PATH
    )
    preprocess.C_PRE_001()
    preprocess.C_PRE_002(sort_by_cols=['자산총계', '설립일자', '소유건축물실거래가합계', '소유건축물건수'], key_cols=['기준년월', '가명식별자'], df_sorted_path=df_sorted_path)
    preprocess.C_PRE_003(date_key_col='기준년월', df_real_unique_path=df_real_unique_path, df_ordernum_bsdt_path=df_ordernum_bsdt_path)
    preprocess.C_PRE_004(date_cols=date_col_list)
    preprocess.C_PRE_005(start_date_col = '상장일자', end_date_col = '상장폐지일자')
    preprocess.C_PRE_007(nan_categorical_cols = ['주소지시군구'])
    df_train = preprocess.C_PRE_008()
    df_train.to_csv(PROJECT_PATH + '/' + project_id + '/' + 'preprocessed_data' + '/' + project_id + '.csv', index=False)
    
# 학습, 합성 수행
def id05_04_01(
    project_id:str,
    model_id:str,
    generate_length:int,
):
    cart = Synthpop()
    with open(PROJECT_PATH + '/' + project_id + '/' + 'config.yaml') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    df_train = pd.read_csv(PROJECT_PATH + '/' + project_id + '/' + 'preprocessed_data' + '/' + project_id + '.csv')
    try:
        cart.fit(df_train)
    except:
        return 'preprocessing First'
    df_generated = cart.generate()
    df_generated.to_csv(PROJECT_PATH + '/' + project_id + '/' + 'generated_data' + '/' + project_id + '.csv', index=False)
    return False

# 학습 중단
def id05_04_02(
    project_id:str,
):
    pass

# 후처리 수행
def id05_05_01(
    project_id:str,
    postprocess_id:str,
):
    postprocess = Postprocess(synthetic_data_path=df_fake_unique_post_op_path, df_ordernum_bsdt_path=df_ordernum_bsdt_path,
                          f_rate_path=f_info_path, meta_path=meta_path)
    postprocess.C_POST_003(start_date_col='상장일자', end_date_col='상장폐지일자')
    postprocess.C_POST_004(date_cols=['설립일자', '상장일자', '상장폐지일자'])
    postprocess.C_POST_005(nan_categorical_cols=['주소지시군구'])
    postprocess.C_POST_006()
    postprocess.C_POST_007()
    
    

# 검증지표 출력
def id05_06_01(
    project_id:str,
):
    with open(PROJECT_PATH + '/' + project_id + '/' + 'config.yaml') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    if data['상태'] != '완료':
        return False
    with open(PROJECT_PATH + '/' + project_id + '/' + 'metrics') as file:
        metrics = file.read()
    return metrics

# 시각화자료 출력
def id05_06_02():
    pass

###################### 이부분 굳이 필요할까
# 전처리 이후 데이터 경로 반환
def id05_07_01(
    project_id:str,
):
    pass

# 합성 이후 데이터 경로 반환
def id05_07_02(
    project_id:str,
):
    pass

# 후처리 이후 데이터 경로 반환
def id05_07_03(
    project_id:str,
):
    pass