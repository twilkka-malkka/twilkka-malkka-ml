# Script 상세 실행 가이드

## 실행 전 주의사항
- **ROOT 실행**: 모든 명령어는 프로젝트 루트(Root) 디렉토리에서 실행하는 것을 권장합니다.
- **Fail-Fast**: 이전 단계가 성공적으로 완료되어야 다음 단계 실행이 가능합니다.
- **Environment**: `Python 3.12` 환경을 권장하며, `3.10` 이상이어야 합니다.
## Step 0. 환경 구성 및 데이터 준비
### 1) 필수 패키지 설치

```bash
pip install -r requirements.txt
```
### 2) 데이터 체크리스트
 `00_data/01_raw/` 경로에 다음 파일들이 존재하는지 확인하십시오.
- `netflix_users.csv`: 사용자 기본 정보 (Raw data)
- `netflix_watch_history.csv`: 영상 시청 기록

## Step 1. Raw user 데이터 검증
### 목적
raw csv 파일에 학습 데이터 생성에 필요한 필수 컬럼들이 포함되어 있는지 검증합니다.
### 명령어
```bash
python 03_scripts/01_validate_raw.py [filename]
```
### 인자
| 인자 | 설명                                        | 기본값 |
| :--- |:------------------------------------------|:----|
| filename | Raw user data CSV 파일명 (netflix_users.csv) | 없음  |
### Output
검증 통과 메시지 혹은 누락된 컬럼 리스트(🛑 Fail 리포트).

## Step 2. user data 정제
### 목적
Raw 데이터를 정제하고 모델 학습 데이터셋과 성능 평가 데이터셋으로 분할하여 저장합니다.  
이후 script는 모두 학습 데이터셋만을 사용하며, `04_train.py`에서 사용하는 train/test set 역시 학습 데이터셋을 분할하여 사용합니다.
성능 평가 데이터셋은 모델 학습 과정에 사용되지 않으며 모델의 최종 성능을 평가하고, streamlit 페이지에서 load하는 용도로 사용됩니다.  
`00_data/00_raw/user_data_to_analysis.csv`파일의 경우 이 스크립트에서 분할 비율을 0.1로 설정하여 얻은 평가 데이터셋에 정답 라벨을 제거한 파일입니다.

### 명령어
```bash
python 03_scripts/02_make_dataset.py [filename] [-s TEST_SIZE]
```
### 인자
| 인자 | 설명                          | 기본값 |
| :--- |:----------------------------|:----|
| filename | Raw user data CSV 파일명       | 없음  |
| -s, --test_size | 학습/평가 데이터 분할 비율 ($0 < s < 1$) | 0.2 |
### 출력
`00_data/01_interim/` 내 정제된 user data 파일`[filename]_train.csv`, `[filename]_test.csv` 생성

## Step 3. watch features 추가
### 목적
`netflix_watch_history.csv`로부터 추출된 `watch features`를 `user_id` 기준으로 병합하여 모델 학습 데이터셋을 완성합니다.
### 명령어
```bash
python 03_scripts/03_build_features.py -o [output] [input]
```
### 인자
| 인자           | 설명                 | 기본값         |
|:-------------|:-------------------|:------------|
| input        | [Step 2]에서 정제된 학습 데이터셋 파일명 | 없음          |
| -o, --output | 출력 csv 파일명         | `input`과 동일 |
### 출력
`00_data/02_processed/` 내 모델 학습 데이터셋 파일`[output].csv` 생성
## Step 4. 모델 학습 및 결과 저장
### 목적
모델 학습 데이터를 불러와 선택한 모델을 훈련하고, 훈련된 모델 아티펙트와 모델 성능 지표를 파일로 저장합니다. 
### 명령어
```bash
python 03_scripts/04_train.py -m [model_list] [filename]
```
### 인자
| 인자          | 설명                                           | 기본값  |
|:------------|:---------------------------------------------|:-----|
| filename    | 모델 학습 데이터 CSV 파일명                            | 없음   |
| -m, --model | 학습할 모델 종류(`lr`, `rf`, `xgb`) `all`은 모든 모델 학습 | `all` |
### 출력
- `05_artifacts/00_models` 내부에 `model_type` 폴더 생성 후 학습된 모델 및 설정값(threshold, save type) 저장  
xgboost의 경우 다른 모델과 달리 `.joblib`이 아닌 자체 `json`형식으로 저장
- `05_artifacts/01_preprocessors` 내부에 전처리기 저장(훈련 시 사용한 경우만)
- `05_artifacts/02_metrics` 내부에 `metrics(classification report)`, `pr_curve_data`, `feature_importances` 저장
### 저장 예시
```
05_artifacts/
└── 00_models/
    └── XGBClassifier/
        ├── model.json
        └── config.json
└── 02_metrics/       
    ├── XGBoost_feature_importance.csv
    ├── XGBoost_metrics.json
    └── XGBoost_pr_curve_data.csv
```
