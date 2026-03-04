# 튈까말까 OTT 고객 이탈률 EDA 및 머신러닝

기업 데이터 기반 EDA → 피처 엔지니어링 → ML/DL 모델링 → 예측/평가

## 팀원

| GitHub | Profile | Name |
|--|---|------|
| leedhroxx | @leedhroxx| 김민하  |
| rusidian | @rusidian | 장한재  |
| rshyun24| @rshyun24 | 배재현  |
| yoonha315 | @yoonha315 | 전윤하  |
| soll07 | @soll07 | 정다솔  |
| Gloveman | @Gloveman | 이창우  |

---

## 개발 규칙
### 변수 네이밍
- 변수: `snake_case`
- 클래스: `PascalCase`
- 상수: `UPPER_CASE`
- DataFrame: 접미사 `_df`
  - 예: `user_df`, `sample_df`

---

## 폴더 구조

```text
project/
├─ data/                      # 데이터 단계별 관리(원본/중간/최종)
│  ├─ 00_raw/
│  ├─ 01_interim/
│  └─ 02_processed/
│
├─ notebooks/                 # 개인 실험/탐색용(재현 로직은 src로 이전 권장)
│  ├─ 00_ingestion/
│  ├─ 01_preprocessing/
│  ├─ 02_features/
│  ├─ 03_models/
│  ├─ 04_training/
│  └─ 99_sandbox/
│
├─ src/                       # 팀 공용 핵심 로직(테스트/재사용/재현 가능한 코드)
│  ├─ common/                 # 경로/seed/logger/config 로더 등 공통
│  ├─ data/                   # [대분류] 데이터
│  │  ├─ ingestion/           # 수집/적재
│  │  ├─ preprocessing/       # 전처리/분할/피쳐
│  │  └─ io/                  # read/write, 포맷 핸들링
│  ├─ model/                  # [대분류] 모델
│  │  ├─ architectures/       # 모델 정의
│  │  ├─ training/            # 학습/평가/튜닝
│  │  ├─ inference/           # 추론/후처리
│  │  └─ registry/            # save/load, 버전/아티팩트 관리
│  └─ front/                  # [대분류] 프론트(Streamlit 전용)
│     ├─ ui/                  # 공용 UI 컴포넌트(사이드바/폼 등)
│     ├─ views/               # 화면 단위 렌더 함수(페이지가 호출)
│     ├─ state/               # session_state 관리
│     └─ viz/                 # 시각화 래퍼
│     
│
├─ scripts/                   # CLI 실행 엔트리(배치/파이프라인 “정답 실행 경로”)
│  ├─ 01_fetch_data.py
│  ├─ 02_make_dataset.py
│  ├─ 03_build_features.py
│  ├─ 04_train.py
│  └─ 05_infer.py
│
├─ configs/                   # 설정 파일 모음(yaml 등)
│
├─ artifacts/                 # 모델/전처리기/지표 등 산출물 저장소(기본 git 제외 권장)
│  ├─ models/
│  ├─ preprocessors/
│  └─ metrics/
│
├─ runs/                      # 실험 로그/결과(MLflow/CSV/메트릭 스냅샷 등, 기본 git 제외 권장)
│
├─ tests/                     # 단위/통합 테스트(전처리/피처/스키마/유스케이스 등)
│
├── app.py                    # Streamlit 메인 진입점(streamlit run app.py)
├── pages/                    # Streamlit 페이지(얇게 유지: 입력/버튼/표시 + src.front 호출)
│   ├── 01_Data_Analysis.py
│   ├── 02_Price_Prediction.py
│   └── 03_Intro.py
│
├─ requirements.txt           # 파이썬 의존성 목록
├─ .env.example               # 환경변수 샘플(실제 .env는 커밋 금지)
├─ LICENSE                    # 라이선스
├─ .gitignore                 # git 제외 규칙
└─ README.md                  # 프로젝트 사용법/규칙/실행 가이드
```
## 작업 방식
### 1) 개인 작업(노트북)
- 개인 작업은 `notebooks/`에서만 진행합니다.
- 노트북 파일명은 **작성자 + 주제**로 통일합니다.  
  - 예: `notebooks/00_ingestion/hanjae_eda_baseline.ipynb`
- 노트북에는 **EDA/실험 과정, 모델링 시도, 결론/인사이트/검증 결과**를 기록합니다.
  - 다만, 반복 사용하거나 파이프라인에 포함될 로직(전처리/피처 생성/학습·평가/추론)은 함수·클래스로 정리해 `src/`로 옮깁니다.

### 2) 팀 공용 코드(`src/`)
- 전처리/피처/학습/추론 로직은 함수/클래스로 모듈화해서 `src/`에 구현합니다.
- 공용 로직은 PR 리뷰를 통해 품질을 유지합니다.

### 3) 실행(정답 경로 = `scripts/`)
- 파이프라인 실행은 `scripts/`만 사용합니다.
- 실행 순서 예시:
  ```bash
  python scripts/01_fetch_data.py --config configs/data.yaml
  python scripts/02_make_dataset.py --config configs/data.yaml
  python scripts/03_build_features.py --config configs/features.yaml
  python scripts/04_train.py --config configs/train.yaml
  python scripts/05_evaluate.py --config configs/train.yaml
  python scripts/06_infer.py --config configs/infer.yaml
  ```
  
### 4) 산출물 관리
- `data/00_raw`: 원본 데이터(수정 금지)
- `artifacts/`: 모델/스케일러/인코더/토크나이저/메트릭
- `reports/`: 최종 결과(리포트, 이미지, 표)
- `.env`, API Key, 개인정보 포함 데이터는 **절대 커밋 금지**
- `artifacts/`, `runs/`는 기본적으로 `.gitignore` 대상입니다.  
  - 용량이 큰 파일(모델 체크포인트/로그 등)로 레포가 비대해지는 것을 방지합니다.
- **최종 모델/전처리기(및 필요한 산출물)는 Git에 커밋하지 않고**, 외부 저장소(예: Drive/S3/Release/W&B/MLflow 등)에 업로드하여 공유합니다.
- Git에는 **최종 결과 요약과 재현 정보만** `reports/`에 정리하여 커밋합니다.  
  - 예: 최종 점수/지표, 사용한 config 이름(또는 스냅샷), run/experiment ID, 외부 저장소 링크(또는 파일 식별자)

---

## Git

### 커밋 메시지 작성 규칙
- 본 프로젝트는 Conventional Commits 형식을 따릅니다.
- 커밋 제목(head)은 **반드시 영어로** 작성합니다.
- 커밋 타입은 **소문자**(`feat`, `fix`, `chore`, `docs`, `refactor` 등)를 사용합니다.
- 제목은 **72자 이내**로 간결하게 작성합니다.
- 필요할 경우 커밋 본문(body)에 **한국어 설명**을 추가할 수 있습니다.
- 이슈가 있는 경우 커밋 메시지에 **이슈 번호**를 함께 명시합니다. (예: `#12`)
- 이슈 제목은 **한국어로 작성해도 무방합니다.**

### 커밋 메시지 형식

(type)(scope): 간단한 요약

예시:
```git

git commit -m "docs(readme): add development and git conventions

- README에 변수 네이밍/폴더 구조/커밋 규칙을 정리함

Refs: #1
```

### 커밋 type
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `chore`: 설정 변경
- `docs`: 문서 수정
- `refactor`: 코드 구조 개선(기능 변경 없음)

### 커밋 Scope

Conventional Commits의 `scope`는  
변경이 영향을 주는 **범위(모듈/기능/도메인/폴더)** 를 짧게 요약한 라벨입니다.

> 형식: `type(scope): subject`  
> 예: `feat(preprocess): add missing value imputer`

---

### Scope 작성 원칙
- **변경 범위를 한 단어(또는 짧은 단어 조합)** 로 표현합니다.
- **팀에서 자주 쓰는 범위는 고정된 단어**로 통일합니다.
- 애매하거나 범위가 너무 넓으면 **scope는 생략 가능**합니다.
  - 예: `docs: update README`

---

### Scope 목록

- `readme` : README 및 프로젝트 설명 문서
- `docs` : 기타 문서(가이드, 규칙, 회의록 등)
- `data` : 데이터 파일/데이터 적재/데이터 스키마 관련
- `preprocess` : 전처리(결측치/인코딩/스케일링 등)
- `eda` : EDA 분석 코드, 통계/패턴 분석
- `viz` : 시각화(플롯, 차트 함수, 스타일)
- `feature` : 피처 엔지니어링/파생변수 생성
- `model` : 모델링/학습/평가 코드
- `report` : 결과 리포트/결과 정리(노트북 결과 포함)
- `config` : 설정 파일, 경로, 환경 변수 등
- `deps` : 라이브러리/requirements 등 의존성

---

### 예시
- `docs(readme): add project rules`
- `feat(preprocess): add outlier handling`
- `fix(data): handle invalid date formats`
- `refactor(viz): simplify plotting utilities`
- `chore(deps): update pandas and numpy`
- `docs: reorganize contribution guide`  *(scope 생략 예시)*

---

### Scope 선택 팁
- “**어떤 파일을 고쳤나?**”가 아니라  
  “**무엇을 위한 변경인가? (어디 범위인가?)**”를 기준으로 고릅니다.
- 여러 범위를 동시에 크게 건드렸다면
  - 커밋을 나누거나,
  - scope를 생략하고 `type: ...`로만 작성하는 것도 괜찮습니다.

---

### Issue 작성 규칙

본 프로젝트에서는 개발을 시작하기 전에  
반드시 작업 내용을 Issue로 먼저 작성합니다.

Issue는 작업의 목적과 범위를 명확히 하기 위한 용도이며,  
커밋 및 PR은 해당 Issue를 기준으로 연결합니다.

Issue 작성자는 기본적으로 Author를 작성합니다.  
Assignee는 무조건 지정합니다.

생성한 Issue에 대한 작업이 완료되면 Close합니다.

#### 양식
```markdown
## 작업 내용
- 무엇을 할 것인지 간단히 설명

## 목적
- 이 작업이 왜 필요한지

## 완료 기준
- [ ] 어떤 상태가 되면 완료인지

Author: @github_name
```
---

### Pull Request
PR을 하시기 전에 무조건 `sync`를 진행한 이후에 요청을 보냅니다.   
또한 PR 중에서 파일이 겹치는 경우에는 PR을 요청하지 마세요.

---