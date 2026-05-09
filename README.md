# Titanic — Machine Learning from Disaster

Kaggle 타이타닉 생존 예측 대회

> 대회 링크: https://www.kaggle.com/competitions/titanic

---

## 목표

승객 정보(성별, 나이, 객실 등급 등)를 바탕으로 생존 여부(0=사망, 1=생존)를 예측한다.

---

## 파일 구조

```
├── titanic_analysis.ipynb   # 전체 분석 노트북 (메인)
├── train.csv                # 학습 데이터 (정답 포함, 891명)
├── test.csv                 # 테스트 데이터 (정답 없음, 418명)
├── gender_submission.csv    # Kaggle 제출 양식 샘플
└── submission.csv           # 최종 예측 결과 (Kaggle 제출용)
```

---

## 노트북 구성

### 섹션 1 — 데이터 탐색 및 전처리 (EDA)

- 결측값 현황 분석 및 처리
  - `Age`: Pclass·Sex 그룹별 중앙값으로 대체
  - `Embarked`: 최빈값(S)으로 대체
  - `Fare`: 중앙값으로 대체
  - `Cabin`: 결측 비율 77% → `HasCabin`(있고/없고) 변수로 변환 후 원본 삭제
- 주요 변수별 생존율 시각화 (성별, 객실 등급, 나이, 요금, 항구, 가족 구성)
- 상관계수 행렬 분석
- 피처 엔지니어링

| 새 변수 | 생성 방법 | 의미 |
|---------|-----------|------|
| `Title` | 이름에서 직위 추출 | Mr / Mrs / Miss / Master / Rare |
| `FamilySize` | SibSp + Parch + 1 | 본인 포함 총 가족 탑승 인원 |
| `IsAlone` | FamilySize == 1 | 혼자 탑승 여부 |
| `HasCabin` | Cabin 결측 여부 | 객실 정보 보유 여부 |
| `AgeBin` | 나이를 5구간으로 분할 | 0-12 / 13-18 / 19-35 / 36-60 / 61+ |
| `FareBin` | 요금을 4분위로 분할 | 저렴 / 보통 / 비쌈 / 매우 비쌈 |

### 섹션 2 — 모델 생성 및 교차 검증

7개 모델을 5-겹 교차 검증(Stratified K-Fold)으로 비교.

| 모델 | 특징 |
|------|------|
| Logistic Regression | 기준 모델(baseline) |
| Decision Tree | 조건 분기 기반 |
| Random Forest | 다수의 나무 다수결 |
| Gradient Boosting | 순차적 오차 보정 |
| SVM | 최적 경계선 탐색 |
| KNN | K개 이웃 다수결 |
| XGBoost | Gradient Boosting 최적화 구현 |

### 섹션 3 — 하이퍼파라미터 최적화 및 최종 모델 선택

- Random Forest: `GridSearchCV` (216개 조합)
- XGBoost: `RandomizedSearchCV` (50개 조합)
- Gradient Boosting: `GridSearchCV` (36개 조합)
- Soft Voting 앙상블 구성 (RF + GB + XGB + LR)
- 교차 검증 평균 정확도 기준으로 최종 모델 자동 선택

### 섹션 4 — 결론 및 제언

**주요 발견사항:**

1. **성별**이 가장 강력한 생존 예측 변수 — 여성 ~74% vs 남성 ~19%
2. **객실 등급**이 두 번째로 중요 — 1등급 ~63% vs 3등급 ~24%
3. **어린이(0~10세)** 생존율이 가장 높음
4. **소가족(1~3명)** 이 혼자 탑승보다 생존율 높음
5. 피처 엔지니어링으로 생성한 `Title`, `FareBin`이 높은 예측력 보임

---

## 실행 방법

### 환경 준비

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### 노트북 실행

```bash
jupyter notebook titanic_analysis.ipynb
```

열리면 상단 메뉴에서 `Kernel → Restart & Run All` 클릭.  
전체 실행 완료 후 `submission.csv` 자동 생성.

---

## 성능 결과

| 지표 | 값 |
|------|----|
| 5-겹 교차 검증 정확도 | ~0.83 |
| 예측 대상 | 418명 (테스트 데이터 전체) |

---

## 사용 라이브러리

- `pandas`, `numpy` — 데이터 처리
- `matplotlib`, `seaborn` — 시각화
- `scikit-learn` — 전처리, 모델, 평가
- `xgboost` — 그래디언트 부스팅 최적화 구현
