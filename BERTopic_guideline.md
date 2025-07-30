# BERTopic 토픽 모델링 분석 가이드

## 📋 개요
이 코드는 BERTopic 라이브러리를 사용하여 한국어 텍스트 데이터에 대한 포괄적인 토픽 모델링 분석을 수행합니다.

## 🎯 주요 기능
1. **Coherence Score** - 토픽 일관성 평가
2. **Topic Diversity** - 토픽 다양성 측정  
3. **Similarity Matrix** - 토픽 간 유사도 매트릭스 생성
4. **키워드 분석** - 토픽별 상위 10개 키워드와 가중치 추출
5. **예시문장** - 각 토픽의 대표적인 문서 추출

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install bertopic sentence-transformers pandas numpy matplotlib seaborn gensim plotly
```

### 2. 데이터 준비
- 전처리된 데이터가 `Results/pre_dataframe.xlsx` 경로에 있어야 합니다
- 텍스트 데이터가 포함된 컬럼명을 확인하세요 (기본값: 'cleaned_text')

### 3. 분석 실행
```python
# 기본 실행
python BERTopic_Analysis.py

# 또는 Python 스크립트 내에서
from BERTopic_Analysis import BERTopicAnalyzer

analyzer = BERTopicAnalyzer()
results = analyzer.run_full_analysis("Results/pre_dataframe.xlsx")
```

## 📊 결과 파일

### Excel 파일
- `bertopic_analysis_results_YYYYMMDD_HHMMSS.xlsx`
  - **Topic_Summary**: 토픽별 기본 정보
  - **Keywords**: 토픽별 키워드와 가중치
  - **Examples**: 토픽별 예시 문서
  - **Metrics**: Coherence, Topic Diversity 점수

### 시각화 파일
- `topic_similarity_matrix.png`: 토픽 간 유사도 히트맵
- `topic_barchart.html`: 토픽별 문서 수 바차트
- `topic_distance.html`: 토픽 간 거리 시각화
- `topic_hierarchy.html`: 계층적 토픽 구조
- `topic_heatmap.html`: 토픽-키워드 히트맵

### 기타 파일
- `similarity_matrix_YYYYMMDD_HHMMSS.npy`: 유사도 매트릭스 (NumPy 배열)
- `bertopic_model_YYYYMMDD_HHMMSS/`: 훈련된 BERTopic 모델
- `analysis_summary_YYYYMMDD_HHMMSS.txt`: 분석 결과 요약

## ⚙️ 고급 설정

### 모델 파라미터 조정
```python
analyzer = BERTopicAnalyzer(language_model="jhgan/ko-sroberta-multitask")

# 모델 설정 사용자 정의
analyzer.setup_model(
    min_topic_size=15,        # 최소 토픽 크기
    n_neighbors=20,           # UMAP n_neighbors
    n_components=5,           # UMAP 차원 수  
    min_cluster_size=15,      # HDBSCAN 최소 클러스터 크기
    random_state=42
)

analyzer.fit_transform()
```

### 개별 분석 함수 사용
```python
# 데이터 로드
df = analyzer.load_data("your_data.xlsx", "text_column")

# 모델 학습
analyzer.setup_model()
topics, probabilities = analyzer.fit_transform()

# 개별 지표 계산
coherence = analyzer.calculate_coherence()
diversity = analyzer.calculate_topic_diversity()
similarity_matrix = analyzer.create_similarity_matrix()
topic_details = analyzer.extract_topic_keywords_and_examples()
```

## 📋 데이터 요구사항

### 입력 데이터 형식
- **Excel (.xlsx)** 또는 **CSV (.csv)** 파일
- 텍스트 데이터가 포함된 컬럼 필요
- 전처리가 완료된 데이터 권장 (불용어 제거, 정규화 등)

### 권장 데이터 크기
- **최소**: 100개 이상의 문서
- **권장**: 1,000개 이상의 문서
- **최적**: 10,000개 이상의 문서

## 🎨 결과 해석

### Coherence Score
- **범위**: 0 ~ 1 (높을수록 좋음)
- **해석**: 토픽 내 키워드들의 의미적 일관성
- **권장값**: 0.4 이상

### Topic Diversity
- **범위**: 0 ~ 1 (높을수록 좋음)  
- **해석**: 토픽 간 키워드 중복도 (다양성)
- **권장값**: 0.7 이상

### Similarity Matrix
- **범위**: -1 ~ 1 (코사인 유사도)
- **해석**: 토픽 간 유사도
- **활용**: 유사한 토픽 병합 또는 계층 구조 파악

## 🔧 문제 해결

### 일반적인 오류와 해결방법

1. **메모리 부족 오류**
   ```python
   # 배치 크기 조정
   analyzer.setup_model(min_topic_size=20, min_cluster_size=20)
   ```

2. **텍스트 컬럼 오류**
   ```python
   # 컬럼명 확인 후 지정
   df = analyzer.load_data("data.xlsx", "actual_text_column_name")
   ```

3. **한국어 임베딩 모델 변경**
   ```python
   # 다른 한국어 모델 사용
   analyzer = BERTopicAnalyzer("BM-K/KoSimCSE-roberta-multitask")
   ```

### 성능 최적화

1. **GPU 사용** (CUDA 환경에서)
   ```python
   import torch
   device = "cuda" if torch.cuda.is_available() else "cpu"
   analyzer = BERTopicAnalyzer()
   analyzer.embedding_model.to(device)
   ```

2. **배치 처리**
   ```python
   # 대용량 데이터의 경우 배치 단위로 처리
   batch_size = 1000
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i+batch_size]
       # 배치별 처리
   ```

## 📞 지원

### 추가 기능 요청 또는 버그 리포트
- 코드 개선사항이나 오류 발견 시 알려주세요
- 새로운 평가 지표나 시각화 기능 추가 가능

### 권장 후속 분석
1. **동적 토픽 모델링**: 시간에 따른 토픽 변화 분석
2. **가이드된 토픽 모델링**: 사전 정의된 키워드로 토픽 가이드
3. **다국어 토픽 모델링**: 여러 언어 동시 분석
4. **온라인 토픽 모델링**: 실시간 데이터 스트림 처리

---

**마지막 업데이트**: 2025년 1월
**버전**: 1.0.0
**작성자**: AI 데이터 분석 전문가 