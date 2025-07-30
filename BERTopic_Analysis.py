"""
BERTopic 토픽 모델링 분석 (TensorFlow 없이)
- Coherence 계산
- Topic Diversity 계산  
- Similarity Matrix 생성
- 토픽별 키워드 10개, 가중치, 예시문장 추출

🎯 토픽 수 제어 방법:
1. 자동 토픽 수 결정: max_topics=None (기본값, HDBSCAN 사용)
2. 최대 토픽 수 제한: max_topics=10 (HDBSCAN + reduce_topics)
3. 정확한 토픽 수 지정: run_analysis_with_fixed_topics(n_topics=8) (KMeans 사용)

📖 사용 예시:
# 방법 1: 자동 결정
analyzer.run_full_analysis(data_path, max_topics=None)

# 방법 2: 최대 10개로 제한
analyzer.run_full_analysis(data_path, max_topics=10)

# 방법 3: 정확히 8개 토픽 생성
analyzer.run_analysis_with_fixed_topics(data_path, n_topics=8)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import pickle
from typing import List, Dict, Tuple, Any
import matplotlib.font_manager as fm

# TensorFlow 환경변수 설정 (TensorFlow 비활성화)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 지연 임포트를 위한 함수들
def import_bertopic():
    """BERTopic 모듈을 안전하게 임포트"""
    try:
        from bertopic import BERTopic
        return BERTopic
    except Exception as e:
        print(f"BERTopic 임포트 실패: {e}")
        print("다음 명령어로 설치해보세요:")
        print("pip uninstall tensorflow")
        print("pip install bertopic sentence-transformers")
        raise

def import_sentence_transformer():
    """SentenceTransformer를 안전하게 임포트"""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception as e:
        print(f"SentenceTransformer 임포트 실패: {e}")
        print("PyTorch 기반 설치를 시도해보세요:")
        print("pip install sentence-transformers torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        raise

def import_sklearn_modules():
    """scikit-learn 모듈들을 임포트"""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    return CountVectorizer, cosine_similarity

def import_umap_hdbscan():
    """UMAP과 HDBSCAN을 임포트"""
    from umap import UMAP
    from hdbscan import HDBSCAN
    return UMAP, HDBSCAN

def load_stopwords(stopwords_path='stopwords.txt'):
    """stopwords.txt 파일에서 불용어 리스트를 로드"""
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f if line.strip()]
            print(f"[DEBUG] 불용어 개수: {len(stopwords)}")
            print(f"[DEBUG] 불용어 예시: {stopwords[:10]}")
            return stopwords
    except Exception as e:
        print(f"불용어 파일 로드 실패: {e}")
        return None

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'AppleGothic'
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        warnings.warn("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
plt.rcParams['axes.unicode_minus'] = False

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

class BERTopicAnalyzer:
    """BERTopic을 이용한 토픽 모델링 분석 클래스 (TensorFlow 없이)"""
    
    def __init__(self, language_model: str = "jhgan/ko-sroberta-multitask"):
        """
        BERTopic 분석기 초기화
        
        Args:
            language_model: 한국어 임베딩을 위한 사전 훈련된 모델
        """
        self.language_model = language_model
        self.embedding_model = None
        self.topic_model = None
        self.documents = None
        self.embeddings = None
        self.topics = None
        self.probabilities = None
        self.results = {}
        
    def load_data(self, file_path: str, text_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        전처리된 데이터 로드
        
        Args:
            file_path: 데이터 파일 경로 (pre_dataframe.xlsx)
            text_column: 텍스트가 포함된 컬럼명 (cleaned_text)
            
        Returns:
            로드된 DataFrame
        """
        try:
            print(f"데이터 로딩 중: {file_path}")
            
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            else:
                raise ValueError("지원되지 않는 파일 형식입니다. .xlsx 또는 .csv 파일을 사용해주세요.")
                
            print(f"데이터 로드 완료: {len(df)}개 문서")
            print(f"컬럼: {list(df.columns)}")
            
            # cleaned_text 컬럼이 존재하는지 확인
            if text_column not in df.columns:
                print(f"❌ '{text_column}' 컬럼을 찾을 수 없습니다.")
                print(f"사용 가능한 컬럼: {list(df.columns)}")
                raise ValueError(f"'{text_column}' 컬럼이 없습니다. pre_dataframe.xlsx 파일을 확인하세요.")
            
            # 빈 값 제거
            original_count = len(df)
            df = df.dropna(subset=[text_column])
            df = df[df[text_column].str.strip() != '']
            
            print(f"빈 값 제거 후: {len(df)}개 문서 (제거된 문서: {original_count - len(df)}개)")
            
            if len(df) == 0:
                raise ValueError("유효한 텍스트 데이터가 없습니다.")
            
            self.documents = df[text_column].tolist()
            print(f"✅ 분석 대상 문서 수: {len(self.documents)}")
            
            return df
            
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생: {e}")
            raise
    
    def setup_model(self, 
                   min_topic_size: int = 10,
                   n_neighbors: int = 15,
                   n_components: int = 5,
                   min_cluster_size: int = 10,
                   metric: str = 'euclidean',
                   cluster_selection_method: str = 'eom',
                   max_topics: int = None,
                   random_state: int = 42) -> None:
        """
        BERTopic 모델 설정
        
        Args:
            max_topics: 최대 토픽 수 제한 (None이면 자동 결정)
        """
        print("🔧 모델 설정 중...")
        
        # max_topics 저장 (나중에 reduce_topics에서 사용)
        self.max_topics = max_topics
        
        try:
            # 필요한 모듈들 동적 임포트
            BERTopic = import_bertopic()
            SentenceTransformer = import_sentence_transformer()
            CountVectorizer, cosine_similarity = import_sklearn_modules()
            UMAP, HDBSCAN = import_umap_hdbscan()
            
            # 임베딩 모델 설정
            print(f"임베딩 모델 로딩 중: {self.language_model}")
            self.embedding_model = SentenceTransformer(self.language_model, device='cpu')
            
            # UMAP 차원 축소 설정
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric=metric,
                random_state=random_state
            )
            
            # HDBSCAN 클러스터링 설정
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                cluster_selection_method=cluster_selection_method,
                prediction_data=True
            )
            
            # CountVectorizer 설정 (한국어 처리)
            stopwords = load_stopwords('stopwords.txt')
            if stopwords is None:
                print("[WARNING] stopwords.txt를 불러오지 못했습니다. 불용어가 적용되지 않습니다.")
            else:
                print(f"[DEBUG] CountVectorizer에 적용된 불용어 개수: {len(stopwords)}")
            vectorizer_model = CountVectorizer(
                ngram_range=(1, 2),
                stop_words=stopwords,
                min_df=2,
                max_df=0.95
            )
            
            # BERTopic 모델 생성
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                language='korean',
                calculate_probabilities=True,
                verbose=True
            )
            
            print("✅ 모델 설정 완료")
            if max_topics:
                print(f"📊 최대 토픽 수 제한: {max_topics}개")
            
        except Exception as e:
            print(f"❌ 모델 설정 중 오류 발생: {e}")
            print("\n해결 방법:")
            print("1. pip uninstall tensorflow tensorflow-gpu")
            print("2. pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            print("3. pip install bertopic sentence-transformers")
            raise
    
    def setup_model_with_kmeans(self, 
                               n_topics: int,
                               n_neighbors: int = 15,
                               n_components: int = 5,
                               random_state: int = 42) -> None:
        """
        KMeans를 사용하여 정확한 토픽 수를 지정하는 BERTopic 모델 설정
        
        Args:
            n_topics: 생성할 토픽 수 (정확히 이 개수만큼 생성됨)
            n_neighbors: UMAP n_neighbors 파라미터
            n_components: UMAP 차원 수
            random_state: 랜덤 시드
        """
        print(f"🔧 KMeans 기반 모델 설정 중... (토픽 수: {n_topics})")
        
        try:
            # 필요한 모듈들 동적 임포트
            BERTopic = import_bertopic()
            SentenceTransformer = import_sentence_transformer()
            CountVectorizer, cosine_similarity = import_sklearn_modules()
            UMAP, _ = import_umap_hdbscan()
            
            from sklearn.cluster import KMeans
            
            # 임베딩 모델 설정
            print(f"임베딩 모델 로딩 중: {self.language_model}")
            self.embedding_model = SentenceTransformer(self.language_model, device='cpu')
            
            # UMAP 차원 축소 설정
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric='cosine',
                random_state=random_state
            )
            
            # KMeans 클러스터링 설정 (정확한 토픽 수 지정)
            kmeans_model = KMeans(
                n_clusters=n_topics,
                random_state=random_state,
                n_init=10
            )
            
            # CountVectorizer 설정 (한국어 처리)
            stopwords = load_stopwords('stopwords.txt')
            if stopwords is None:
                print("[WARNING] stopwords.txt를 불러오지 못했습니다. 불용어가 적용되지 않습니다.")
            else:
                print(f"[DEBUG] CountVectorizer에 적용된 불용어 개수: {len(stopwords)}")
            vectorizer_model = CountVectorizer(
                ngram_range=(1, 2),
                stop_words=stopwords,
                min_df=2,
                max_df=0.95
            )
            
            # BERTopic 모델 생성 (KMeans 사용)
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=kmeans_model,  # KMeans를 클러스터링으로 사용
                vectorizer_model=vectorizer_model,
                language='korean',
                calculate_probabilities=False,  # KMeans는 확률 계산 안 함
                verbose=True
            )
            
            print(f"✅ KMeans 기반 모델 설정 완료 (정확히 {n_topics}개 토픽 생성)")
            
        except Exception as e:
            print(f"❌ KMeans 모델 설정 중 오류 발생: {e}")
            raise
    
    def fit_transform(self) -> Tuple[List[int], np.ndarray]:
        """
        BERTopic 모델 학습 및 토픽 할당
        """
        print("🚀 BERTopic 모델 학습 중...")
        
        if self.documents is None:
            raise ValueError("먼저 데이터를 로드해주세요.")
        
        if self.topic_model is None:
            self.setup_model()
        
        try:
            # 토픽 모델링 실행
            self.topics, self.probabilities = self.topic_model.fit_transform(self.documents)
            
            unique_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
            noise_docs = sum(1 for topic in self.topics if topic == -1)
            
            print(f"✅ 초기 토픽 모델링 완료!")
            print(f"   - 발견된 토픽 수: {unique_topics}")
            print(f"   - 노이즈 문서 수: {noise_docs}")
            
            # max_topics가 지정된 경우 토픽 수 줄이기
            if hasattr(self, 'max_topics') and self.max_topics and unique_topics > self.max_topics:
                print(f"🔄 토픽 수를 {self.max_topics - 1}개로 줄이는 중...")
                
                # 토픽 수 줄이기
                self.topic_model.reduce_topics(self.documents, nr_topics=self.max_topics)
                
                # 새로운 토픽 할당 얻기
                self.topics = self.topic_model.transform(self.documents)[0]
                
                final_unique_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
                final_noise_docs = sum(1 for topic in self.topics if topic == -1)
                
                print(f"✅ 토픽 수 조정 완료!")
                print(f"   - 최종 토픽 수: {final_unique_topics}")
                print(f"   - 노이즈 문서 수: {final_noise_docs}")
            
            return self.topics, self.probabilities
            
        except Exception as e:
            print(f"❌ 토픽 모델링 중 오류 발생: {e}")
            raise
    
    def calculate_coherence(self) -> float:
        """
        토픽 일관성(Coherence) 계산
        """
        try:
            print("📊 Coherence 계산 중...")
            
            try:
                from gensim.models import CoherenceModel
                from gensim.corpora import Dictionary
            except ImportError:
                print("⚠️ gensim이 설치되지 않아 Coherence 계산을 건너뛰겠습니다.")
                print("설치: pip install gensim")
                return None
            
            # 토픽별 키워드 추출
            topic_words = []
            topic_info = self.topic_model.get_topic_info()
            
            for topic_id in topic_info['Topic'].values:
                if topic_id != -1:  # 노이즈 토픽 제외
                    words = [word for word, _ in self.topic_model.get_topic(topic_id)]
                    topic_words.append(words)
            
            if len(topic_words) == 0:
                print("⚠️ 유효한 토픽이 없어서 Coherence를 계산할 수 없습니다.")
                return None
            
            # 문서를 토큰으로 분할
            texts = [doc.split() for doc in self.documents]
            
            # Gensim Dictionary 생성
            dictionary = Dictionary(texts)
            
            # Coherence 모델 생성
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            
            coherence_score = coherence_model.get_coherence()
            self.results['coherence'] = coherence_score
            
            print(f"✅ Coherence Score (C_V): {coherence_score:.4f}")
            return coherence_score
            
        except Exception as e:
            print(f"❌ Coherence 계산 중 오류 발생: {e}")
            return None
    
    def calculate_topic_diversity(self) -> float:
        """
        토픽 다양성(Topic Diversity) 계산
        """
        print("📊 Topic Diversity 계산 중...")
        
        try:
            topic_info = self.topic_model.get_topic_info()
            all_words = set()
            total_words = 0
            
            for topic_id in topic_info['Topic'].values:
                if topic_id != -1:  # 노이즈 토픽 제외
                    words = [word for word, _ in self.topic_model.get_topic(topic_id)]
                    all_words.update(words)
                    total_words += len(words)
            
            if total_words == 0:
                return 0.0
            
            diversity_score = len(all_words) / total_words
            self.results['topic_diversity'] = diversity_score
            
            print(f"✅ Topic Diversity Score: {diversity_score:.4f}")
            print(f"   - 총 고유 키워드 수: {len(all_words)}")
            print(f"   - 총 키워드 수: {total_words}")
            
            return diversity_score
            
        except Exception as e:
            print(f"❌ Topic Diversity 계산 중 오류 발생: {e}")
            return None
    
    def create_similarity_matrix(self) -> np.ndarray:
        """
        토픽 간 유사도 매트릭스 생성
        """
        print("📊 Similarity Matrix 생성 중...")
        
        try:
            # sklearn 모듈 임포트
            _, cosine_similarity = import_sklearn_modules()
            
            # 토픽 임베딩 추출
            topic_embeddings = []
            topic_info = self.topic_model.get_topic_info()
            
            for topic_id in topic_info['Topic'].values:
                if topic_id != -1:  # 노이즈 토픽 제외
                    topic_words = [word for word, _ in self.topic_model.get_topic(topic_id)]
                    # 토픽의 키워드들을 하나의 문서로 결합
                    topic_text = ' '.join(topic_words)
                    embedding = self.embedding_model.encode([topic_text])
                    topic_embeddings.append(embedding[0])
            
            if len(topic_embeddings) == 0:
                print("⚠️ 유효한 토픽이 없습니다.")
                return None
            
            # 유사도 매트릭스 계산
            topic_embeddings = np.array(topic_embeddings)
            similarity_matrix = cosine_similarity(topic_embeddings)
            
            self.results['similarity_matrix'] = similarity_matrix
            
            # 유사도 매트릭스 시각화
            self.plot_similarity_matrix(similarity_matrix)
            
            print(f"✅ Similarity Matrix 생성 완료: {similarity_matrix.shape}")
            return similarity_matrix
            
        except Exception as e:
            print(f"❌ Similarity Matrix 생성 중 오류 발생: {e}")
            return None
    
    def plot_similarity_matrix(self, similarity_matrix: np.ndarray) -> None:
        """
        유사도 매트릭스 시각화
        """
        try:
            plt.figure(figsize=(12, 10))
            
            # 토픽 라벨 생성
            topic_info = self.topic_model.get_topic_info()
            valid_topics = [t for t in topic_info['Topic'].values if t != -1]
            labels = [f'Topic {i}' for i in valid_topics]
            
            # 히트맵 생성
            sns.heatmap(
                similarity_matrix,
                annot=False,  # 점수 표시 안함
                cmap='viridis',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Cosine Similarity'}
            )
            
            plt.title('Similarrity Matrix', fontsize=16, pad=20)
            plt.xlabel('Topics', fontsize=12)
            plt.ylabel('Topics', fontsize=12)
            plt.tight_layout()
            
            # 결과 폴더에 저장
            os.makedirs('Results', exist_ok=True)
            plt.savefig(f'Results/BERTopic_similarity_matrix_max{self.max_topics - 1}.png', dpi=300, bbox_inches='tight')
            # plt.show()
            
            print(f"✅ 유사도 매트릭스 시각화 완료: Results/BERTopic_similarity_matrix_max{self.max_topics - 1}.png")
            
        except Exception as e:
            print(f"❌ 유사도 매트릭스 시각화 중 오류 발생: {e}")
    
    def extract_topic_keywords_and_examples(self, top_k: int = 10) -> Dict[int, Dict]:
        """
        각 토픽별 키워드, 가중치, 예시문장 추출 (실제 확률/유사도 값 계산)
        """
        print(f"📊 토픽별 키워드 {top_k}개, 가중치, 예시문장 추출 중...")
        
        try:
            topic_details = {}
            topic_info = self.topic_model.get_topic_info()
            
            for topic_id in topic_info['Topic'].values:
                if topic_id != -1:  # 노이즈 토픽 제외
                    # 키워드와 가중치 추출
                    topic_words = self.topic_model.get_topic(topic_id)
                    keywords = []
                    weights = []
                    
                    for word, weight in topic_words[:top_k]:
                        keywords.append(word)
                        weights.append(weight)
                    
                    # 해당 토픽의 문서 인덱스 찾기
                    topic_docs_indices = [i for i, topic in enumerate(self.topics) if topic == topic_id]
                    
                    # 예시 문장 추출 (실제 확률/유사도 값 계산)
                    if topic_docs_indices:
                        doc_probs = []
                        
                        if self.probabilities is not None:
                            # HDBSCAN의 경우 - 실제 확률 사용
                            print(f"   Topic {topic_id}: HDBSCAN 확률값 사용")
                            for i in topic_docs_indices:
                                prob = float(self.probabilities[i][topic_id]) if hasattr(self.probabilities[i], '__getitem__') else float(self.probabilities[i])
                                doc_probs.append((i, prob))
                        
                        elif hasattr(self, 'embedding_model') and self.embedding_model is not None:
                            # KMeans의 경우 - 토픽 중심과 문서 간 코사인 유사도 계산
                            print(f"   Topic {topic_id}: 임베딩 유사도 계산 중...")
                            topic_words_list = [word for word, _ in self.topic_model.get_topic(topic_id)]
                            topic_text = ' '.join(topic_words_list[:5])  # 상위 5개 키워드
                            topic_embedding = self.embedding_model.encode([topic_text])[0]
                            
                            for i in topic_docs_indices:
                                doc_embedding = self.embedding_model.encode([self.documents[i]])[0]
                                # 코사인 유사도 계산 (정규화된 0~1 범위)
                                similarity = np.dot(topic_embedding, doc_embedding) / (
                                    np.linalg.norm(topic_embedding) * np.linalg.norm(doc_embedding)
                                )
                                # -1~1 범위를 0~1로 정규화
                                normalized_similarity = (similarity + 1) / 2
                                doc_probs.append((i, float(normalized_similarity)))
                        
                        else:
                            # 임베딩 모델이 없는 경우 - TF-IDF 기반 유사도 계산
                            print(f"   Topic {topic_id}: TF-IDF 기반 유사도 계산 중...")
                            try:
                                from sklearn.feature_extraction.text import TfidfVectorizer
                                from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
                                
                                # 토픽 키워드로 대표 문서 생성
                                topic_words_list = [word for word, _ in self.topic_model.get_topic(topic_id)]
                                topic_repr = ' '.join(topic_words_list[:10])
                                
                                # 해당 토픽의 문서들과 토픽 대표 문서 결합
                                docs_for_comparison = [topic_repr] + [self.documents[i] for i in topic_docs_indices]
                                
                                # TF-IDF 벡터화
                                tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
                                tfidf_matrix = tfidf.fit_transform(docs_for_comparison)
                                
                                # 토픽 대표 문서(첫 번째)와 각 문서 간 유사도 계산
                                topic_vector = tfidf_matrix[0:1]  # 첫 번째 문서 (토픽 대표)
                                doc_vectors = tfidf_matrix[1:]     # 나머지 문서들
                                
                                similarities = sklearn_cosine_similarity(topic_vector, doc_vectors)[0]
                                
                                for idx, similarity in enumerate(similarities):
                                    doc_index = topic_docs_indices[idx]
                                    doc_probs.append((doc_index, float(similarity)))
                                    
                            except ImportError:
                                # sklearn이 없는 경우 문서 길이 기반 점수
                                print(f"   Topic {topic_id}: 문서 길이 기반 점수 사용")
                                topic_words_set = set([word for word, _ in self.topic_model.get_topic(topic_id)])
                                
                                for i in topic_docs_indices:
                                    doc_words = set(self.documents[i].split())
                                    # Jaccard 유사도 계산
                                    intersection = len(topic_words_set.intersection(doc_words))
                                    union = len(topic_words_set.union(doc_words))
                                    jaccard_similarity = intersection / union if union > 0 else 0.0
                                    doc_probs.append((i, float(jaccard_similarity)))
                        
                        # 확률/유사도 순으로 정렬
                        doc_probs.sort(key=lambda x: x[1], reverse=True)
                        
                        # 통계 정보 출력
                        if doc_probs:
                            prob_values = [prob for _, prob in doc_probs]
                            print(f"     문서 수: {len(doc_probs)}, 확률/유사도 범위: {min(prob_values):.4f} ~ {max(prob_values):.4f}")
                        
                        # 상위 3개 문서를 예시로 선택
                        example_docs = []
                        for i, prob in doc_probs[:3]:
                            example_docs.append({
                                'text': self.documents[i],
                                'probability': prob,
                                'probability_type': self._get_probability_type()
                            })
                    else:
                        example_docs = []
                    
                    topic_details[topic_id] = {
                        'keywords': keywords,
                        'weights': weights,
                        'example_documents': example_docs,
                        'document_count': len(topic_docs_indices)
                    }
            
            self.results['topic_details'] = topic_details
            
            print(f"✅ 토픽 정보 추출 완료: {len(topic_details)}개 토픽")
            return topic_details
            
        except Exception as e:
            print(f"❌ 토픽 정보 추출 중 오류 발생: {e}")
            return {}
    
    def _get_probability_type(self) -> str:
        """현재 확률 계산 방식을 반환"""
        if self.probabilities is not None:
            return "HDBSCAN_probability"
        elif hasattr(self, 'embedding_model') and self.embedding_model is not None:
            return "embedding_cosine_similarity"
        else:
            return "tfidf_similarity"
    
    def save_results(self, output_dir: str = 'Results', top_k: int = 10) -> None:
        """
        결과 저장 관리 함수 (엑셀, 시각화 등)
        """
        import pandas as pd
        import os
        from datetime import datetime
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = os.path.join(output_dir, f'bertopic_analysis_results_{timestamp}.xlsx')

        # 불용어 로드
        stopwords = set(load_stopwords('stopwords.txt'))

        # 토픽별 키워드와 가중치 저장
        topic_info = self.topic_model.get_topic_info()
        used_words = set()
        keywords_data = []

        for topic_id in topic_info['Topic'].values:
            if topic_id == -1:
                continue
            topic_words = self.topic_model.get_topic(topic_id)
            unique_words = []
            for word, weight in topic_words:
                # 완전일치 또는 불용어가 단어 내 포함되어 있으면 제거
                if word in stopwords:
                    continue
                if any(sw in word for sw in stopwords):
                    continue
                if word not in used_words:
                    unique_words.append((word, weight))
                    used_words.add(word)
                if len(unique_words) >= top_k:
                    break
            for rank, (word, weight) in enumerate(unique_words, 1):
                keywords_data.append({
                    'Topic': topic_id,
                    'Keyword Rank': rank,
                    'Keyword': word,
                    'Weight': weight
                })

        # DataFrame으로 변환 (엑셀 저장 등)
        keywords_df = pd.DataFrame(keywords_data)
        keywords_df.to_excel(f'{output_dir}/중복없는_토픽키워드.xlsx', index=False)

        # 기존 결과 저장 로직
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 토픽별 키워드/가중치 시트
            keywords_df.to_excel(writer, sheet_name='Keywords', index=False)
            # 기존 summary, examples 등 다른 시트도 필요시 추가
        print(f"✅ 토픽별 키워드/가중치 엑셀 저장: {excel_path}")

    def create_visualizations(self, output_dir: str = 'Results') -> None:
        """
        토픽 모델링 결과 시각화 생성
        """
        print("📊 시각화 생성 중...")
        
        # 전제 조건 확인 (완화된 버전)
        if self.topic_model is None:
            print("❌ 토픽 모델이 초기화되지 않았습니다. 먼저 fit_transform()을 실행하세요.")
            return
        
        # 토픽 할당이 없어도 모델 기반 시각화는 가능
        if self.topics is None:
            print("⚠️ 토픽 할당 정보가 없습니다. 모델 기반 시각화만 진행합니다.")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ 출력 디렉토리 생성/확인 완료: {output_dir}")
            
            # 토픽 정보 확인
            topic_info = self.topic_model.get_topic_info()
            valid_topics = len(topic_info) - 1 if -1 in topic_info['Topic'].values else len(topic_info)
            print(f"📊 시각화할 토픽 수: {valid_topics}")
            
            if valid_topics == 0:
                print("⚠️ 유효한 토픽이 없어서 시각화를 건너뜁니다.")
                return
            
            visualizations_created = []
            
            # 1. 토픽 분포 시각화 (기본)
            try:
                print("   1. 토픽 분포 시각화 생성 중...")
                fig1 = self.topic_model.visualize_barchart(top_n_topics=valid_topics)  # API 수정
                fig1.write_html(f'{output_dir}/BERTopic_topic_barchart.html')
                visualizations_created.append("토픽 분포 차트 (barchart)")
                print("   ✅ 토픽 분포 시각화 완료")
            except Exception as e:
                print(f"   ❌ 토픽 분포 시각화 실패: {e}")
            
            # 2. Topic Word Scores 시각화 (개선된 버전)
            try:
                print("   2. 토픽 키워드 점수 시각화 생성 중...")
                fig2 = self.topic_model.visualize_barchart(
                    top_n_topics=valid_topics,  # API 수정: top_nr_topics → top_n_topics
                    n_words=5,  # 각 토픽당 상위 5개 키워드
                    title="Topic Word Scores"
                )
                fig2.write_html(f'{output_dir}/BERTopic_topic_word_scores.html')
                visualizations_created.append("토픽 키워드 점수 차트")
                print("   ✅ 토픽 키워드 점수 시각화 완료")
            except Exception as e:
                print(f"   ❌ 토픽 키워드 점수 시각화 실패: {e}")
            
            # 3. 토픽 간 거리 시각화
            try:
                print("   3. 토픽 간 거리 시각화 생성 중...")
                fig3 = self.topic_model.visualize_topics()
                fig3.write_html(f'{output_dir}/BERTopic_topic_distance.html')
                visualizations_created.append("토픽 간 거리 시각화")
                print("   ✅ 토픽 간 거리 시각화 완료")
            except Exception as e:
                print(f"   ❌ 토픽 간 거리 시각화 실패: {e}")
                print(f"      원인: UMAP 차원축소가 필요할 수 있습니다.")
            
            # 4. 계층적 토픽 시각화
            try:
                print("   4. 계층적 토픽 시각화 생성 중...")
                fig4 = self.topic_model.visualize_hierarchy()
                fig4.write_html(f'{output_dir}/BERTopic_topic_hierarchy.html')
                visualizations_created.append("계층적 토픽 구조")
                print("   ✅ 계층적 토픽 시각화 완료")
            except Exception as e:
                print(f"   ❌ 계층적 토픽 시각화 실패: {e}")
                print(f"      원인: 토픽 수가 부족하거나 계층 구조 계산이 어려울 수 있습니다.")
            
            # 5. 토픽별 히트맵
            try:
                print("   5. 토픽 히트맵 시각화 생성 중...")
                fig5 = self.topic_model.visualize_heatmap()
                fig5.write_html(f'{output_dir}/BERTopic_topic_heatmap.html')
                visualizations_created.append("토픽 히트맵")
                print("   ✅ 토픽 히트맵 시각화 완료")
            except Exception as e:
                print(f"   ❌ 토픽 히트맵 시각화 실패: {e}")
                print(f"      원인: 토픽 간 유사도 계산이 어려울 수 있습니다.")
            
            # 결과 요약
            if visualizations_created:
                print(f"\n✅ 시각화 파일 생성 완료: {output_dir}/")
                print("📋 생성된 시각화:")
                for viz in visualizations_created:
                    print(f"   - {viz}")
            else:
                print("❌ 시각화 파일을 하나도 생성하지 못했습니다.")
            
        except Exception as e:
            print(f"❌ 시각화 생성 중 전체적인 오류 발생: {e}")
            print("🔧 해결 방법:")
            print("  1. plotly 라이브러리 업데이트: pip install --upgrade plotly")
            print("  2. BERTopic 모델이 정상적으로 학습되었는지 확인")
            print("  3. 충분한 토픽이 생성되었는지 확인")
            import traceback
            print(f"\n상세 오류 정보:\n{traceback.format_exc()}")
    
    def create_safe_visualizations(self, output_dir: str = 'Results') -> None:
        """
        안전한 최소 시각화 생성 (가장 기본적인 것들만)
        """
        print("📊 안전한 시각화 생성 중...")
        
        if self.topic_model is None:
            print("❌ 토픽 모델이 없습니다.")
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 가장 기본적인 토픽 분포 차트만 생성 (API 수정)
            print("   토픽 분포 차트 생성 중...")
            fig = self.topic_model.visualize_barchart(top_n_topics=5)  # top_nr_topics → top_n_topics
            fig.write_html(f'{output_dir}/safe_topic_barchart.html')
            print("   ✅ 기본 토픽 분포 차트 완료")
            
            # 토픽 정보를 텍스트로 저장
            topic_info = self.topic_model.get_topic_info()
            with open(f'{output_dir}/topic_summary.txt', 'w', encoding='utf-8') as f:
                f.write("토픽 요약 정보\n")
                f.write("=" * 30 + "\n\n")
                for _, row in topic_info.iterrows():
                    if row['Topic'] != -1:
                        f.write(f"토픽 {row['Topic']}: {row['Count']}개 문서\n")
                        # 상위 키워드 추출
                        keywords = [word for word, _ in self.topic_model.get_topic(row['Topic'])[:5]]
                        f.write(f"키워드: {', '.join(keywords)}\n\n")
            
            print("   ✅ 토픽 요약 텍스트 파일 완료")
            
        except Exception as e:
            print(f"❌ 안전한 시각화도 실패: {e}")
            import traceback
            print(f"상세 오류:\n{traceback.format_exc()}")
    
    def run_full_analysis(self, data_path: str, text_column: str = 'cleaned_text', max_topics: int = None) -> Dict[str, Any]:
        """
        전체 BERTopic 분석 파이프라인 실행 (pre_dataframe.xlsx의 cleaned_text 컬럼 사용)
        
        Args:
            data_path: 데이터 파일 경로
            text_column: 텍스트 컬럼명
            max_topics: 최대 토픽 수 제한 (None이면 자동 결정)
        """
        print("🚀 BERTopic 전체 분석 시작...")
        print("=" * 50)
        
        try:
            # 1. 데이터 로드
            df = self.load_data(data_path, text_column)
            
            # 2. 모델 설정 및 학습
            self.setup_model(max_topics=max_topics)
            self.fit_transform()
            
            # 3. 평가 지표 계산
            coherence_score = self.calculate_coherence()
            diversity_score = self.calculate_topic_diversity()
            similarity_matrix = self.create_similarity_matrix()
            
            # 4. 토픽 정보 추출
            topic_details = self.extract_topic_keywords_and_examples()
            
            # 5. 결과 저장
            self.save_results()
            
            # 6. 시각화 생성
            self.create_visualizations()
            
            print("\n" + "=" * 50)
            print("✅ BERTopic 분석 완료!")
            print("=" * 50)
            
            # 결과 요약 출력
            print(f"\n📊 분석 결과 요약:")
            print(f"  - 총 문서 수: {len(self.documents)}")
            print(f"  - 발견된 토픽 수: {len(set(self.topics)) - (1 if -1 in self.topics else 0)}")
            if max_topics:
                print(f"  - 최대 토픽 수 제한: {max_topics}")
            if coherence_score:
                print(f"  - Coherence Score: {coherence_score:.4f}")
            if diversity_score:
                print(f"  - Topic Diversity: {diversity_score:.4f}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            raise

    def run_analysis_with_fixed_topics(self, data_path: str, n_topics: int, text_column: str = 'cleaned_text') -> Dict[str, Any]:
        """
        정확한 토픽 수를 지정하여 BERTopic 분석 실행 (KMeans 사용)
        
        Args:
            data_path: 데이터 파일 경로
            n_topics: 생성할 토픽 수 (정확히 이 개수만큼 생성됨)
            text_column: 텍스트 컬럼명
        """
        print("🚀 고정 토픽 수 BERTopic 분석 시작...")
        print("=" * 50)
        print(f"📊 지정된 토픽 수: {n_topics}개")
        print("=" * 50)
        
        # 고정 토픽 수 저장 (파일명에 사용)
        self.fixed_topics = n_topics
        
        try:
            # 1. 데이터 로드
            df = self.load_data(data_path, text_column)
            
            # 2. KMeans 기반 모델 설정 및 학습
            self.setup_model_with_kmeans(n_topics=n_topics)
            self.fit_transform()
            
            # 3. 평가 지표 계산 (확률이 없으므로 일부 건너뛰기)
            coherence_score = self.calculate_coherence()
            diversity_score = self.calculate_topic_diversity()
            similarity_matrix = self.create_similarity_matrix()
            
            # 4. 토픽 정보 추출
            topic_details = self.extract_topic_keywords_and_examples()
            
            # 5. 결과 저장
            self.save_results()
            
            # 6. 시각화 생성
            self.create_visualizations()
            
            print("\n" + "=" * 50)
            print("✅ 고정 토픽 수 BERTopic 분석 완료!")
            print("=" * 50)
            
            # 결과 요약 출력
            print(f"\n📊 분석 결과 요약:")
            print(f"  - 총 문서 수: {len(self.documents)}")
            print(f"  - 생성된 토픽 수: {len(set(self.topics)) - (1 if -1 in self.topics else 0)}")
            print(f"  - 지정된 토픽 수: {n_topics}")
            if coherence_score:
                print(f"  - Coherence Score: {coherence_score:.4f}")
            if diversity_score:
                print(f"  - Topic Diversity: {diversity_score:.4f}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            raise

    def test_visualizations(self) -> None:
        """
        시각화 기능을 테스트하는 함수
        """
        print("🧪 시각화 기능 테스트 중...")
        
        # 기본 상태 확인
        print(f"   topic_model: {self.topic_model is not None}")
        print(f"   topics: {self.topics is not None}")
        print(f"   documents: {len(self.documents) if self.documents else 0}")
        
        if self.topic_model is not None:
            try:
                topic_info = self.topic_model.get_topic_info()
                print(f"   토픽 정보: {len(topic_info)}개 토픽")
                print(f"   토픽 ID들: {list(topic_info['Topic'].values)[:5]}...")  # 처음 5개만
            except Exception as e:
                print(f"   토픽 정보 확인 실패: {e}")
        
        if self.topics is not None:
            unique_topics = set(self.topics)
            print(f"   할당된 토픽들: {len(unique_topics)}개")
            print(f"   노이즈 문서: {sum(1 for t in self.topics if t == -1)}개")
        
        # 실제 시각화 테스트
        self.create_visualizations()

    def load_existing_model_and_test_viz(self, model_path: str = None) -> None:
        """
        기존에 학습된 모델을 로드하고 시각화만 테스트
        """
        print("📂 기존 모델 로드 및 시각화 테스트...")
        
        try:
            # 가장 최근 모델 찾기
            if model_path is None:
                import glob
                model_dirs = glob.glob("Results/bertopic_model_*")
                if not model_dirs:
                    print("❌ 기존 모델을 찾을 수 없습니다.")
                    return
                model_path = sorted(model_dirs)[-1]  # 가장 최근 모델
            
            print(f"🔄 모델 로드 중: {model_path}")
            
            # BERTopic 모듈 임포트
            BERTopic = import_bertopic()
            
            # 모델 로드
            self.topic_model = BERTopic.load(model_path)
            print("✅ 모델 로드 완료")
            
            # 토픽 정보 확인
            topic_info = self.topic_model.get_topic_info()
            print(f"📊 로드된 토픽 수: {len(topic_info)}개")
            
            # 기존 데이터가 있으면 토픽 할당 재생성 시도
            try:
                print("🔄 기존 데이터로 토픽 할당 재생성 시도...")
                data_path = "Results/pre_dataframe.xlsx"
                df = pd.read_excel(data_path)
                if 'cleaned_text' in df.columns:
                    self.documents = df['cleaned_text'].dropna().tolist()
                    print(f"   📊 문서 수: {len(self.documents)}")
                    
                    # 토픽 할당 재생성
                    self.topics = self.topic_model.transform(self.documents)[0]
                    print(f"   ✅ 토픽 할당 재생성 완료: {len(set(self.topics))}개 토픽")
                else:
                    print("   ⚠️ cleaned_text 컬럼을 찾을 수 없습니다.")
            except Exception as e:
                print(f"   ⚠️ 토픽 할당 재생성 실패: {e}")
                print("   💡 모델만으로 시각화를 진행합니다.")
            
            # 안전한 시각화 테스트
            print("\n🧪 안전한 시각화 테스트 실행...")
            self.create_safe_visualizations()
            
            # 전체 시각화 테스트
            print("\n🧪 전체 시각화 테스트 실행...")
            self.create_visualizations()
            
        except Exception as e:
            print(f"❌ 모델 로드 및 시각화 테스트 실패: {e}")
            import traceback
            print(f"상세 오류:\n{traceback.format_exc()}")


# === 분석 전 Coherence & Diversity 계산 및 최적 토픽수 추천 ===
def calculate_coherence_and_diversity(texts, min_topics=2, max_topics=15):
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel, LdaModel
    import numpy as np

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    coherence_scores = []
    diversity_scores = []

    for num_topics in range(min_topics, max_topics+1):
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
        coherence_model = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        # Topic diversity: unique word 비율
        topic_words = [set([w for w, _ in lda.show_topic(i, topn=10)]) for i in range(num_topics)]
        all_words = set().union(*topic_words)
        diversity = len(all_words) / (num_topics * 10)
        coherence_scores.append(coherence)
        diversity_scores.append(diversity)

    return coherence_scores, diversity_scores

def find_optimal_topics(coherence_scores, min_topics=2):
    import numpy as np
    return int(np.argmax(coherence_scores)) + min_topics

def find_optimal_topics_bertopic(data_path, text_column='cleaned_text', min_topics=2, max_topics=8):
    """
    BERTopic 기반 최적 토픽 수(coherence 기준) 추천
    (주의: 느릴 수 있음)
    """
    from time import time
    coherence_scores = []
    for n_topics in range(min_topics, max_topics+1):
        print(f"\n[BERTopic 최적 토픽 탐색] n_topics={n_topics} 분석 중...")
        analyzer = BERTopicAnalyzer()
        t0 = time()
        analyzer.run_analysis_with_fixed_topics(data_path, n_topics=n_topics, text_column=text_column)
        score = analyzer.calculate_coherence()
        elapsed = time() - t0
        print(f"n_topics={n_topics}, coherence={score:.4f} (소요시간: {elapsed:.1f}초)")
        coherence_scores.append(score)
    optimal_topics = coherence_scores.index(max(coherence_scores)) + min_topics
    print(f"\n[BERTopic 기반] 최적 토픽 수: {optimal_topics} (coherence={max(coherence_scores):.4f})")
    return optimal_topics, coherence_scores

# === main 함수 또는 분석 진입점에 아래 코드 추가 ===
if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    print("==== [Step 1] pre_dataframe.xlsx 로드 ====")
    if not os.path.exists("Results/pre_dataframe.xlsx"):
        print("❌ Results/pre_dataframe.xlsx 파일이 없습니다. preprocessing.py를 먼저 실행하세요.")
        exit(1)
    df = pd.read_excel("Results/pre_dataframe.xlsx")
    if 'cleaned_text' not in df.columns:
        print("❌ 'cleaned_text' 컬럼이 없습니다. preprocessing.py를 먼저 실행하세요.")
        exit(1)
    texts = df['cleaned_text'].dropna().tolist()
    if len(texts) == 0:
        print("❌ 'cleaned_text'에 데이터가 없습니다. preprocessing.py를 확인하세요.")
        exit(1)
    print(f"✅ cleaned_text 샘플: {texts[:3]}")
    print(f"✅ 전체 문서 수: {len(texts)}")

    # LDA 기반 Coherence & Topic Diversity 곡선 시각화
    print("\n==== [Step 2] LDA 기반 Coherence & Topic Diversity 곡선 시각화 ====")
    min_topics, max_topics = 3, 8
    try:
        coherence_scores, diversity_scores = calculate_coherence_and_diversity([t.split() for t in texts], min_topics, max_topics)
        optimal_topics = 7  # 또는 원하는 값
        plt.figure(figsize=(10,5))
        plt.plot(range(min_topics, max_topics+1), coherence_scores, marker='o', label='Coherence')
        plt.plot(range(min_topics, max_topics+1), diversity_scores, marker='s', label='Diversity')
        plt.axvline(optimal_topics, color='red', linestyle='--', label=f'Optimal: {optimal_topics}')
        plt.xlabel('Number of Topics')
        plt.ylabel('Score')
        plt.title('Coherence & Topic Diversity')
        plt.legend()
        plt.tight_layout()
        os.makedirs('Results', exist_ok=True)
        plt.savefig('Results/lda_coherence_diversity_curve.png', dpi=200)
        plt.close()
        print("✅ Results/lda_coherence_diversity_curve.png 저장 완료!")
    except Exception as e:
        print(f"❌ LDA Coherence & Diversity 곡선 생성 중 오류: {e}")

    print("\n==== [Step 3] BERTopic 분석 ====")
    analyzer = BERTopicAnalyzer(language_model="paraphrase-multilingual-MiniLM-L12-v2")
    analyzer.documents = texts
    analyzer.run_analysis_with_fixed_topics("Results/pre_dataframe.xlsx", n_topics=7)
