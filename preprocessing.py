"""
최적화된 텍스트 분석 및 단어 빈도수 분석 코드 (외부 stopwords 파일 사용 + 결과 저장)
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import re

from sklearn.feature_extraction.text import CountVectorizer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 한국어 폰트 설정 (시스템에 따라 조정 필요)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        logger.warning("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

plt.rcParams['axes.unicode_minus'] = False

# 형태소 분석기 지연 로딩 (여러 옵션 제공)
def get_morphological_analyzer():
    """형태소 분석기 초기화 (지연 로딩) - 여러 대안 제공"""
    # 1. KiwiPy 시도 (Java 불필요)
    try:
        from kiwipiepy import Kiwi
        logger.info("KiwiPy 형태소 분석기를 사용합니다.")
        return Kiwi(), 'kiwi'
    except ImportError:
        pass
    
    # 2. PyKoSpacing + 간단한 규칙 기반 분석 시도
    try:
        import soynlp
        from soynlp.noun import LRNounExtractor
        logger.info("SoyNLP 명사 추출기를 사용합니다.")
        return LRNounExtractor(), 'soynlp'
    except ImportError:
        pass
    
    # 3. KoNLPy 시도 (Java 필요)
    try:
        from konlpy.tag import Okt
        logger.info("KoNLPy Okt 형태소 분석기를 사용합니다.")
        return Okt(), 'konlpy'
    except Exception as e:
        logger.warning(f"KoNLPy 초기화 실패: {e}")
    
    # 4. 모든 분석기 실패 시 None 반환
    logger.warning("모든 형태소 분석기 초기화에 실패했습니다. 간단한 토큰화를 사용합니다.")
    return None, 'simple'

class StopwordsManager:
    """불용어 관리 클래스"""
    
    def __init__(self, stopwords_file: str = './stopwords.txt'):
        self.stopwords_file = Path(stopwords_file)
        self.stopwords: Set[str] = set()
        self._load_stopwords()
    
    def _load_stopwords(self):
        """불용어 파일에서 불용어 로드"""
        try:
            if self.stopwords_file.exists():
                with open(self.stopwords_file, 'r', encoding='utf-8') as f:
                    # 각 줄을 읽어서 공백 제거 후 set에 추가
                    self.stopwords = {line.strip() for line in f if line.strip()}
                logger.info(f"불용어 {len(self.stopwords)}개를 로드했습니다: {self.stopwords_file}")
            else:
                # 기본 불용어 생성 및 파일 저장
                self._create_default_stopwords()
                logger.info(f"기본 불용어 파일을 생성했습니다: {self.stopwords_file}")
        except Exception as e:
            logger.error(f"불용어 파일 로드 중 오류: {e}")
            self._create_default_stopwords()
    
    def _create_default_stopwords(self):
        """기본 불용어 세트 생성 및 파일 저장"""
        default_stopwords = [
            # 지시어/대명사
            '이것', '그것', '저것', '이런', '그런', '저런', '이렇게', '그렇게', '저렇게',
            '여기', '거기', '저기', '이곳', '그곳', '저곳',
            
            # 어미/조사
            '입니다', '습니다', '있습니다', '없습니다', '했습니다', '됩니다', '합니다',
            '이다', '하다', '되다', '있다', '없다', '같다', '다른', '많다', '적다',
            
            # 접속사
            '그리고', '하지만', '그러나', '또한', '따라서', '그래서', '그런데', '그러면',
            '만약', '비록', '심지어', '특히', '예를 들어', '즉', '한편',
            
            # 조사/전치사
            '때문', '위해', '통해', '대해', '에서', '에게', '에게서', '으로', '로서',
            '부터', '까지', '마다', '보다', '처럼', '같이', '함께', '대신',
            
            # 의존명사/형식명사
            '것은', '것이', '것을', '것의', '것도', '것만', '것까지', '것부터',
            '때는', '때가', '때를', '곳은', '곳이', '곳을', '점은', '점이', '점을',
            
            # 수량/정도 표현
            '하나', '둘', '셋', '매우', '정말', '너무', '아주', '꽤', '상당히',
            '조금', '약간', '거의', '완전히', '전혀', '별로',
            
            # 시간 표현
            '오늘', '어제', '내일', '지금', '나중', '이전', '이후', '동안', '사이',
            '요즘', '최근', '과거', '미래', '현재',
            
            # 일반적인 단어
            '사람', '경우', '때문', '문제', '상황', '방법', '결과', '이유', '목적',
            '과정', '단계', '부분', '전체', '내용', '정보', '자료', '데이터',
            
            # 감정/평가 표현
            '좋다', '나쁘다', '괜찮다', '어렵다', '쉽다', '중요하다', '필요하다',
            '가능하다', '불가능하다', '확실하다', '애매하다',
            
            # 기타 일반어
            '우리', '저희', '제가', '당신', '여러분', '모든', '각각', '서로',
            '자신', '스스로', '직접', '간접', '반드시', '절대', '가끔', '종종',

            # 불용어 처리 
            '가끔','각각','서로','자신','스스로','직접','간접','반드시','절대','가끔','종종',

            #불용어 단어
            
        ]
        self.stopwords = set(default_stopwords)
        
        # 파일로 저장
        try:
            with open(self.stopwords_file, 'w', encoding='utf-8') as f:
                for word in sorted(self.stopwords):
                    f.write(f"{word}\n")
            logger.info(f"기본 불용어 {len(self.stopwords)}개를 파일에 저장했습니다.")
        except Exception as e:
            logger.error(f"불용어 파일 저장 중 오류: {e}")
    
    def add_stopword(self, word: str):
        """불용어 추가"""
        self.stopwords.add(word.strip())
        self._save_stopwords()
    
    def remove_stopword(self, word: str):
        """불용어 제거"""
        self.stopwords.discard(word.strip())
        self._save_stopwords()
    
    def _save_stopwords(self):
        """현재 불용어 세트를 파일에 저장"""
        try:
            with open(self.stopwords_file, 'w', encoding='utf-8') as f:
                for word in sorted(self.stopwords):
                    f.write(f"{word}\n")
        except Exception as e:
            logger.error(f"불용어 파일 저장 중 오류: {e}")
    
    def is_stopword(self, word: str) -> bool:
        """단어가 불용어인지 확인"""
        return word.strip() in self.stopwords
    
    def get_stopwords(self) -> Set[str]:
        """불용어 세트 반환"""
        return self.stopwords.copy()
    
    def get_stopwords_dataframe(self) -> pd.DataFrame:
        """불용어를 데이터프레임으로 반환"""
        stopwords_list = sorted(list(self.stopwords))
        return pd.DataFrame({
            '불용어': stopwords_list,
            '순번': range(1, len(stopwords_list) + 1)
        })

class TextAnalyzer:
    """텍스트 분석을 위한 클래스"""
    
    def __init__(self, results_dir: str = './Results', stopwords_file: str = './stopwords.txt'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.morphological_analyzer = None
        self.analyzer_type = None
        self.stopwords_manager = StopwordsManager(stopwords_file)
        
    def _get_analyzer(self):
        """형태소 분석기 getter (지연 로딩)"""
        if self.morphological_analyzer is None:
            self.morphological_analyzer, self.analyzer_type = get_morphological_analyzer()
        return self.morphological_analyzer, self.analyzer_type
    
    @staticmethod
    def clean_text(text: str) -> str:
        """텍스트 전처리"""
        if pd.isna(text) or not text:
            return ""
        
        # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', str(text))
        # 연속된 공백을 하나로 통일
        text = re.sub(r'\s+', ' ', text)
        text = normalize_ai(text) # 인공지능 통일
        return text.strip()
    
    def extract_meaningful_words(self, text: str, min_length: int = 2) -> List[str]:
        """의미있는 단어 추출 (형태소 분석기 종류에 따라 다른 방식 사용)"""
        if not text:
            return []
        
        analyzer, analyzer_type = self._get_analyzer()
        
        try:
            if analyzer_type == 'kiwi':
                # KiwiPy 사용
                result = analyzer.analyze(text)
                words = []
                for token in result[0][0]:
                    if token.tag in ['NNG', 'NNP', 'VA', 'VV'] and len(token.form) >= min_length:
                        if not self.stopwords_manager.is_stopword(token.form):
                            words.append(token.form)
                return words
                
            elif analyzer_type == 'soynlp':
                # SoyNLP 사용 (명사만 추출)
                nouns = analyzer.extract(text)
                words = []
                for noun in nouns.keys():
                    if len(noun) >= min_length and not self.stopwords_manager.is_stopword(noun):
                        words.append(noun)
                return words
                
            elif analyzer_type == 'konlpy':
                # KoNLPy 사용
                pos_tags = analyzer.pos(text)
                meaningful_pos = ['Noun', 'Adjective', 'Verb']
                words = []
                for word, pos in pos_tags:
                    if (pos in meaningful_pos and 
                        len(word) >= min_length and 
                        not self.stopwords_manager.is_stopword(word)):
                        words.append(word)
                return words
                
            else:
                # 간단한 토큰화 (형태소 분석기 없을 때)
                return self._simple_tokenize(text, min_length)
                
        except Exception as e:
            logger.warning(f"형태소 분석 중 오류 발생 ({analyzer_type}): {e}")
            return self._simple_tokenize(text, min_length)
    
    def _simple_tokenize(self, text: str, min_length: int = 2) -> List[str]:
        """간단한 토큰화 (형태소 분석기가 없을 때 사용)"""
        # 한글 단어만 추출 (최소 길이 이상)
        korean_words = re.findall(r'[가-힣]+', text)
        
        # 길이 조건과 불용어 필터링
        words = []
        for word in korean_words:
            if len(word) >= min_length and not self.stopwords_manager.is_stopword(word):
                words.append(word)
        
        return words

def normalize_ai(text):
    # 대소문자 구분 없이 'ai', 'AI', 'Ai', 'aI' → '인공지능'
    text = re.sub(r'\b(ai|AI)\b', '인공지능', text, flags=re.IGNORECASE)
    # 'AI'와 '인공지능'이 띄어쓰기/조사 등으로 분리되어 있을 때도 처리
    text = text.replace('AI', '인공지능').replace('ai', '인공지능')
    return text

class DataProcessor:
    """데이터 처리를 위한 클래스"""
    
    def __init__(self, filepath_config: Dict[str, str]):
        self.filepath_config = filepath_config
        self.df = None
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """데이터 로드 및 전처리"""
        logger.info(f"데이터 로드 중: {self.filepath_config['path']}")
        
        try:
            # 파일 존재 여부 확인
            if not Path(self.filepath_config['path']).exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.filepath_config['path']}")
            
            # Excel 파일 로드
            raw_df = pd.read_excel(self.filepath_config['path'])
            logger.info(f"원본 데이터 크기: {raw_df.shape}")
            
            # 데이터 소스에 따른 전처리
            if self.filepath_config['name'] == '빅카인즈':
                df = self._preprocess_bigkinds_data(raw_df)
            else:
                df = self._preprocess_crawling_data(raw_df)
            
            # 공통 전처리
            df = self._common_preprocessing(df)
            
            logger.info(f"전처리 완료된 데이터 크기: {df.shape}")
            self.df = df
            return df
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {e}")
            raise
    
    def _preprocess_bigkinds_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """빅카인즈 데이터 전처리"""
        df = df[['일자', '제목', '본문']].copy()
        df.rename(columns={'일자': 'date', '본문': 'contents', '제목': 'title'}, inplace=True)
        
        # 날짜 변환
        df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
        return df
    
    def _preprocess_crawling_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """크롤링 데이터 전처리"""
        df = df[['시작 날짜', '제목', '정제데이터']].copy()
        df.rename(columns={'시작 날짜': 'date', '정제데이터': 'contents', '제목': 'title'}, inplace=True)
        
        # 날짜 변환
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    
    def _common_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """공통 전처리 작업"""
        # 중복 제거 (contents 기준)
        initial_count = len(df)
        df = df.drop_duplicates(subset=['contents']).reset_index(drop=True)
        logger.info(f"중복 제거: {initial_count} -> {len(df)} 개")
        
        # 결측값 제거
        df = df.dropna(subset=['contents']).reset_index(drop=True)
        df = df.dropna(subset=['date']).reset_index(drop=True)
        
        # 제목과 내용 결합
        df['title_contents'] = (df['title'].fillna('') + ' ' + df['contents'].fillna('')).str.strip()
        
        # 날짜 관련 컬럼 추가
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        return df

class WordFrequencyAnalyzer:
    """단어 빈도수 분석 클래스"""
    
    def __init__(self, text_analyzer: TextAnalyzer):
        self.text_analyzer = text_analyzer
    
    def analyze_word_frequency(self, df: pd.DataFrame, 
                             text_column: str = 'title_contents',
                             top_n: int = 200) -> pd.DataFrame:
        """단어 빈도수 분석"""
        logger.info("텍스트 전처리 시작...")
        
        # 텍스트 전처리
        df['cleaned_text'] = df[text_column].apply(self.text_analyzer.clean_text)
        
        # 단어 추출
        logger.info("형태소 분석 및 단어 추출 시작...")
        all_words = []
        
        for text in tqdm(df['cleaned_text'], desc="단어 추출 진행"):
            words = self.text_analyzer.extract_meaningful_words(text)
            all_words.extend(words)
        
        # 빈도수 계산
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(top_n)
        
        # 결과를 DataFrame으로 변환
        word_df = pd.DataFrame(top_words, columns=['단어', '빈도수'])
        
        logger.info(f"분석 완료: 총 {len(all_words)}개 단어, {len(word_counts)}개 고유 단어")
        return word_df

class Visualizer:
    """시각화 클래스 (이미지 크기 문제 해결)"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
    
    def create_word_frequency_plot(self, word_df: pd.DataFrame, 
                                 top_n: int = 100, 
                                 figsize: Tuple[int, int] = (12, 8),
                                 save_path: Optional[str] = None):
        """단어 빈도수 시각화 (세로 막대 그래프)"""
        
        # 상위 N개 단어 선택
        top_words = word_df.head(top_n)
        
        # DPI 및 figsize 자동 조정
        max_pixels = 32000  # 안전한 최대 픽셀 수
        
        # 기본 DPI 계산 (너무 높지 않게)
        base_dpi = min(300, max_pixels // max(figsize))
        
        # Figure 생성 시 DPI 제한
        plt.figure(figsize=figsize, dpi=min(base_dpi, 150))
        
        # 세로 막대 그래프 생성 (bar 사용)
        bars = plt.bar(range(len(top_words)), top_words['빈도수'], 
                      color='steelblue', alpha=0.7)
        
        # X축 설정 (단어 표시)
        plt.xticks(range(len(top_words)), top_words['단어'], rotation=45, ha='right')
        
        # 축 라벨 및 제목
        plt.xlabel('단어', fontsize=12)
        plt.ylabel('빈도수', fontsize=12)
        plt.title(f'상위 {top_n}개 단어 빈도수', fontsize=14, fontweight='bold')
        
        # 값 표시 (막대 위에)
        for i, v in enumerate(top_words['빈도수']):
            plt.text(i, v + max(top_words['빈도수']) * 0.01, str(v), 
                    ha='center', va='bottom', fontsize=9)
        
        # 그리드 추가 (읽기 쉽도록)
        plt.grid(axis='y', alpha=0.3)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장 (DPI 제한하여 이미지 크기 문제 방지)
        if save_path:
            try:
                # 안전한 DPI로 저장
                safe_dpi = min(150, max_pixels // max(figsize))
                plt.savefig(save_path, dpi=safe_dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                logger.info(f"그래프 저장 완료: {save_path} (DPI: {safe_dpi})")
            except Exception as e:
                logger.warning(f"고해상도 저장 실패, 낮은 DPI로 재시도: {e}")
                # 더 낮은 DPI로 재시도
                plt.savefig(save_path, dpi=72, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                logger.info(f"그래프 저장 완료 (낮은 해상도): {save_path}")
        
        # plt.show()
    
    def create_vertical_bar_plot(self, word_df: pd.DataFrame, 
                               top_n: int = 20, 
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None):
        """세로 막대 그래프 버전"""
        
        # 상위 N개 단어 선택
        top_words = word_df.head(top_n)
        
        # DPI 및 figsize 자동 조정
        max_pixels = 32000  # 안전한 최대 픽셀 수
        base_dpi = min(300, max_pixels // max(figsize))
        
        # Figure 생성 시 DPI 제한
        plt.figure(figsize=figsize, dpi=min(base_dpi, 150))
        
        # 세로 막대 그래프 생성
        bars = plt.bar(range(len(top_words)), top_words['빈도수'], 
                      color='steelblue', alpha=0.7)
        
        # X축 설정 (단어 표시)
        plt.xticks(range(len(top_words)), top_words['단어'], rotation=45, ha='right')
        
        # 축 라벨 및 제목
        plt.xlabel('단어', fontsize=12)
        plt.ylabel('빈도수', fontsize=12)
        plt.title(f'상위 {top_n}개 단어 빈도수', fontsize=14, fontweight='bold')
        
        # 값 표시 (막대 위에)
        for i, v in enumerate(top_words['빈도수']):
            plt.text(i, v + max(top_words['빈도수']) * 0.01, str(v), 
                    ha='center', va='bottom', fontsize=9)
        
        # 그리드 추가 (읽기 쉽도록)
        plt.grid(axis='y', alpha=0.3)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장
        if save_path:
            try:
                safe_dpi = min(150, max_pixels // max(figsize))
                plt.savefig(save_path, dpi=safe_dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                logger.info(f"그래프 저장 완료: {save_path} (DPI: {safe_dpi})")
            except Exception as e:
                logger.warning(f"고해상도 저장 실패, 낮은 DPI로 재시도: {e}")
                plt.savefig(save_path, dpi=72, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                logger.info(f"그래프 저장 완료 (낮은 해상도): {save_path}")
        
        # plt.show()

class ResultSaver:
    """결과 저장 관리 클래스"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
    
    def save_comprehensive_results(self, 
                                 df: pd.DataFrame, 
                                 word_df: pd.DataFrame, 
                                 stopwords_manager: StopwordsManager,
                                 analyzer_type: str = 'unknown') -> Dict[str, str]:
        """종합 결과를 여러 형태로 저장"""
        
        saved_files = {}
        
        # 1. 기본 결과들을 개별 파일로 저장
        
        # 전처리된 데이터
        pre_df_path = self.results_dir / 'pre_dataframe.xlsx'
        df.to_excel(pre_df_path, index=False)
        saved_files['전처리_데이터'] = str(pre_df_path)
        logger.info(f"전처리된 데이터 저장: {pre_df_path}")
        
        # 단어 빈도수 분석 결과
        word_freq_path = self.results_dir / 'word_frequency_analysis.xlsx'
        word_df.to_excel(word_freq_path, index=False)
        saved_files['단어빈도수_분석'] = str(word_freq_path)
        logger.info(f"단어 빈도수 분석 결과 저장: {word_freq_path}")
        
        # 불용어 목록
        stopwords_df = stopwords_manager.get_stopwords_dataframe()
        stopwords_path = self.results_dir / 'stopwords_list.xlsx'
        stopwords_df.to_excel(stopwords_path, index=False)
        saved_files['불용어_목록'] = str(stopwords_path)
        logger.info(f"불용어 목록 저장: {stopwords_path}")
        
        # 2. 종합 결과를 하나의 Excel 파일로 저장 (여러 시트)
        comprehensive_path = self.results_dir / 'comprehensive_analysis_results.xlsx'
        
        try:
            with pd.ExcelWriter(comprehensive_path, engine='openpyxl') as writer:
                # 시트 1: 분석 요약
                summary_df = self._create_analysis_summary(df, word_df, stopwords_manager, analyzer_type)
                summary_df.to_excel(writer, sheet_name='분석요약', index=False)
                
                # 시트 2: 단어 빈도수 분석 결과
                word_df.to_excel(writer, sheet_name='단어빈도수', index=False)
                
                # 시트 3: 불용어 목록
                stopwords_df.to_excel(writer, sheet_name='불용어목록', index=False)
                
                # 시트 4: 전처리된 데이터 (처음 1000행만)
                df_sample = df.head(1000) if len(df) > 1000 else df
                df_sample.to_excel(writer, sheet_name='전처리데이터_샘플', index=False)
                
                # 시트 5: 통계 정보
                stats_df = self._create_statistics_summary(df, word_df)
                stats_df.to_excel(writer, sheet_name='통계정보', index=False)
            
            saved_files['종합_결과'] = str(comprehensive_path)
            logger.info(f"종합 분석 결과 저장: {comprehensive_path}")
            
        except Exception as e:
            logger.error(f"종합 결과 저장 중 오류: {e}")
        
        return saved_files
    
    def _create_analysis_summary(self, 
                               df: pd.DataFrame, 
                               word_df: pd.DataFrame,
                               stopwords_manager: StopwordsManager,
                               analyzer_type: str) -> pd.DataFrame:
        """분석 요약 정보 생성"""
        
        summary_data = [
            ['분석 일시', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['사용된 형태소 분석기', analyzer_type],
            ['전체 문서 수', len(df)],
            ['분석 기간 (시작)', df['date'].min().strftime('%Y-%m-%d') if not df['date'].isna().all() else 'N/A'],
            ['분석 기간 (종료)', df['date'].max().strftime('%Y-%m-%d') if not df['date'].isna().all() else 'N/A'],
            ['추출된 고유 단어 수', len(word_df)],
            ['사용된 불용어 수', len(stopwords_manager.get_stopwords())],
            ['가장 빈도 높은 단어', word_df.iloc[0]['단어'] if len(word_df) > 0 else 'N/A'],
            ['가장 빈도 높은 단어 횟수', word_df.iloc[0]['빈도수'] if len(word_df) > 0 else 'N/A'],
            ['평균 단어 빈도수', round(word_df['빈도수'].mean(), 2) if len(word_df) > 0 else 'N/A']
        ]
        
        return pd.DataFrame(summary_data, columns=['항목', '값'])
    
    def _create_statistics_summary(self, df: pd.DataFrame, word_df: pd.DataFrame) -> pd.DataFrame:
        """통계 정보 요약 생성"""
        
        stats_data = []
        
        # 기본 통계
        stats_data.extend([
            ['데이터 통계', ''],
            ['전체 문서 수', len(df)],
            ['중복 제거 후 문서 수', len(df)],
            ['평균 문서 길이 (문자)', round(df['title_contents'].str.len().mean(), 2) if 'title_contents' in df.columns else 'N/A'],
            ['', ''],
        ])
        
        # 단어 빈도수 통계
        if len(word_df) > 0:
            stats_data.extend([
                ['단어 빈도수 통계', ''],
                ['총 고유 단어 수', len(word_df)],
                ['빈도수 평균', round(word_df['빈도수'].mean(), 2)],
                ['빈도수 중앙값', word_df['빈도수'].median()],
                ['빈도수 표준편차', round(word_df['빈도수'].std(), 2)],
                ['최대 빈도수', word_df['빈도수'].max()],
                ['최소 빈도수', word_df['빈도수'].min()],
                ['', ''],
            ])
        
        # 날짜별 통계 (가능한 경우)
        if 'date' in df.columns and not df['date'].isna().all():
            date_stats = df.groupby(df['date'].dt.date).size()
            stats_data.extend([
                ['날짜별 통계', ''],
                ['일평균 문서 수', round(date_stats.mean(), 2)],
                ['최대 일간 문서 수', date_stats.max()],
                ['최소 일간 문서 수', date_stats.min()],
                ['분석 기간 (일)', (df['date'].max() - df['date'].min()).days],
            ])
        
        return pd.DataFrame(stats_data, columns=['항목', '값'])

def main():
    """메인 실행 함수"""
    #########################################################
    # 설정 - 크롤링 데이터 사용
    # ctrl + / 를 사용해서 주석 해제
    #########################################################

    # FILEPATH = {
    #     'path': './RAW_DATA/AI 광고 키워드.xlsx',
    #     'name': '빅카인즈'
    # }

    FILEPATH = {
        'path': './RAW_DATA/2025_6_28_22_56_49_4600_channel_download_ai광고.xlsx',
        'name': '크롤링링'
    }
    
    RESULTS_DIR = './Results'
    STOPWORDS_FILE = './stopwords.txt'  # 불용어 파일 경로
    
    try:
        # 인스턴스 생성
        data_processor = DataProcessor(FILEPATH)
        text_analyzer = TextAnalyzer(RESULTS_DIR, STOPWORDS_FILE)
        word_analyzer = WordFrequencyAnalyzer(text_analyzer)
        visualizer = Visualizer(Path(RESULTS_DIR))
        result_saver = ResultSaver(Path(RESULTS_DIR))
        
        # 1. 데이터 로드 및 전처리
        df = data_processor.load_and_preprocess_data()
        
        # 2. 단어 빈도수 분석
        word_df = word_analyzer.analyze_word_frequency(df, top_n=100)
        
        # 3. 시각화
        plot_path = Path(RESULTS_DIR) / 'word_frequency_plot.png'
        visualizer.create_word_frequency_plot(word_df, top_n=20, save_path=plot_path)
        
        # 4. 종합 결과 저장 (불용어 포함)
        saved_files = result_saver.save_comprehensive_results(
            df=df,
            word_df=word_df,
            stopwords_manager=text_analyzer.stopwords_manager,
            analyzer_type=text_analyzer.analyzer_type or 'simple'
        )
        
        # 5. 결과 출력
        print("\n" + "="*50)
        print("분석 완료!")
        print("="*50)
        
        # 상위 10개 단어 출력
        print("\n=== 상위 10개 단어 ===")
        print(word_df.head(10).to_string(index=False))
        
        # 불용어 관리 정보 출력
        print(f"\n=== 불용어 관리 정보 ===")
        print(f"불용어 파일: {STOPWORDS_FILE}")
        print(f"총 불용어 개수: {len(text_analyzer.stopwords_manager.get_stopwords())}")
        print("불용어 파일을 수정하여 분석 결과를 조정할 수 있습니다.")
        
        # 저장된 파일들 정보 출력
        print(f"\n=== 저장된 파일들 ===")
        for desc, path in saved_files.items():
            print(f"{desc}: {path}")
        
        # 분석 통계 출력
        print(f"\n=== 분석 통계 ===")
        print(f"전체 문서 수: {len(df)}")
        print(f"추출된 고유 단어 수: {len(word_df)}")
        print(f"사용된 형태소 분석기: {text_analyzer.analyzer_type or 'simple'}")
        if len(word_df) > 0:
            print(f"가장 빈도 높은 단어: {word_df.iloc[0]['단어']} ({word_df.iloc[0]['빈도수']}회)")
        
        logger.info("모든 분석 및 저장 완료!")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()