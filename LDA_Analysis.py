#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA 토픽 모델링 분석 스크립트
작성자: 베테랑 데이터 분석가
목적: AI 광고 데이터의 토픽 분석

사용법:
======
1. 대화형 모드 (권장):
   python LDA_Analysis.py

2. 명령행 모드:
   python LDA_Analysis.py --topics 5         # 5개 토픽으로 분석
   python LDA_Analysis.py --help             # 도움말 보기
   
3. 사용자 정의 데이터:
   python LDA_Analysis.py --data custom.xlsx --column text_col --topics 8

⚠️ 참고: 토픽 수는 항상 수동으로 지정해야 합니다 (2-50 권장)

주요 분석 항목:
==============
1. LDA 토픽 모델링 (수동 토픽 수 지정)
2. Coherence 점수 계산 (c_v, c_uci, c_npmi, u_mass)
3. Topic Diversity 측정
4. Similarity Matrix 생성
5. 토픽별 키워드, 가중치, 예제 문장 추출
6. 시각화 및 종합 보고서 생성

출력 파일 (토픽 수 포함):
========================
- Excel 분석 결과: Results/lda_analysis_results_{토픽수}topics_*.xlsx
- 텍스트 요약: Results/lda_analysis_summary_{토픽수}topics_*.txt
- 시각화 이미지들: Results/*_{토픽수}topics_*.png
- LDA 모델: Results/lda_model_{토픽수}topics_*
- 유사도 매트릭스: Results/LDA_topic_similarity_matrix_{토픽수}topics_*.npy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 토픽 모델링 관련
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import gensim.downloader as api

# 전처리 및 유틸리티
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import itertools
from datetime import datetime
import os
import argparse
import sys

# 시각화 설정
# 한국어 폰트 설정 (시스템에 따라 조정 필요)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        logger.warning("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")


plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)


class LDAAnalyzer:
    """
    LDA 토픽 모델링 분석을 수행하는 클래스
    """
    
    def __init__(self, data_path="Results/pre_dataframe.xlsx", text_column="cleaned_text"):
        """
        초기화
        
        Args:
            data_path (str): 데이터 파일 경로
            text_column (str): 분석할 텍스트 컬럼명
        """
        self.data_path = data_path
        self.text_column = text_column
        self.df = None
        self.texts = None
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.num_topics = None  # 토픽 수 저장
        self.coherence_scores = {}
        self.topic_diversity = None
        self.similarity_matrix = None
        self.results = {}
        
    def load_data(self):
        """
        데이터 로드 및 기본 전처리
        """
        print("📊 데이터 로딩 중...")
        try:
            self.df = pd.read_excel(self.data_path)
            print(f"✅ 데이터 로드 완료: {len(self.df)}개 문서")
            print(f"📋 컬럼: {list(self.df.columns)}")
            
            # cleaned_text 컬럼 확인
            if self.text_column not in self.df.columns:
                print(f"❌ '{self.text_column}' 컬럼을 찾을 수 없습니다.")
                print(f"🔍 사용 가능한 컬럼: {list(self.df.columns)}")
                return False
                
            # 텍스트 데이터 정리
            self.df = self.df.dropna(subset=[self.text_column])
            self.df = self.df[self.df[self.text_column].str.len() > 10]  # 너무 짧은 텍스트 제거
            
            print(f"✅ 전처리 후 문서 수: {len(self.df)}개")
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {str(e)}")
            return False
    
    def preprocess_texts(self):
        """
        텍스트 전처리 및 토큰화
        """
        print("🔧 텍스트 전처리 중...")
        
        texts = self.df[self.text_column].tolist()
        
        # 기본 전처리
        processed_texts = []
        for text in texts:
            if pd.isna(text):
                continue
                
            # 문자열로 변환
            text = str(text)
            
            # 공백 기준 토큰화 (이미 전처리된 텍스트라고 가정)
            tokens = text.split()
            
            # 너무 짧거나 긴 토큰 제거
            tokens = [token for token in tokens if 2 <= len(token) <= 15]
            
            if len(tokens) >= 3:  # 최소 3개 이상의 토큰
                processed_texts.append(tokens)
        
        self.texts = processed_texts
        print(f"✅ 전처리 완료: {len(self.texts)}개 문서")
        
        # Gensim 사전 및 코퍼스 생성
        self.dictionary = corpora.Dictionary(self.texts)
        
        # 너무 빈번하거나 희소한 단어 제거
        self.dictionary.filter_extremes(no_below=5, no_above=0.7)
        
        # 코퍼스 생성
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        
        print(f"📚 사전 크기: {len(self.dictionary)} 단어")
        print(f"📄 코퍼스 크기: {len(self.corpus)} 문서")
    
    def find_optimal_topics(self, min_topics=2, max_topics=15):
        """
        최적의 토픽 수 찾기 (Coherence 기반)
        
        Args:
            min_topics (int): 최소 토픽 수
            max_topics (int): 최대 토픽 수
        """
        print("🔍 최적 토픽 수 탐색 중...")
        
        coherence_scores = []
        topic_nums = range(min_topics, max_topics + 1)
        
        for num_topics in topic_nums:
            print(f"   토픽 수 {num_topics} 테스트 중...")
            
            # LDA 모델 훈련
            lda_model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=20,
                alpha='auto',
                per_word_topics=True,
                eval_every=None
            )
            
            # Coherence 계산
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=self.texts,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            
            coherence_score = coherence_model.get_coherence()
            coherence_scores.append(coherence_score)
            self.coherence_scores[num_topics] = coherence_score
            
            print(f"   토픽 수 {num_topics}: Coherence = {coherence_score:.4f}")
        
        # 최적 토픽 수 선택
        optimal_topics = topic_nums[np.argmax(coherence_scores)]
        print(f"✅ 최적 토픽 수: {optimal_topics} (Coherence: {max(coherence_scores):.4f})")
        
        # Coherence 점수 시각화
        plt.figure(figsize=(10, 6))
        plt.plot(topic_nums, coherence_scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Coherence Score', fontsize=12)
        plt.title('Coherence Score by Topics', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(topic_nums)
        
        # 최적점 표시
        max_idx = np.argmax(coherence_scores)
        plt.axvline(x=topic_nums[max_idx], color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=max(coherence_scores), color='red', linestyle='--', alpha=0.7)
        # plt.text(topic_nums[max_idx], max(coherence_scores), 
        #         f'  최적점: {optimal_topics}토픽\n  Coherence: {max(coherence_scores):.4f}',
        #         fontsize=10, ha='left', va='bottom',
        #         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'Results/LDA_coherence_optimization_{min_topics}to{max_topics}topics_{timestamp}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        return optimal_topics
    
    def train_lda_model(self, num_topics):
        """
        LDA 모델 훈련
        
        Args:
            num_topics (int): 토픽 수
        """
        if num_topics is None:
            raise ValueError("토픽 수가 지정되지 않았습니다.")
        
        self.num_topics = num_topics  # 토픽 수 저장
        
        print(f"🎯 LDA 모델 훈련 중... (토픽 수: {num_topics})")
        
        # LDA 모델 훈련
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=50,
            iterations=100,
            alpha='auto',
            eta='auto',
            per_word_topics=True,
            eval_every=None
        )
        
        print("✅ LDA 모델 훈련 완료")
        
        # 모델 저장 (토픽 수 포함)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"Results/lda_model_{num_topics}topics_{timestamp}"
        self.lda_model.save(model_path)
        print(f"💾 모델 저장: {model_path}")
    
    def calculate_coherence(self):
        """
        다양한 Coherence 메트릭 계산
        """
        print("📊 Coherence 점수 계산 중...")
        
        coherence_metrics = ['c_v', 'c_uci', 'c_npmi', 'u_mass']
        
        for metric in coherence_metrics:
            try:
                coherence_model = CoherenceModel(
                    model=self.lda_model,
                    texts=self.texts,
                    dictionary=self.dictionary,
                    coherence=metric
                )
                score = coherence_model.get_coherence()
                self.coherence_scores[metric] = score
                print(f"   {metric.upper()}: {score:.4f}")
            except Exception as e:
                print(f"   {metric.upper()}: 계산 실패 ({str(e)})")
        
        return self.coherence_scores
    
    def calculate_topic_diversity(self):
        """
        Topic Diversity 계산
        토픽 간 키워드 겹침을 측정하여 다양성 평가
        """
        print("🌈 Topic Diversity 계산 중...")
        
        # 각 토픽의 상위 키워드 추출
        num_words = 20
        topic_words = []
        
        for topic_id in range(self.lda_model.num_topics):
            words = [word for word, _ in self.lda_model.show_topic(topic_id, num_words)]
            topic_words.append(set(words))
        
        # 토픽 간 겹치는 단어 계산
        unique_words = set()
        total_words = 0
        
        for words in topic_words:
            unique_words.update(words)
            total_words += len(words)
        
        # Topic Diversity 계산 (겹치지 않는 단어의 비율)
        self.topic_diversity = len(unique_words) / total_words
        
        print(f"✅ Topic Diversity: {self.topic_diversity:.4f}")
        print(f"   전체 고유 단어: {len(unique_words)}")
        print(f"   전체 단어 (중복포함): {total_words}")
        
        # 토픽 간 단어 겹침 매트릭스 계산
        overlap_matrix = np.zeros((self.lda_model.num_topics, self.lda_model.num_topics))
        
        for i in range(self.lda_model.num_topics):
            for j in range(self.lda_model.num_topics):
                if i != j:
                    intersection = len(topic_words[i].intersection(topic_words[j]))
                    union = len(topic_words[i].union(topic_words[j]))
                    overlap_matrix[i][j] = intersection / union if union > 0 else 0
        
        # 겹침 매트릭스 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='Reds',
                   xticklabels=[f'Topic {i}' for i in range(self.lda_model.num_topics)],
                   yticklabels=[f'Topic {i}' for i in range(self.lda_model.num_topics)])
        plt.title('Overlap by Topics', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'Results/LDA_topic_overlap_matrix_{self.num_topics}topics_{timestamp}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        return self.topic_diversity
    
    def calculate_similarity_matrix(self):
        """
        토픽 간 유사도 매트릭스 계산
        """
        print("🔗 토픽 유사도 매트릭스 계산 중...")
        
        # 토픽 분포 추출
        topic_distributions = []
        
        for topic_id in range(self.lda_model.num_topics):
            # 각 토픽의 단어 분포를 벡터로 변환
            topic_words = dict(self.lda_model.show_topic(topic_id, len(self.dictionary)))
            
            # 전체 단어에 대한 확률 벡터 생성
            topic_vector = np.zeros(len(self.dictionary))
            for word_id, word in self.dictionary.items():
                if word in topic_words:
                    topic_vector[word_id] = topic_words[word]
            
            topic_distributions.append(topic_vector)
        
        # 코사인 유사도 계산
        topic_distributions = np.array(topic_distributions)
        self.similarity_matrix = cosine_similarity(topic_distributions)
        
        # 유사도 매트릭스 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.similarity_matrix,
                   annot=True,
                   fmt='.3f',
                   cmap='coolwarm',
                   center=0,
                   xticklabels=[f'Topic {i}' for i in range(self.lda_model.num_topics)],
                   yticklabels=[f'Topic {i}' for i in range(self.lda_model.num_topics)])
        plt.title('Topic Similarrity Matrix(Cosine Similarity)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'Results/LDA_topic_similarity_matrix_{self.num_topics}topics_{timestamp}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 유사도 매트릭스 저장
        np.save(f'Results/LDA_topic_similarity_matrix_{self.num_topics}topics_{timestamp}.npy', self.similarity_matrix)
        
        return self.similarity_matrix
    
    def extract_topic_information(self):
        """
        토픽별 상세 정보 추출
        - 키워드 10개와 가중치
        - 해석을 위한 예제 문장
        """
        print("📝 토픽별 상세 정보 추출 중...")
        
        topic_info = []
        
        for topic_id in range(self.lda_model.num_topics):
            print(f"   토픽 {topic_id} 분석 중...")
            
            # 토픽의 상위 키워드 10개와 가중치
            topic_words = self.lda_model.show_topic(topic_id, 10)
            keywords = [word for word, weight in topic_words]
            weights = [weight for word, weight in topic_words]
            
            # 해당 토픽에 가장 관련성이 높은 문서들 찾기
            doc_topic_probs = []
            for doc_idx, doc in enumerate(self.corpus):
                doc_topics = self.lda_model.get_document_topics(doc)
                topic_prob = 0
                for t_id, prob in doc_topics:
                    if t_id == topic_id:
                        topic_prob = prob
                        break
                doc_topic_probs.append((doc_idx, topic_prob))
            
            # 상위 확률 문서들 선택 (상위 5개)
            doc_topic_probs.sort(key=lambda x: x[1], reverse=True)
            top_docs = doc_topic_probs[:5]
            
            # 예제 문장 추출
            example_sentences = []
            for doc_idx, prob in top_docs:
                if doc_idx < len(self.df):
                    sentence = str(self.df.iloc[doc_idx][self.text_column])
                    example_sentences.append({
                        'sentence': sentence[:200] + '...' if len(sentence) > 200 else sentence,
                        'probability': prob
                    })
            
            # 토픽 해석 생성 (키워드 기반)
            interpretation = self.generate_topic_interpretation(keywords)
            
            topic_info.append({
                'topic_id': topic_id,
                'keywords': keywords,
                'weights': weights,
                'interpretation': interpretation,
                'example_sentences': example_sentences,
                'total_documents': len([prob for _, prob in doc_topic_probs if prob > 0.1])
            })
        
        self.results['topic_information'] = topic_info
        return topic_info
    
    def generate_topic_interpretation(self, keywords):
        """
        키워드를 바탕으로 토픽 해석 생성
        
        Args:
            keywords (list): 토픽의 주요 키워드들
            
        Returns:
            str: 토픽 해석
        """
        # 키워드 분석을 통한 간단한 해석 생성
        keyword_str = ', '.join(keywords[:5])
        
        # 광고 관련 키워드 패턴 분석
        ad_patterns = {
            '브랜드': ['브랜드', '기업', '회사', '마케팅'],
            '기술': ['AI', '인공지능', '기술', '디지털', '데이터'],
            '소비자': ['고객', '소비자', '사용자', '사람들'],
            '효과': ['효과', '성과', '결과', '성공'],
            '미디어': ['미디어', '광고', '콘텐츠', '채널'],
            '제품': ['제품', '서비스', '솔루션']
        }
        
        # 키워드 매칭
        matched_categories = []
        for category, patterns in ad_patterns.items():
            if any(pattern in keyword_str for pattern in patterns):
                matched_categories.append(category)
        
        if matched_categories:
            interpretation = f"이 토픽은 주로 {', '.join(matched_categories)} 관련 내용을 다루며, "
        else:
            interpretation = "이 토픽은 "
            
        interpretation += f"핵심 키워드는 {keyword_str} 등입니다."
        
        return interpretation
    
    def create_comprehensive_report(self):
        """
        종합 분석 보고서 생성
        """
        print("📋 종합 분석 보고서 생성 중...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 토픽별 상세 정보를 DataFrame으로 변환
        topic_details = []
        for topic in self.results['topic_information']:
            topic_details.append({
                '토픽ID': topic['topic_id'],
                '해석': topic['interpretation'],
                '키워드': ', '.join(topic['keywords']),
                '키워드_가중치': ', '.join([f"{w:.4f}" for w in topic['weights']]),
                '관련문서수': topic['total_documents'],
                '예제문장1': topic['example_sentences'][0]['sentence'] if topic['example_sentences'] else '',
                '예제문장1_확률': topic['example_sentences'][0]['probability'] if topic['example_sentences'] else 0,
                '예제문장2': topic['example_sentences'][1]['sentence'] if len(topic['example_sentences']) > 1 else '',
                '예제문장2_확률': topic['example_sentences'][1]['probability'] if len(topic['example_sentences']) > 1 else 0
            })
        
        topic_df = pd.DataFrame(topic_details)
        
        # 2. Coherence 점수 DataFrame
        coherence_df = pd.DataFrame([
            {'메트릭': k, '점수': v} for k, v in self.coherence_scores.items()
        ])
        
        # 3. 분석 요약 정보
        summary_info = {
            '분석일시': timestamp,
            '전체문서수': len(self.df),
            '토픽수': self.lda_model.num_topics,
            '사전크기': len(self.dictionary),
            'Topic_Diversity': self.topic_diversity,
            '최고_Coherence': max([v for k, v in self.coherence_scores.items() if isinstance(v, (int, float))])
        }
        
        summary_df = pd.DataFrame([summary_info])
        
        # 4. Excel 파일로 저장
        excel_path = f'Results/lda_analysis_results_{self.num_topics}topics_{timestamp}.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='분석요약', index=False)
            topic_df.to_excel(writer, sheet_name='토픽상세정보', index=False)
            coherence_df.to_excel(writer, sheet_name='Coherence점수', index=False)
            
            # 유사도 매트릭스도 저장
            if self.similarity_matrix is not None:
                similarity_df = pd.DataFrame(
                    self.similarity_matrix,
                    columns=[f'토픽{i}' for i in range(self.lda_model.num_topics)],
                    index=[f'토픽{i}' for i in range(self.lda_model.num_topics)]
                )
                similarity_df.to_excel(writer, sheet_name='유사도매트릭스')
        
        print(f"✅ 종합 보고서 저장: {excel_path}")
        
        # 5. 텍스트 요약 보고서
        report_text = f"""
LDA 토픽 모델링 분석 보고서
============================

분석 일시: {timestamp}
데이터 파일: {self.data_path}

1. 기본 정보
-----------
- 전체 문서 수: {len(self.df):,}개
- 토픽 수: {self.lda_model.num_topics}개
- 사전 크기: {len(self.dictionary):,}개 단어
- Topic Diversity: {self.topic_diversity:.4f}

2. Coherence 점수
----------------
"""
        
        for metric, score in self.coherence_scores.items():
            if isinstance(score, (int, float)):
                report_text += f"- {metric.upper()}: {score:.4f}\n"
        
        report_text += f"""
3. 토픽별 상세 정보
------------------
"""
        
        for i, topic in enumerate(self.results['topic_information']):
            report_text += f"""
토픽 {i}: {topic['interpretation']}
키워드: {', '.join(topic['keywords'][:5])}
관련 문서 수: {topic['total_documents']}개
예제 문장: "{topic['example_sentences'][0]['sentence'] if topic['example_sentences'] else 'N/A'}"
"""
        
        # 텍스트 보고서 저장
        report_path = f'Results/lda_analysis_summary_{self.num_topics}topics_{timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✅ 텍스트 요약 저장: {report_path}")
        
        return excel_path, report_path
    
    def visualize_topics(self):
        """
        토픽 시각화
        """
        print("📊 토픽 시각화 생성 중...")
        
        # 1. 토픽별 키워드 중요도 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for topic_id in range(min(6, self.lda_model.num_topics)):
            topic_words = self.lda_model.show_topic(topic_id, 10)
            words, weights = zip(*topic_words)
            
            axes[topic_id].barh(range(len(words)), weights)
            axes[topic_id].set_yticks(range(len(words)))
            axes[topic_id].set_yticklabels(words)
            axes[topic_id].set_title(f'Topic {topic_id}', fontweight='bold')
            axes[topic_id].invert_yaxis()
        
        # 빈 subplot 제거
        for i in range(self.lda_model.num_topics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Topic Keywords (Top10)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'Results/LDA_topic_keywords_visualization_{self.num_topics}topics_{timestamp}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 2. 토픽 분포 히스토그램
        doc_topics = []
        for doc in self.corpus:
            doc_topic_dist = self.lda_model.get_document_topics(doc)
            doc_topic_dict = {topic_id: 0 for topic_id in range(self.lda_model.num_topics)}
            for topic_id, prob in doc_topic_dist:
                doc_topic_dict[topic_id] = prob
            doc_topics.append(doc_topic_dict)
        
        topic_counts = {i: 0 for i in range(self.lda_model.num_topics)}
        for doc_topic in doc_topics:
            dominant_topic = max(doc_topic, key=doc_topic.get)
            if doc_topic[dominant_topic] > 0.3:  # 30% 이상일 때만 해당 토픽으로 분류
                topic_counts[dominant_topic] += 1
        
        plt.figure(figsize=(12, 6))
        topics = list(topic_counts.keys())
        counts = list(topic_counts.values())
        
        bars = plt.bar([f'Topic {i}' for i in topics], counts, color='skyblue', alpha=0.7)
        plt.title('Distribution of documents by topic', fontsize=14, fontweight='bold')
        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Number of documents', fontsize=12)
        plt.xticks(rotation=45)
        
        # 막대 위에 수치 표시
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'Results/LDA_topic_distribution_{self.num_topics}topics_{timestamp}.png', dpi=300, bbox_inches='tight')
        # plt.show()
    
    def run_complete_analysis(self, num_topics):
        """
        전체 LDA 분석 실행
        
        Args:
            num_topics (int): 토픽 수 (필수)
        """
        if num_topics is None:
            raise ValueError("토픽 수가 지정되지 않았습니다.")
            
        print("🚀 LDA 토픽 모델링 분석 시작")
        print("=" * 50)
        
        # 1. 데이터 로드
        if not self.load_data():
            return False
        
        # 2. 텍스트 전처리
        self.preprocess_texts()
        
        # 3. LDA 모델 훈련
        self.train_lda_model(num_topics)
        
        # 4. Coherence 계산
        self.calculate_coherence()
        
        # 5. Topic Diversity 계산
        self.calculate_topic_diversity()
        
        # 6. 유사도 매트릭스 계산
        self.calculate_similarity_matrix()
        
        # 7. 토픽별 상세 정보 추출
        self.extract_topic_information()
        
        # 8. 시각화
        self.visualize_topics()
        
        # 9. 종합 보고서 생성
        excel_path, report_path = self.create_comprehensive_report()
        
        print("\n" + "=" * 50)
        print("🎉 LDA 분석 완료!")
        print(f"📊 결과 파일: {excel_path}")
        print(f"📝 요약 보고서: {report_path}")
        print(f"🎯 토픽 수: {self.num_topics}")
        print(f"📈 Topic Diversity: {self.topic_diversity:.4f}")
        print(f"🔍 최고 Coherence: {max([v for k, v in self.coherence_scores.items() if isinstance(v, (int, float))]):.4f}")
        
        return True


def main():
    """
    메인 실행 함수
    """
    # 명령행 인수 파서 설정
    parser = argparse.ArgumentParser(
        description='LDA 토픽 모델링 분석 도구 (수동 토픽 수 지정)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python LDA_Analysis.py                    # 대화형 모드 (토픽 수 수동 지정)
  python LDA_Analysis.py --topics 5         # 5개 토픽으로 분석
  python LDA_Analysis.py --data custom.xlsx --topics 8  # 사용자 정의 데이터 파일
        """
    )
    
    parser.add_argument(
        '--topics', '-t',
        type=int,
        help='토픽 수 지정 (2-50 권장, 필수)',
        metavar='N'
    )
    
    parser.add_argument(
        '--auto', '-a',
        action='store_true',
        help='⚠️ 지원되지 않음 - 수동으로 토픽 수를 지정해주세요'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default="Results/pre_dataframe.xlsx",
        help='분석할 데이터 파일 경로 (기본값: Results/pre_dataframe.xlsx)',
        metavar='FILE'
    )
    
    parser.add_argument(
        '--column', '-c',
        type=str,
        default="cleaned_text",
        help='분석할 텍스트 컬럼명 (기본값: cleaned_text)',
        metavar='COLUMN'
    )
    
    args = parser.parse_args()
    
    print("🔬 LDA 토픽 모델링 분석 도구")
    print("작성자: 베테랑 데이터 분석가 (10년+ 경험)")
    print("=" * 60)
    
    # Results 디렉토리 확인 및 생성
    if not os.path.exists('Results'):
        os.makedirs('Results')
        print("📁 Results 디렉토리 생성")
    
    # 토픽 수 결정
    num_topics = None
    
    # 명령행 인수가 있는 경우
    if args.auto:
        print("⚠️ 자동 최적화 모드는 지원하지 않습니다. 수동으로 토픽 수를 지정해주세요.")
        while True:
            try:
                num_topics = int(input("🔢 토픽 수를 입력하세요 (2-20 권장): "))
                if 2 <= num_topics <= 50:
                    print(f"✅ 토픽 수를 {num_topics}개로 설정했습니다.")
                    break
                else:
                    print("⚠️ 토픽 수는 2-50 사이의 값을 권장합니다.")
            except ValueError:
                print("❌ 올바른 숫자를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n👋 분석을 취소합니다.")
                return
        
    elif args.topics:
        if 2 <= args.topics <= 50:
            print(f"✅ 토픽 수 {args.topics}개로 설정 (명령행 옵션)")
            num_topics = args.topics
        else:
            print("❌ 토픽 수는 2-50 사이의 값을 권장합니다.")
            while True:
                try:
                    num_topics = int(input("🔢 토픽 수를 다시 입력하세요 (2-20 권장): "))
                    if 2 <= num_topics <= 50:
                        print(f"✅ 토픽 수를 {num_topics}개로 설정했습니다.")
                        break
                    else:
                        print("⚠️ 토픽 수는 2-50 사이의 값을 권장합니다.")
                except ValueError:
                    print("❌ 올바른 숫자를 입력해주세요.")
                except KeyboardInterrupt:
                    print("\n👋 분석을 취소합니다.")
                    return
            
    # 대화형 모드 - 항상 수동 지정
    else:
        print("\n🎯 토픽 수를 지정해주세요:")
        while True:
            try:
                num_topics = int(input("🔢 토픽 수를 입력하세요 (2-20 권장): "))
                if 2 <= num_topics <= 50:
                    print(f"✅ 토픽 수를 {num_topics}개로 설정했습니다.")
                    break
                else:
                    print("⚠️ 토픽 수는 2-50 사이의 값을 권장합니다.")
            except ValueError:
                print("❌ 올바른 숫자를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n👋 분석을 취소합니다.")
                return
    
    # 데이터 파일 경로 출력
    if args.data != "Results/pre_dataframe.xlsx":
        print(f"📂 사용자 정의 데이터 파일: {args.data}")
    if args.column != "cleaned_text":
        print(f"📝 사용자 정의 텍스트 컬럼: {args.column}")
    
    # LDA 분석기 초기화 및 실행
    analyzer = LDAAnalyzer(data_path=args.data, text_column=args.column)
    
    try:
        print(f"\n🚀 {num_topics}개 토픽으로 LDA 분석을 시작합니다...")
            
        success = analyzer.run_complete_analysis(num_topics)
        
        if success:
            print("\n✅ 모든 분석이 성공적으로 완료되었습니다!")
            print(f"\n📋 생성된 파일들 ({num_topics}개 토픽):")
            print(f"- Excel 분석 결과: Results/lda_analysis_results_{num_topics}topics_*.xlsx")
            print(f"- 텍스트 요약: Results/lda_analysis_summary_{num_topics}topics_*.txt")
            print(f"- 시각화 이미지들: Results/*_{num_topics}topics_*.png")
            print(f"- LDA 모델: Results/lda_model_{num_topics}topics_*")
            print(f"- 유사도 매트릭스: Results/LDA_topic_similarity_matrix_{num_topics}topics_*.npy")
            
            print("\n💡 사용 팁:")
            print("- 다음 번에는 명령행 옵션을 사용해보세요:")
            print(f"  python LDA_Analysis.py --topics {num_topics}")
        else:
            print("\n❌ 분석 중 오류가 발생했습니다.")
            
    except Exception as e:
        print(f"\n💥 예상치 못한 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
