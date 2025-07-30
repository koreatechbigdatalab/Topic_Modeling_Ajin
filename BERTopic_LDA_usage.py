#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTopic과 LDA 토픽 모델링 통합 실행 스크립트
사용법: python BERTopic_LDA_usage.py
"""

import sys
import os

# 기존 분석 모듈들 임포트
from BERTopic_Analysis import BERTopicAnalyzer
from LDA_Analysis import LDAAnalyzer

########################################################
# 토픽 수 설정
########################################################
MAX_TOPICS = 10

def main():
    """BERTopic과 LDA 분석을 모두 실행"""
    
    print("🔬 BERTopic & LDA 토픽 모델링 통합 분석")
    print("=" * 50)
    
    # 데이터 경로 설정
    data_path = 'Results/pre_dataframe.xlsx'
    
    # 1. BERTopic 분석 실행
    print("\n🚀 BERTopic 분석 시작...")
    try:
        bertopic_analyzer = BERTopicAnalyzer()
        bertopic_results = bertopic_analyzer.run_full_analysis(
            data_path, 
            text_column='cleaned_text',
            max_topics= MAX_TOPICS + 1
        )
        print("✅ BERTopic 분석 완료!")
    except Exception as e:
        print(f"❌ BERTopic 분석 실패: {e}")
        bertopic_results = None
    
    # 2. LDA 분석 실행
    print("\n🚀 LDA 분석 시작...")
    try:
        lda_analyzer = LDAAnalyzer(data_path, 'cleaned_text')
        lda_success = lda_analyzer.run_complete_analysis(num_topics= MAX_TOPICS)
        print("✅ LDA 분석 완료!")
    except Exception as e:
        print(f"❌ LDA 분석 실패: {e}")
        lda_success = False
    
    # 3. 결과 요약
    print("\n" + "=" * 50)
    print("📊 분석 결과 요약")
    print("=" * 50)
    
    if bertopic_results:
        print("✅ BERTopic: 성공")
    else:
        print("❌ BERTopic: 실패")
        
    if lda_success:
        print("✅ LDA: 성공")
    else:
        print("❌ LDA: 실패")
    
    print("\n📁 결과 파일들은 'Results' 폴더에서 확인하세요!")
    print("💡 BERTopic과 LDA 결과를 비교해보세요.")

if __name__ == "__main__":
    main()

