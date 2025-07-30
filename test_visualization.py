"""
간단한 BERTopic 시각화 테스트 스크립트
"""

from BERTopic_Analysis import BERTopicAnalyzer

def test_visualization_only():
    """기존 모델로 시각화만 테스트"""
    print("🧪 시각화 기능 테스트 스크립트")
    print("=" * 40)
    
    try:
        # 분석기 초기화
        analyzer = BERTopicAnalyzer()
        
        # 기존 모델 로드 및 시각화 테스트
        analyzer.load_existing_model_and_test_viz()
        
        print("\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        print(f"\n상세 오류:\n{traceback.format_exc()}")

if __name__ == "__main__":
    test_visualization_only() 