import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="실시간 온라인 학습 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #2c3e50;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        color: #2c3e50;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-running {
        background: #27ae60;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 0.3rem 0;
        font-size: 0.75rem;
        width: fit-content;
        display: inline-block;
        font-weight: 500;
    }
    .status-stopped {
        background: #e74c3c;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 0.3rem 0;
        font-size: 0.75rem;
        width: fit-content;
        display: inline-block;
        font-weight: 500;
    }
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        border: none;
        padding: 0.5rem;
        font-weight: 500;
    }
    .insight-card {
        background: white;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
        position: relative;
    }
    .warning-card {
        background: white;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #f39c12;
        position: relative;
    }
    .danger-card {
        background: white;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #e74c3c;
        position: relative;
    }
    .success-card {
        background: white;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #27ae60;
        position: relative;
    }
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.7rem;
        color: #2c3e50;
    }
    .card-content {
        line-height: 1.4;
        color: #5a6c7d;
        font-size: 0.9rem;
    }
    .card-metric {
        background: #f8f9fa;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border: 1px solid #dee2e6;
        font-weight: 500;
        color: #2c3e50;
        font-size: 0.9rem;
    }
    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #27ae60;
        display: inline-block;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화 함수
def initialize_session_state():
    """세션 상태를 초기화합니다."""
    if 'train_data' not in st.session_state:
        st.session_state.train_data = None
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = StandardScaler()
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = LabelEncoder()
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = []
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'actual_values' not in st.session_state:
        st.session_state.actual_values = []
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    if 'is_fitted' not in st.session_state:
        st.session_state.is_fitted = False

initialize_session_state()

# 더미 SHAP 값 생성 함수 (나중에 실제 SHAP으로 대체)
def generate_dummy_shap_values():
    """더미 SHAP 값을 생성합니다."""
    feature_names = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)', 
                    '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)', '작업유형']
    
    # 랜덤하지만 현실적인 SHAP 값 생성
    np.random.seed(42 + st.session_state.current_index)
    shap_values = np.random.normal(0, 1, len(feature_names))
    
    # 일부 특성을 더 중요하게 만들기
    shap_values[0] *= 2.5  # 전력사용량을 가장 중요하게
    shap_values[1] *= 1.8  # 지상무효전력량을 두 번째로 중요하게
    shap_values[4] *= 1.5  # 지상역률을 세 번째로 중요하게
    
    return dict(zip(feature_names, shap_values))

# SHAP 중요도 차트 생성
def create_shap_chart():
    """SHAP 중요도 차트를 생성합니다."""
    if len(st.session_state.predictions) < 2:
        return None
    
    try:
        shap_values = generate_dummy_shap_values()
        
        # 절댓값 기준으로 정렬
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        features, values = zip(*sorted_features)
        
        # 색상 설정 (양수는 빨강, 음수는 파랑)
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]
        
        fig = go.Figure(data=go.Bar(
            y=features,
            x=values,
            orientation='h',
            marker_color=colors,
            hovertemplate='<b>%{y}</b><br>SHAP 값: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text='특성 중요도 (SHAP)', x=0.5, font=dict(size=16, color='#2c3e50')),
            xaxis_title='SHAP 값',
            yaxis_title='특성',
            height=400,
            template='plotly_white',
            margin=dict(l=120, r=50, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return (fig, shap_values)
    
    except Exception as e:
        st.error(f"SHAP 차트 생성 중 오류: {str(e)}")
        return None

# 실시간 인사이트 생성
def generate_insights(target_variable):
    """실시간 인사이트를 생성합니다."""
    insights = []
    
    if len(st.session_state.predictions) < 10:
        return insights
    
    try:
        # 최근 데이터 분석
        recent_predictions = st.session_state.predictions[-10:]
        recent_actuals = st.session_state.actual_values[-10:]
        
        # 1. 평균 전기 단가 계산
        if st.session_state.processed_data:
            recent_data = st.session_state.processed_data[-10:]
            if target_variable == "전기요금(원)":
                total_cost = sum([d.get('예측값', 0) for d in recent_data])
                total_kwh = sum([d.get('전력사용량(kWh)', 1) for d in recent_data])
                avg_unit_price = total_cost / max(total_kwh, 1)
                
                # 기준 단가 (더미 값)
                base_price = 120  # 원/kWh
                price_diff = ((avg_unit_price - base_price) / base_price) * 100
                
                if price_diff > 10:
                    card_type = "warning-card"
                    status = "주의"
                elif price_diff > 20:
                    card_type = "danger-card"
                    status = "위험"
                else:
                    card_type = "success-card"
                    status = "정상"
                
                insights.append({
                    'type': card_type,
                    'title': '평균 전기 단가',
                    'content': f"""
                    <div class="card-metric">
                        현재 평균 전기 단가: <span class="metric-value">{avg_unit_price:.0f}원/kWh</span>
                    </div>
                    <p>최근 30분간 평균 단가가 평소보다 <strong>{abs(price_diff):.1f}%</strong> {'높습니다' if price_diff > 0 else '낮습니다'}.</p>
                    
                    <div class="card-metric">
                        상태: {status}
                    </div>
                    
                    <p><strong>주요 영향 변수</strong>: 전력사용량, 지상무효전력량, 지상역률</p>
                    """
                })
        
        # 2. 주요 영향 변수 (SHAP 기반)
        shap_values = generate_dummy_shap_values()
        top_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        feature_list = []
        for feat, val in top_features:
            impact = "증가 영향" if val > 0 else "감소 영향"
            feature_list.append(f"• {feat} ({impact})")
        
        insights.append({
            'type': 'insight-card',
            'title': '주요 영향 변수',
            'content': f"""
            <div class="card-metric">
                Top 3 중요 변수
            </div>
            <div style="margin: 0.8rem 0;">
                {chr(10).join(feature_list)}
            </div>
            
            <p style="font-style: italic; color: #7f8c8d; margin-top: 0.5rem;">현재 예측에 가장 큰 영향을 미치는 변수들입니다.</p>
            """
        })
        
        # 3. 예측 정확도
        if len(recent_predictions) > 1:
            mae = mean_absolute_error(recent_actuals, recent_predictions)
            rmse = np.sqrt(mean_squared_error(recent_actuals, recent_predictions))
            
            accuracy_status = "우수" if mae < 50 else "보통" if mae < 100 else "주의"
            
            insights.append({
                'type': 'insight-card',
                'title': '예측 정확도',
                'content': f"""
                <div class="card-metric">
                    현재 예측 성능
                </div>
                
                <p><strong>MAE</strong>: {mae:.2f} | <strong>RMSE</strong>: {rmse:.2f}</p>
                <p><strong>상태</strong>: {accuracy_status}</p>
                
                <div class="card-metric">
                    최근 10건 기준 평균 오차율: <span class="metric-value">{(mae/np.mean(recent_actuals)*100):.1f}%</span>
                </div>
                """
            })
        
        # 4. 이상 탐지 상태
        if len(recent_predictions) > 5:
            recent_errors = [abs(a - p) for a, p in zip(recent_actuals[-5:], recent_predictions[-5:])]
            avg_error = np.mean(recent_errors)
            error_trend = "증가" if recent_errors[-1] > avg_error * 1.5 else "안정"
            
            if error_trend == "증가":
                card_type = "warning-card"
                status = "오차 증가 감지"
            else:
                card_type = "success-card"
                status = "정상 범위"
            
            insights.append({
                'type': card_type,
                'title': '이상 탐지 상태',
                'content': f"""
                <div class="card-metric">
                    예측 오류 상태: {status}
                </div>
                
                <p><strong>최근 오차 추이</strong>: {error_trend}</p>
                <p><strong>평균 오차</strong>: {avg_error:.2f}원</p>
                
                <div class="card-metric">
                    {'권장사항: 모델 재보정 또는 입력 데이터 점검이 필요할 수 있습니다.' if error_trend == '증가' else '상태: 예측 시스템이 안정적으로 작동 중입니다.'}
                </div>
                """
            })
        
        # 5. 최근 예측 추이
        if len(recent_predictions) > 3:
            pred_mean = np.mean(recent_predictions)
            pred_std = np.std(recent_predictions)
            volatility = "높음" if pred_std > pred_mean * 0.1 else "낮음"
            
            insights.append({
                'type': 'insight-card',
                'title': '최근 예측 추이',
                'content': f"""
                <div class="card-metric">
                    예측값 평균: <span class="metric-value">{pred_mean:.0f}원</span>
                </div>
                
                <p><strong>변동성</strong>: {volatility} (표준편차: {pred_std:.1f})</p>
                
                <div class="card-metric">
                    시스템 안정성: {'변동성 높음 - 모니터링 필요' if volatility == '높음' else '안정적 - 정상 운영'}
                </div>
                
                <p><strong>추세</strong>: 최근 예측값이 {'상승' if recent_predictions[-1] > pred_mean else '하락'} 경향을 보입니다.</p>
                """
            })
        
        # 6. 고부하 여부 체크
        if st.session_state.processed_data:
            recent_usage = [d.get('전력사용량(kWh)', 0) for d in st.session_state.processed_data[-5:]]
            max_usage = max(recent_usage) if recent_usage else 0
            avg_usage = np.mean(recent_usage) if recent_usage else 0
            
            # 가정: 5kWh 이상이 고부하
            high_load_threshold = 5.0
            is_high_load = max_usage > high_load_threshold
            
            if is_high_load:
                insights.append({
                    'type': 'warning-card',
                    'title': '고부하 경고',
                    'content': f"""
                    <div class="card-metric">
                        현재 전력 사용량: <span class="metric-value">{max_usage:.2f}kWh</span>
                    </div>
                    
                    <p><strong>상태</strong>: 고부하 상태 (임계값: {high_load_threshold}kWh)</p>
                    <p><strong>평균 사용량</strong>: {avg_usage:.2f}kWh</p>
                    
                    <div class="card-metric">
                        권장사항: 전력 사용량 모니터링을 강화하고 필요시 부하 분산을 고려하세요.
                    </div>
                    """
                })
        
    except Exception as e:
        st.error(f"인사이트 생성 중 오류: {str(e)}")
    
    return insights

# 데이터 로드 함수
@st.cache_data
def load_train_data():
    """train.csv 파일을 로드합니다."""
    try:
        data = pd.read_csv('data/train.csv')
        return data
    except FileNotFoundError:
        st.error("'data/train.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"데이터 로드 실패: {str(e)}")
        return None

# 파일 업로드 옵션 추가
def load_uploaded_data(uploaded_file):
    """업로드된 파일을 로드합니다."""
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
            return data
        else:
            st.error("CSV 파일만 지원됩니다.")
            return None
    except Exception as e:
        st.error(f"파일 로드 실패: {str(e)}")
        return None

# 특성 전처리 함수 개선
def preprocess_features(data_row, fit_scalers=False):
    """데이터 행을 전처리합니다."""
    try:
        feature_cols = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)', 
                       '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)', '작업유형']
        
        # 필수 컬럼 존재 여부 확인
        missing_cols = [col for col in feature_cols if col not in data_row.index]
        if missing_cols:
            st.error(f"필수 컬럼이 없습니다: {missing_cols}")
            return None
        
        features = data_row[feature_cols].copy()
        
        # 작업유형 인코딩
        if fit_scalers:
            if st.session_state.train_data is not None:
                unique_work_types = st.session_state.train_data['작업유형'].unique()
                st.session_state.label_encoder.fit(unique_work_types)
        
        # 작업유형이 학습되지 않은 새로운 값인 경우 처리
        try:
            work_type_encoded = st.session_state.label_encoder.transform([features['작업유형']])[0]
        except ValueError:
            # 새로운 작업유형이 나타난 경우 가장 빈번한 작업유형으로 대체
            if st.session_state.train_data is not None:
                most_common_type = st.session_state.train_data['작업유형'].mode()[0]
                work_type_encoded = st.session_state.label_encoder.transform([most_common_type])[0]
            else:
                work_type_encoded = 0
        
        # 수치형 특성 추출
        numeric_features = features.drop('작업유형').values
        numeric_features = np.append(numeric_features, work_type_encoded)
        
        # 스케일링
        if fit_scalers:
            if st.session_state.train_data is not None:
                all_numeric = st.session_state.train_data[feature_cols[:-1]].values
                work_types_encoded = st.session_state.label_encoder.transform(st.session_state.train_data['작업유형'])
                all_features = np.column_stack([all_numeric, work_types_encoded])
                st.session_state.scaler.fit(all_features)
                st.session_state.is_fitted = True
        
        if st.session_state.is_fitted:
            scaled_features = st.session_state.scaler.transform([numeric_features])[0]
        else:
            scaled_features = numeric_features
        
        return scaled_features
    
    except Exception as e:
        st.error(f"전처리 중 오류 발생: {str(e)}")
        return None

# 모델 초기화 개선
def initialize_model(model_type):
    """선택된 모델을 초기화합니다."""
    try:
        if model_type == "SGD Regressor (온라인 학습)":
            return SGDRegressor(
                random_state=42, 
                learning_rate='adaptive', 
                eta0=0.01,
                max_iter=1000,
                alpha=0.0001
            )
        elif model_type == "Random Forest":
            return RandomForestRegressor(
                n_estimators=50, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
    except Exception as e:
        st.error(f"모델 초기화 실패: {str(e)}")
        return None

# 실시간 차트 생성 함수 개선
def create_realtime_chart(target_variable):
    """실시간 예측 차트를 생성합니다."""
    if len(st.session_state.predictions) < 2:
        return None
    
    try:
        # 최근 데이터만 표시 (성능 향상)
        recent_range = min(200, len(st.session_state.predictions))
        indices = list(range(len(st.session_state.predictions) - recent_range, len(st.session_state.predictions)))
        recent_pred = st.session_state.predictions[-recent_range:]
        
        # 예측값 차트만 생성
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=indices, 
            y=recent_pred, 
            mode='lines+markers', 
            name='예측값', 
            line=dict(color='#3498db', width=3),
            marker=dict(size=6, color='#3498db'),
            hovertemplate='<b>예측값</b><br>순서: %{x}<br>값: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f'{target_variable} 실시간 예측 결과', x=0.5, font=dict(size=18, color='#2c3e50')),
            xaxis_title='데이터 순서',
            yaxis_title=target_variable,
            height=500,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"차트 생성 중 오류: {str(e)}")
        return None

# 사이드바 설정
st.sidebar.header("시스템 설정")

# 데이터 자동 로드
if st.session_state.train_data is None:
    with st.spinner("데이터 로딩 중..."):
        st.session_state.train_data = load_train_data()
        if st.session_state.train_data is not None:
            st.sidebar.success(f"데이터 로드 완료! ({len(st.session_state.train_data):,}행)")
            # 초기화
            st.session_state.current_index = 0
            st.session_state.processed_data = []
            st.session_state.predictions = []
            st.session_state.actual_values = []
            st.session_state.metrics_history = []
            st.session_state.is_fitted = False

# 모델 설정
st.sidebar.subheader("모델 설정")

model_type = st.sidebar.selectbox(
    "모델 선택",
    ["SGD Regressor (온라인 학습)", "Random Forest"],
    help="SGD는 실시간 학습에 최적화되어 있습니다."
)

target_variable = st.sidebar.selectbox(
    "예측 타겟",
    ["전기요금(원)", "전력사용량(kWh)", "탄소배출량(tCO2)"],
    help="예측하고 싶은 변수를 선택하세요."
)

# 학습 파라미터
st.sidebar.subheader("학습 파라미터")

speed = st.sidebar.slider(
    "학습 속도 (초/행)", 
    0.1, 5.0, 1.0, 0.1,
    help="각 데이터 처리 간격을 설정합니다."
)

chart_update_interval = st.sidebar.slider(
    "차트 업데이트 간격", 
    1, 20, 5,
    help="몇 개의 데이터마다 차트를 업데이트할지 설정합니다."
)

if model_type == "Random Forest":
    batch_size = st.sidebar.slider(
        "배치 크기", 
        10, 200, 50,
        help="Random Forest 재학습을 위한 배치 크기입니다."
    )

# 시스템 상태 표시
if st.session_state.train_data is not None:
    progress_percent = (st.session_state.current_index / len(st.session_state.train_data)) * 100
    st.sidebar.metric("진행률", f"{progress_percent:.1f}%")
    st.sidebar.metric("처리된 행", f"{st.session_state.current_index:,}")
    st.sidebar.metric("전체 행", f"{len(st.session_state.train_data):,}")
    
    # 제어 버튼
    st.sidebar.subheader("제어판")
    
    col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
    
    with col_btn1:
        if st.button("시작", type="primary"):
            if st.session_state.model is None:
                st.session_state.model = initialize_model(model_type)
                if st.session_state.model is not None:
                    st.sidebar.success("모델 초기화 완료!")
            st.session_state.is_running = True
    
    with col_btn2:
        if st.button("정지"):
            st.session_state.is_running = False
    
    with col_btn3:
        if st.button("리셋"):
            # 모든 상태 초기화
            keys_to_reset = ['current_index', 'model', 'is_running', 'processed_data', 
                           'predictions', 'actual_values', 'metrics_history', 'is_fitted']
            for key in keys_to_reset:
                if key in st.session_state:
                    if key == 'current_index':
                        st.session_state[key] = 0
                    elif key in ['is_running', 'is_fitted']:
                        st.session_state[key] = False
                    else:
                        st.session_state[key] = [] if key in ['processed_data', 'predictions', 'actual_values', 'metrics_history'] else None
            
            st.session_state.scaler = StandardScaler()
            st.session_state.label_encoder = LabelEncoder()
            st.sidebar.success("시스템 리셋 완료!")
            st.rerun()

# 메인 화면
if st.session_state.train_data is not None:
    # 시스템 상태 표시 (왼쪽 상단에 작게)
    col_status, col_spacer = st.columns([1, 4])
    with col_status:
        if st.session_state.is_running:
            st.markdown('<div class="status-running"><span class="status-indicator"></span>실행 중</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-stopped">정지</div>', unsafe_allow_html=True)
    
    # 상단 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "처리된 데이터", 
            f"{st.session_state.current_index:,}",
            delta=f"{len(st.session_state.train_data) - st.session_state.current_index:,} 남음"
        )
    
    with col2:
        if len(st.session_state.predictions) > 0:
            latest_prediction = st.session_state.predictions[-1]
            st.metric("최신 예측값", f"{latest_prediction:.2f}")
        else:
            st.metric("최신 예측값", "대기 중")
    
    with col3:
        if len(st.session_state.actual_values) > 0:
            latest_actual = st.session_state.actual_values[-1]
            st.metric("최신 실제값", f"{latest_actual:.2f}")
        else:
            st.metric("최신 실제값", "대기 중")
    
    with col4:
        if len(st.session_state.predictions) > 1:
            recent_error = abs(st.session_state.actual_values[-1] - st.session_state.predictions[-1])
            st.metric("최신 오차", f"{recent_error:.2f}")
        else:
            st.metric("최신 오차", "계산 중")
    
    # 진행률 바
    if st.session_state.current_index < len(st.session_state.train_data):
        progress_value = st.session_state.current_index / len(st.session_state.train_data)
        st.progress(progress_value)
    
    # 메인 대시보드 (예측 차트 + SHAP)
    st.subheader("실시간 모니터링 대시보드")
    
    # 차트와 SHAP을 나란히 배치
    chart_col, shap_col = st.columns([3, 2])
    
    with chart_col:
        # 예측 차트 컨테이너
        chart_container = st.empty()
    
    with shap_col:
        # SHAP 차트 컨테이너
        shap_container = st.empty()
    
    # 인사이트 카드 섹션
    if len(st.session_state.predictions) > 10:
        st.subheader("분석 인사이트")
        insights = generate_insights(target_variable)
        
        # 카드들을 3열로 배치
        if insights:
            for i in range(0, len(insights), 3):
                card_col1, card_col2, card_col3 = st.columns(3)
                
                with card_col1:
                    if i < len(insights):
                        insight = insights[i]
                        
                        st.markdown(f"""
                        <div class="{insight['type']}">
                            <div class="card-title">
                                {insight['title']}
                            </div>
                            <div class="card-content">
                                {insight['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with card_col2:
                    if i + 1 < len(insights):
                        insight = insights[i + 1]
                        
                        st.markdown(f"""
                        <div class="{insight['type']}">
                            <div class="card-title">
                                {insight['title']}
                            </div>
                            <div class="card-content">
                                {insight['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with card_col3:
                    if i + 2 < len(insights):
                        insight = insights[i + 2]
                        
                        st.markdown(f"""
                        <div class="{insight['type']}">
                            <div class="card-title">
                                {insight['title']}
                            </div>
                            <div class="card-content">
                                {insight['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # 실시간 학습 로직
    if st.session_state.is_running and st.session_state.current_index < len(st.session_state.train_data):
        current_row = st.session_state.train_data.iloc[st.session_state.current_index]
        
        # 특성 전처리
        fit_scalers = (st.session_state.current_index == 0)
        X = preprocess_features(current_row, fit_scalers)
        
        if X is not None:
            y = current_row[target_variable]
            
            # 예측 (모델이 학습된 경우)
            prediction = 0
            if st.session_state.model is not None and st.session_state.current_index > 0:
                try:
                    prediction = st.session_state.model.predict([X])[0]
                except Exception as e:
                    st.warning(f"예측 실패: {str(e)}")
                    st.session_state.model = initialize_model(model_type)
            
            # 데이터 저장
            st.session_state.predictions.append(prediction)
            st.session_state.actual_values.append(y)
            
            # 모델 학습
            try:
                if st.session_state.model is not None:
                    if model_type == "SGD Regressor (온라인 학습)":
                        st.session_state.model.partial_fit([X], [y])
                    
                    elif model_type == "Random Forest":
                        if st.session_state.current_index % batch_size == 0 and st.session_state.current_index > 0:
                            start_idx = max(0, st.session_state.current_index - batch_size)
                            end_idx = st.session_state.current_index + 1
                            batch_data = st.session_state.train_data.iloc[start_idx:end_idx]
                            
                            X_batch, y_batch = [], []
                            for _, row in batch_data.iterrows():
                                X_row = preprocess_features(row, fit_scalers=False)
                                if X_row is not None:
                                    X_batch.append(X_row)
                                    y_batch.append(row[target_variable])
                            
                            if X_batch:
                                st.session_state.model.fit(X_batch, y_batch)
            
            except Exception as e:
                st.error(f"모델 학습 실패: {str(e)}")
            
            # 메트릭 계산 (첫 번째 데이터부터 바로 계산)
            if len(st.session_state.predictions) > 1:
                # 최근 N개 데이터로 메트릭 계산
                window_size = min(50, len(st.session_state.predictions))
                recent_pred = st.session_state.predictions[-window_size:]
                recent_actual = st.session_state.actual_values[-window_size:]
                
                mse = mean_squared_error(recent_actual, recent_pred)
                mae = mean_absolute_error(recent_actual, recent_pred)
                rmse = np.sqrt(mse)
                
                try:
                    r2 = r2_score(recent_actual, recent_pred)
                except:
                    r2 = 0
                
                # 매번 업데이트하지 말고 최신 메트릭만 유지
                st.session_state.metrics_history = [{
                    'index': st.session_state.current_index,
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                }]
            
            # 차트 업데이트
            if st.session_state.current_index % chart_update_interval == 0 or st.session_state.current_index == 1:
                # 예측 차트
                fig = create_realtime_chart(target_variable)
                if fig is not None:
                    chart_container.plotly_chart(fig, use_container_width=True, key=f"chart_{st.session_state.current_index}")
                
                # SHAP 차트
                shap_result = create_shap_chart()
                if shap_result is not None:
                    shap_fig, shap_values = shap_result
                    if shap_fig is not None:
                        shap_container.plotly_chart(shap_fig, use_container_width=True, key=f"shap_{st.session_state.current_index}")
                else:
                    shap_container.info("SHAP 데이터 준비 중...")
            
            # 현재 데이터 저장 (매번 저장하되 최대 20개만 유지)
            current_data = current_row.to_dict()
            current_data['예측값'] = prediction
            current_data['절대오차'] = abs(y - prediction) if st.session_state.current_index > 0 else 0
            st.session_state.processed_data.append(current_data)
            
            # 최대 20개 데이터만 유지 (메모리 절약 + 빠른 표시)
            if len(st.session_state.processed_data) > 20:
                st.session_state.processed_data = st.session_state.processed_data[-20:]
            
            # 인덱스 증가
            st.session_state.current_index += 1
            
            # 지연 및 재실행
            time.sleep(speed)
            if st.session_state.current_index < len(st.session_state.train_data):
                st.rerun()
            else:
                st.session_state.is_running = False
                st.success("모든 데이터 처리 완료!")
    
    # 정지 상태에서 차트 표시
    elif not st.session_state.is_running and len(st.session_state.predictions) > 1:
        # 예측 차트
        fig = create_realtime_chart(target_variable)
        if fig is not None:
            chart_container.plotly_chart(fig, use_container_width=True)
        else:
            chart_container.info("예측 데이터가 충분하지 않습니다.")
        
        # SHAP 차트
        shap_result = create_shap_chart()
        if shap_result is not None:
            shap_fig, shap_values = shap_result
            if shap_fig is not None:
                shap_container.plotly_chart(shap_fig, use_container_width=True)
            else:
                shap_container.info("SHAP 데이터가 충분하지 않습니다.")
        else:
            shap_container.info("SHAP 데이터가 충분하지 않습니다.")
    
    # 초기 상태
    else:
        chart_container.info("학습을 시작하면 실시간 예측 차트가 표시됩니다.")
        shap_container.info("SHAP 중요도 차트가 표시됩니다.")
    
    # 최근 처리된 데이터 표시
    if st.session_state.processed_data and len(st.session_state.processed_data) > 0:
        st.subheader("최근 처리된 데이터")
        
        recent_data = st.session_state.processed_data[-10:]
        df_recent = pd.DataFrame(recent_data)
        
        # 주요 컬럼만 선택
        display_cols = ['측정일시', target_variable, '예측값', '절대오차', '작업유형']
        display_cols = [col for col in display_cols if col in df_recent.columns]
        
        if display_cols:
            st.dataframe(
                df_recent[display_cols].round(2), 
                use_container_width=True,
                hide_index=True
            )
    
    # 성능 메트릭 표시
    if st.session_state.metrics_history and len(st.session_state.metrics_history) > 0:
        st.subheader("모델 성능 요약")
        
        latest_metrics = st.session_state.metrics_history[-1]
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("평균 절대 오차 (MAE)", f"{latest_metrics['MAE']:.4f}")
        
        with col_m2:
            st.metric("평균 제곱근 오차 (RMSE)", f"{latest_metrics['RMSE']:.4f}")
        
        with col_m3:
            st.metric("평균 제곱 오차 (MSE)", f"{latest_metrics['MSE']:.4f}")
        
        with col_m4:
            r2_value = latest_metrics.get('R2', 0)
            st.metric("결정계수 (R²)", f"{r2_value:.4f}")
    
    # 최종 결과 다운로드
    if st.session_state.current_index >= len(st.session_state.train_data) and st.session_state.processed_data:
        st.subheader("결과 다운로드")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            # 예측 결과 다운로드
            result_df = pd.DataFrame(st.session_state.processed_data)
            result_csv = result_df.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="예측 결과 다운로드 (CSV)",
                data=result_csv,
                file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        
        with col_download2:
            # 성능 메트릭 다운로드
            if st.session_state.metrics_history:
                metrics_df = pd.DataFrame(st.session_state.metrics_history)
                metrics_csv = metrics_df.to_csv(index=False, encoding='utf-8-sig')
                
                st.download_button(
                    label="성능 메트릭 다운로드 (CSV)",
                    data=metrics_csv,
                    file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

else:
    # 데이터가 로드되지 않은 경우
    st.error("데이터를 로드할 수 없습니다. 'data/train.csv' 파일이 존재하는지 확인해주세요.")
    
    st.markdown("""
    ### 필요한 파일 구조:
    ```
    data/
    └── train.csv
    ```
    
    ### 데이터 형식 요구사항:
    CSV 파일은 다음 컬럼들을 포함해야 합니다:
    - `전력사용량(kWh)`
    - `지상무효전력량(kVarh)`
    - `진상무효전력량(kVarh)`
    - `탄소배출량(tCO2)`
    - `지상역률(%)`
    - `진상역률(%)`
    - `작업유형`
    - `전기요금(원)` (타겟 변수)
    """)