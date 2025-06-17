#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전력 사용량 분석 대시보드
완전한 단일 파일 버전 - 독립 실행 가능
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# PDF 생성 관련 라이브러리 - 전역에서 확인
PDF_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns
    import io
    import base64
    from PIL import Image as PILImage
    import tempfile
    import os
    
    PDF_AVAILABLE = True
    print("✅ PDF 라이브러리 로드 성공")
except ImportError as e:
    PDF_AVAILABLE = False
    print(f"⚠️ PDF 라이브러리 로드 실패: {e}")

# 페이지 설정
st.set_page_config(
    page_title="통합 전력 분석 대시보드", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .comparison-table {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-style {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .download-section {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .kpi-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #e3ffe7 0%, #d9e7ff 100%);
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로딩 함수
@st.cache_data
def load_data():
    """데이터 로드 및 전처리"""
    try:
        # 실제 데이터 파일 시도
        df = pd.read_csv("./data/train.csv")
        st.success("✅ 실제 데이터 파일을 성공적으로 로드했습니다!")
        
    except FileNotFoundError:
        st.warning("⚠️ './data/train.csv' 파일을 찾을 수 없습니다. 샘플 데이터로 시연합니다.")
        
        # 샘플 데이터 생성 (더 현실적인 데이터)
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
        np.random.seed(42)
        
        # 시간대별 패턴을 반영한 샘플 데이터
        hours = dates.hour
        
        # 주간/야간 패턴
        base_power = 80 + 40 * np.sin((hours - 6) * np.pi / 12)  # 주간 높음
        base_power = np.maximum(base_power, 30)  # 최소값 설정
        
        # 작업유형별 패턴
        work_types = np.random.choice(['Light_Load', 'Medium_Load', 'Maximum_Load'], 
                                    len(dates), p=[0.4, 0.35, 0.25])
        
        # 작업유형에 따른 전력 조정
        power_multiplier = np.where(work_types == 'Light_Load', 0.7,
                            np.where(work_types == 'Medium_Load', 1.0, 1.5))
        
        power_usage = base_power * power_multiplier + np.random.normal(0, 10, len(dates))
        power_usage = np.maximum(power_usage, 10)  # 음수 방지
        
        # 전기요금 (누진제 반영)
        electric_cost = power_usage * (120 + np.random.normal(0, 20, len(dates)))
        electric_cost = np.maximum(electric_cost, 0)
        
        sample_data = {
            '측정일시': dates,
            '작업유형': work_types,
            '전력사용량(kWh)': power_usage,
            '전기요금(원)': electric_cost,
            '탄소배출량(tCO2)': power_usage * 0.45 + np.random.normal(0, 2, len(dates)),
            '지상역률(%)': np.random.normal(95, 3, len(dates)),
            '진상역률(%)': np.random.normal(90, 4, len(dates)),
            '지상무효전력량(kVarh)': power_usage * 0.2 + np.random.normal(0, 3, len(dates)),
            '진상무효전력량(kVarh)': power_usage * 0.15 + np.random.normal(0, 2, len(dates))
        }
        
        df = pd.DataFrame(sample_data)
        
    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        return None

    # 공통 전처리
    try:
        df["측정일시"] = pd.to_datetime(df["측정일시"])
        df["날짜"] = df["측정일시"].dt.date
        df["시간"] = df["측정일시"].dt.hour
        df["월"] = df["측정일시"].dt.month
        df["일"] = df["측정일시"].dt.day
        df["년월"] = df["측정일시"].dt.to_period("M")
        
        # 음수값 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].abs()  # 음수를 양수로 변환
        
        return df
        
    except Exception as e:
        st.error(f"❌ 데이터 전처리 중 오류 발생: {e}")
        return None

# 차트 생성 함수들
def create_dual_axis_chart(df, x_col, y1_col, y2_col, title, x_title, y1_title, y2_title, add_time_zones=False):
    """이중 y축 차트 생성 (개선된 버전)"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 시간대별 배경색 추가
    if add_time_zones and x_title == "시간":
        fig.add_vrect(x0=-0.5, x1=5.5, fillcolor="rgba(25, 25, 112, 0.1)", 
                     layer="below", line_width=0, annotation_text="야간 (00-05시)",
                     annotation_position="top left", annotation_font_size=10)
        
        fig.add_vrect(x0=5.5, x1=17.5, fillcolor="rgba(255, 193, 7, 0.1)", 
                     layer="below", line_width=0, annotation_text="주간 (06-17시)",
                     annotation_position="top", annotation_font_size=10)
        
        fig.add_vrect(x0=17.5, x1=23.5, fillcolor="rgba(138, 43, 226, 0.1)", 
                     layer="below", line_width=0, annotation_text="저녁 (18-23시)",
                     annotation_position="top right", annotation_font_size=10)

    # 첫 번째 y축 (왼쪽) - 더 두껍고 매끄러운 선
    fig.add_trace(
        go.Scatter(
            x=df[x_col], y=df[y1_col], name=y1_title,
            line=dict(color="#2E86C1", width=4, smoothing=1.3),
            mode="lines+markers", marker=dict(size=8, color="#2E86C1"),
            hovertemplate=f"<b>{y1_title}</b><br>%{{x}}<br>%{{y:,.2f}}<extra></extra>"
        ), secondary_y=False
    )

    # 두 번째 y축 (오른쪽)
    fig.add_trace(
        go.Scatter(
            x=df[x_col], y=df[y2_col], name=y2_title,
            line=dict(color="#E74C3C", width=4, smoothing=1.3),
            mode="lines+markers", marker=dict(size=8, color="#E74C3C"),
            hovertemplate=f"<b>{y2_title}</b><br>%{{x}}<br>%{{y:,.2f}}<extra></extra>"
        ), secondary_y=True
    )

    # 축 설정
    fig.update_xaxes(title_text=x_title, showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title_text=y1_title, secondary_y=False, title_font_color="#2E86C1")
    fig.update_yaxes(title_text=y2_title, secondary_y=True, title_font_color="#E74C3C")

    # 레이아웃 설정
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        hovermode="x unified", template="plotly_white", height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="rgba(248, 249, 250, 0.8)"
    )

    return fig

def create_hourly_stack_chart(df):
    """시간대별 작업유형별 전기요금 스택 차트 (개선된 버전)"""
    hourly_worktype = df.groupby(["시간", "작업유형"])["전기요금(원)"].sum().unstack(fill_value=0)

    # 전문적인 색상 팔레트
    colors_map = {
        "Light_Load": "#28A745",     # 녹색
        "Medium_Load": "#FFC107",    # 노란색  
        "Maximum_Load": "#DC3545"    # 빨간색
    }

    fig = go.Figure()
    
    for work_type in hourly_worktype.columns:
        fig.add_trace(
            go.Bar(
                name=work_type,
                x=hourly_worktype.index,
                y=hourly_worktype[work_type],
                marker_color=colors_map.get(work_type, "#6C757D"),
                marker_line=dict(width=0.5, color="white"),
                hovertemplate=f"<b>{work_type}</b><br>시간: %{{x}}시<br>전기요금: %{{y:,.0f}}원<extra></extra>"
            )
        )

    fig.update_layout(
        barmode="stack",
        title=dict(text="<b>시간대별 작업유형별 전기요금 현황</b>", x=0.5, font=dict(size=16)),
        xaxis_title="시간 (Hour)",
        yaxis_title="전기요금 (원)",
        xaxis=dict(tickmode="linear", dtick=1, range=[-0.5, 23.5]),
        yaxis=dict(tickformat=",.0f"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=550,
        plot_bgcolor="rgba(248, 249, 250, 0.8)",
        paper_bgcolor="white"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    
    return fig

def create_enhanced_donut_chart(df):
    """향상된 도넛 차트"""
    worktype_mwh = df.groupby("작업유형")["전력사용량(kWh)"].sum() / 1000
    total_mwh = worktype_mwh.sum()

    colors_map = {
        "Light_Load": "#28A745",
        "Medium_Load": "#FFC107", 
        "Maximum_Load": "#DC3545"
    }

    colors = [colors_map.get(wt, "#6C757D") for wt in worktype_mwh.index]

    fig = go.Figure(data=[
        go.Pie(
            labels=worktype_mwh.index,
            values=worktype_mwh.values,
            hole=0.6,
            marker=dict(colors=colors, line=dict(color="white", width=3)),
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(size=12, color="#2C3E50"),
            hovertemplate="<b>%{label}</b><br>사용량: %{value:.2f} MWh<br>비율: %{percent}<extra></extra>",
            pull=[0.1, 0.1, 0.1]
        )
    ])

    fig.update_layout(
        title=dict(text="<b>작업유형별 전력사용량 분포</b>", x=0.5, font=dict(size=16)),
        height=550,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
        annotations=[
            dict(
                text=f"<b>총 사용량</b><br><span style='font-size:18px'>{total_mwh:,.1f}</span><br><b>MWh</b>",
                x=0.5, y=0.5, font=dict(size=14, color="#2C3E50"),
                showarrow=False, align="center"
            )
        ],
        paper_bgcolor="white"
    )

    return fig

def create_summary_table(current_data, period_type="일"):
    """요약 테이블 생성"""
    numeric_columns = [
        ("전력사용량(kWh)", "kWh"),
        ("지상무효전력량(kVarh)", "kVarh"), 
        ("진상무효전력량(kVarh)", "kVarh"),
        ("탄소배출량(tCO2)", "tCO2"),
        ("지상역률(%)", "%"),
        ("진상역률(%)", "%"),
        ("전기요금(원)", "원")
    ]
    
    ratio_cols = {"지상역률(%)", "진상역률(%)"}
    rows = []
    
    for col, unit in numeric_columns:
        if col in current_data.columns:
            if col in ratio_cols:
                val = current_data[col].mean()
            else:
                val = current_data[col].sum()
            name = col.split("(")[0]
            rows.append({
                "항목": name, 
                f"현재{period_type} 값": f"{val:,.2f}", 
                "단위": unit
            })

    return pd.DataFrame(rows)

def create_comparison_table(current_data, previous_data, period_type="일"):
    """비교 분석 테이블 생성"""
    comparison_dict = {
        "항목": [],
        f"현재{period_type}": [],
        f"이전{period_type}": [], 
        "변화량": [],
        "변화율(%)": []
    }

    numeric_columns = [
        "전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)",
        "탄소배출량(tCO2)", "지상역률(%)", "진상역률(%)", "전기요금(원)"
    ]

    for col in numeric_columns:
        if col in current_data.columns and col in previous_data.columns:
            if col in ["지상역률(%)", "진상역률(%)"]:
                current_val = current_data[col].mean()
                previous_val = previous_data[col].mean()
            else:
                current_val = current_data[col].sum()
                previous_val = previous_data[col].sum()

            change = current_val - previous_val
            change_pct = (change / previous_val * 100) if previous_val != 0 else 0

            comparison_dict["항목"].append(col.split("(")[0])
            comparison_dict[f"현재{period_type}"].append(f"{current_val:,.2f}")
            comparison_dict[f"이전{period_type}"].append(f"{previous_val:,.2f}")
            comparison_dict["변화량"].append(f"{change:+,.2f}")
            comparison_dict["변화율(%)"].append(f"{change_pct:+.1f}%")

    return pd.DataFrame(comparison_dict)

# PDF 생성 함수들
if PDF_AVAILABLE:
    def setup_korean_font():
        """한글 폰트 설정 함수"""
        try:
            font_paths = [
                'C:/Windows/Fonts/malgun.ttf',
                'C:/Windows/Fonts/gulim.ttc',
                '/System/Library/Fonts/AppleGothic.ttf',
                '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font_prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = font_prop.get_name()
                    plt.rcParams['axes.unicode_minus'] = False
                    return font_path
            
            plt.rcParams['font.family'] = 'DejaVu Sans'
            return None
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            return None

    def create_chart_for_pdf(df, chart_type, save_path, figsize=(12, 8)):
        """PDF용 고품질 차트 생성"""
        setup_korean_font()
        
        # 전문적인 스타일 설정
        plt.style.use('default')
        
        # 색상 팔레트
        colors_prof = ['#2E86C1', '#E74C3C', '#28A745', '#F39C12', '#8E44AD', '#17A2B8']
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='white', dpi=300)
        
        if chart_type == "hourly_analysis":
            # 시간대별 전기요금 분석
            hourly_data = df.groupby(['시간', '작업유형'])['전기요금(원)'].sum().unstack(fill_value=0)
            
            bottom = np.zeros(len(hourly_data.index))
            for i, work_type in enumerate(hourly_data.columns):
                ax.bar(hourly_data.index, hourly_data[work_type], 
                      bottom=bottom, label=work_type, 
                      color=colors_prof[i % len(colors_prof)], alpha=0.8)
                bottom += hourly_data[work_type]
            
            ax.set_title('시간대별 작업유형별 전기요금 현황', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('시간', fontsize=12, fontweight='bold')
            ax.set_ylabel('전기요금 (원)', fontsize=12, fontweight='bold')
            ax.legend(title='작업유형', loc='upper left')
            ax.grid(True, alpha=0.3)
            
        elif chart_type == "worktype_distribution":
            # 작업유형별 분포
            worktype_data = df.groupby('작업유형')['전력사용량(kWh)'].sum()
            
            wedges, texts, autotexts = ax.pie(
                worktype_data.values, labels=worktype_data.index,
                autopct='%1.1f%%', colors=colors_prof[:len(worktype_data)],
                startangle=90, explode=[0.05] * len(worktype_data)
            )
            
            ax.set_title('작업유형별 전력사용량 분포', fontsize=16, fontweight='bold', pad=20)
            
        elif chart_type == "daily_trend":
            # 일별 트렌드
            daily_data = df.groupby('날짜').agg({
                '전력사용량(kWh)': 'sum',
                '전기요금(원)': 'sum',
                '탄소배출량(tCO2)': 'sum'
            }).reset_index()
            
            if len(daily_data) > 1:
                ax2 = ax.twinx()
                
                line1 = ax.plot(daily_data['날짜'], daily_data['전력사용량(kWh)'], 
                               color=colors_prof[0], linewidth=3, marker='o', 
                               label='전력사용량(kWh)')
                
                line2 = ax2.plot(daily_data['날짜'], daily_data['전기요금(원)'], 
                                color=colors_prof[1], linewidth=3, marker='s',
                                label='전기요금(원)')
                
                ax.set_ylabel('전력사용량 (kWh)', color=colors_prof[0], fontweight='bold')
                ax2.set_ylabel('전기요금 (원)', color=colors_prof[1], fontweight='bold')
                
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='upper left')
                
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            ax.set_title('일별 전력사용량 및 전기요금 추이', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
        
        # 공통 스타일링
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path

    def generate_comprehensive_pdf_report(df, filename="comprehensive_power_analysis.pdf"):
        """종합적인 PDF 보고서 생성"""
        try:
            temp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(temp_dir, filename)
            
            doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                                  topMargin=0.75*inch, bottomMargin=0.75*inch,
                                  leftMargin=0.75*inch, rightMargin=0.75*inch)
            story = []
            
            # 스타일 정의
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
                                       fontSize=20, alignment=TA_CENTER, spaceAfter=30,
                                       textColor=colors.HexColor('#1B365D'))
            
            section_style = ParagraphStyle('SectionHeader', fontSize=14,
                                         alignment=TA_LEFT, spaceAfter=15, spaceBefore=20,
                                         textColor=colors.HexColor('#1B365D'),
                                         backColor=colors.HexColor('#F0F8FF'))
            
            body_style = ParagraphStyle('Body', fontSize=11, alignment=TA_LEFT,
                                      spaceAfter=12, leading=16)
            
            # 제목 페이지
            story.append(Paragraph("전력 사용량 종합 분석 보고서", title_style))
            story.append(Spacer(1, 30))
            
            # 기본 정보
            total_kwh = df["전력사용량(kWh)"].sum()
            total_cost = df["전기요금(원)"].sum()
            avg_price = total_cost / total_kwh if total_kwh > 0 else 0
            total_carbon = df["탄소배출량(tCO2)"].sum()
            
            summary_text = f"""
            <b>분석 개요</b><br/>
            • 분석 기간: {df['측정일시'].min().date()} ~ {df['측정일시'].max().date()}<br/>
            • 총 데이터: {len(df):,}건<br/>
            • 총 전력사용량: {total_kwh:,.0f} kWh<br/>
            • 총 전기요금: {total_cost:,.0f} 원<br/>
            • 평균 단가: {avg_price:.2f} 원/kWh<br/>
            • 총 탄소배출량: {total_carbon:.2f} tCO2<br/>
            """
            
            story.append(Paragraph(summary_text, body_style))
            story.append(Spacer(1, 30))
            
            # 차트 생성 및 추가
            if len(df) > 0:
                # 시간대별 분석 차트
                chart1_path = os.path.join(temp_dir, "hourly_analysis.png")
                create_chart_for_pdf(df, "hourly_analysis", chart1_path)
                
                story.append(Paragraph("1. 시간대별 전기요금 분석", section_style))
                if os.path.exists(chart1_path):
                    story.append(Image(chart1_path, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 15))
                
                # 작업유형별 분포 차트
                chart2_path = os.path.join(temp_dir, "worktype_dist.png")
                create_chart_for_pdf(df, "worktype_distribution", chart2_path)
                
                story.append(Paragraph("2. 작업유형별 전력사용량 분포", section_style))
                if os.path.exists(chart2_path):
                    story.append(Image(chart2_path, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 15))
                
                # 일별 트렌드 차트
                chart3_path = os.path.join(temp_dir, "daily_trend.png")
                create_chart_for_pdf(df, "daily_trend", chart3_path)
                
                story.append(Paragraph("3. 일별 사용량 및 요금 추이", section_style))
                if os.path.exists(chart3_path):
                    story.append(Image(chart3_path, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 15))
            
            # 상세 분석 표
            story.append(Paragraph("4. 작업유형별 상세 분석", section_style))
            
            worktype_stats = df.groupby("작업유형").agg({
                "전력사용량(kWh)": "sum",
                "전기요금(원)": "sum",
                "탄소배출량(tCO2)": "sum"
            }).round(2)
            
            table_data = [["작업유형", "전력사용량(kWh)", "전기요금(원)", "탄소배출량(tCO2)"]]
            for work_type in worktype_stats.index:
                table_data.append([
                    work_type,
                    f"{worktype_stats.loc[work_type, '전력사용량(kWh)']:,.0f}",
                    f"{worktype_stats.loc[work_type, '전기요금(원)']:,.0f}",
                    f"{worktype_stats.loc[work_type, '탄소배출량(tCO2)']:,.2f}"
                ])
            
            table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FBFF')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#4A90A4')),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
            
            # 주요 발견사항
            peak_hour = df.groupby('시간')['전기요금(원)'].sum().idxmax()
            peak_cost = df.groupby('시간')['전기요금(원)'].sum().max()
            
            findings = f"""
            <b>5. 주요 발견사항 및 권장사항</b><br/><br/>
            
            <b>핵심 발견사항:</b><br/>
            • 피크 사용 시간: {peak_hour}시 ({peak_cost:,.0f}원)<br/>
            • 가장 많이 사용하는 작업유형: {df.groupby('작업유형')['전력사용량(kWh)'].sum().idxmax()}<br/>
            • 평균 탄소집약도: {total_carbon/total_kwh*1000:.2f} kg CO2/kWh<br/><br/>
            
            <b>권장사항:</b><br/>
            1. 피크 시간대({peak_hour}시) 전력 사용량 관리<br/>
            2. 작업유형별 효율성 개선 검토<br/>
            3. 재생에너지 도입을 통한 탄소배출 감축<br/>
            4. 실시간 모니터링 시스템 구축<br/>
            """
            
            story.append(Paragraph(findings, body_style))
            story.append(Spacer(1, 30))
            
            # 푸터
            footer_text = f"보고서 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 에너지 분석팀"
            footer_style = ParagraphStyle('Footer', fontSize=8, alignment=TA_CENTER, textColor=colors.grey)
            story.append(Paragraph(footer_text, footer_style))
            
            # PDF 생성
            doc.build(story)
            return pdf_path
            
        except Exception as e:
            raise Exception(f"PDF 생성 실패: {str(e)}")

    def get_download_link(file_path, filename):
        """파일 다운로드 링크 생성"""
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration: none; background-color: #28A745; color: white; padding: 12px 24px; border-radius: 8px; font-weight: bold; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">📥 PDF 보고서 다운로드</a>'
            return href
        except Exception as e:
            st.error(f"다운로드 링크 생성 실패: {e}")
            return None

# 메인 함수
def main():
    # 헤더
    st.markdown("""
    <div class="header-style">
        <h1>⚡ 통합 전력 분석 대시보드</h1>
        <p>전력 사용량, 무효전력, 탄소배출량 통합 분석 및 최적화 솔루션</p>
    </div>
    """, unsafe_allow_html=True)

    # 데이터 로드
    df = load_data()
    if df is None:
        st.stop()

    # 사이드바 설정
    st.sidebar.markdown("## 🔧 분석 설정")

    # PDF 기능 확인
    if not PDF_AVAILABLE:
        st.sidebar.warning("⚠️ PDF 생성 기능 비활성화")
        st.sidebar.info("다음 명령어로 라이브러리를 설치하세요:")
        st.sidebar.code("pip install reportlab matplotlib seaborn pillow")

    # PDF 보고서 생성 섹션
    if PDF_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📊 PDF 보고서")
        
        if st.sidebar.button("🎯 종합 분석 보고서 생성", type="primary", use_container_width=True):
            with st.spinner("📊 전문 보고서를 생성하고 있습니다..."):
                try:
                    pdf_path = generate_comprehensive_pdf_report(df)
                    st.sidebar.success("✅ 보고서 생성 완료!")
                    
                    filename = f"전력분석보고서_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    download_link = get_download_link(pdf_path, filename)
                    
                    if download_link:
                        st.sidebar.markdown(download_link, unsafe_allow_html=True)
                        st.success("🎉 PDF 보고서가 생성되었습니다! 사이드바에서 다운로드하세요.")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ 보고서 생성 실패: {str(e)}")
                    st.error("보고서 생성 중 오류가 발생했습니다.")

    # 분석 옵션
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📈 분석 옵션")
    
    numeric_columns = [
        "전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)",
        "탄소배출량(tCO2)", "지상역률(%)", "진상역률(%)", "전기요금(원)"
    ]
    
    available_cols = [col for col in numeric_columns if col in df.columns]
    
    col1_select = st.sidebar.selectbox("첫 번째 분석 컬럼", available_cols, index=0)
    col2_select = st.sidebar.selectbox("두 번째 분석 컬럼", available_cols, 
                                     index=min(len(available_cols)-1, 6))

    # === 메인 대시보드 ===
    st.markdown("""
    <div class="section-header">
        <h2>📊 전체 현황 대시보드</h2>
    </div>
    """, unsafe_allow_html=True)

    # KPI 메트릭
    total_kwh = df["전력사용량(kWh)"].sum()
    total_cost = df["전기요금(원)"].sum()
    avg_price = total_cost / total_kwh if total_kwh > 0 else 0
    total_carbon = df["탄소배출량(tCO2)"].sum()

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
        st.metric("⚡ 총 전력사용량", f"{total_kwh:,.0f} kWh")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
        st.metric("💰 총 전기요금", f"{total_cost:,.0f} 원")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
        st.metric("📊 평균 단가", f"{avg_price:.1f} 원/kWh")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
        st.metric("🌱 탄소배출량", f"{total_carbon:.1f} tCO2")
        st.markdown('</div>', unsafe_allow_html=True)

    # 메인 차트
    col_chart1, col_chart2 = st.columns([3, 2])
    
    with col_chart1:
        st.plotly_chart(create_hourly_stack_chart(df), use_container_width=True)
    
    with col_chart2:
        st.plotly_chart(create_enhanced_donut_chart(df), use_container_width=True)

    # 작업유형별 분석 테이블
    st.markdown("### 📋 작업유형별 상세 현황")
    
    worktype_analysis = df.groupby("작업유형").agg({
        "전력사용량(kWh)": ["sum", "mean"],
        "전기요금(원)": ["sum", "mean"], 
        "탄소배출량(tCO2)": ["sum", "mean"],
        "지상역률(%)": "mean"
    }).round(2)
    
    # 컬럼명 정리
    worktype_analysis.columns = [
        "총 전력량(kWh)", "평균 전력량(kWh)",
        "총 전기요금(원)", "평균 전기요금(원)",
        "총 탄소배출량(tCO2)", "평균 탄소배출량(tCO2)",
        "평균 지상역률(%)"
    ]
    
    st.dataframe(worktype_analysis, use_container_width=True)

    st.markdown("---")

    # === 시간대별 상세 분석 ===
    st.markdown("""
    <div class="section-header">
        <h2>🕐 시간대별 상세 분석</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 날짜 선택
        available_dates = sorted(df["날짜"].unique())
        
        if available_dates:
            selected_date = st.date_input(
                "🗓️ 분석할 날짜 선택",
                value=available_dates[-1],
                min_value=available_dates[0],
                max_value=available_dates[-1]
            )

            # 선택된 날짜 데이터
            daily_df = df[df["날짜"] == selected_date]
            
            if not daily_df.empty:
                # 시간별 집계
                hourly_data = daily_df.groupby("시간").agg({
                    col1_select: "sum" if col1_select not in ["지상역률(%)", "진상역률(%)"] else "mean",
                    col2_select: "sum" if col2_select not in ["지상역률(%)", "진상역률(%)"] else "mean"
                }).reset_index()

                # 24시간 데이터 보정
                full_hours = pd.DataFrame({"시간": list(range(24))})
                hourly_data = pd.merge(full_hours, hourly_data, on="시간", how="left").fillna(0)

                # 차트 생성
                fig = create_dual_axis_chart(
                    hourly_data, "시간", col1_select, col2_select,
                    f"{selected_date} 시간별 {col1_select} vs {col2_select}",
                    "시간", col1_select, col2_select, add_time_zones=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ {selected_date}에 해당하는 데이터가 없습니다.")
    
    with col2:
        st.markdown("### 📊 일별 요약")
        
        if 'daily_df' in locals() and not daily_df.empty:
            daily_summary = create_summary_table(daily_df, "일")
            st.dataframe(daily_summary, use_container_width=True, hide_index=True)
            
            # 주요 지표
            st.markdown("### 🎯 핵심 지표")
            day_cost = daily_df["전기요금(원)"].sum()
            day_power = daily_df["전력사용량(kWh)"].sum()
            
            st.metric("일 전기요금", f"{day_cost:,.0f} 원")
            st.metric("일 전력사용량", f"{day_power:,.1f} kWh")
            
            if day_power > 0:
                st.metric("일 평균단가", f"{day_cost/day_power:.2f} 원/kWh")

    st.markdown("---")

    # === 기간별 비교 분석 ===
    st.markdown("""
    <div class="section-header">
        <h2>📈 기간별 비교 분석</h2>
    </div>
    """, unsafe_allow_html=True)

    view_type = st.selectbox("📅 분석 단위 선택", ["일별", "월별"], index=0)
    
    if view_type == "월별":
        # 월별 분석
        monthly_data = df.groupby("년월").agg({
            col1_select: "sum" if col1_select not in ["지상역률(%)", "진상역률(%)"] else "mean",
            col2_select: "sum" if col2_select not in ["지상역률(%)", "진상역률(%)"] else "mean"
        }).reset_index()
        
        monthly_data["년월_str"] = monthly_data["년월"].astype(str)
        
        if not monthly_data.empty:
            fig = create_dual_axis_chart(
                monthly_data, "년월_str", col1_select, col2_select,
                f"월별 {col1_select} vs {col2_select} 추이",
                "월", col1_select, col2_select
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # 일별 분석
        date_range_selection = st.date_input(
            "📅 분석 기간 선택",
            value=(available_dates[0], available_dates[-1]),
            min_value=available_dates[0],
            max_value=available_dates[-1]
        )
        
        if isinstance(date_range_selection, tuple) and len(date_range_selection) == 2:
            start_date, end_date = date_range_selection
            
            period_df = df[(df["날짜"] >= start_date) & (df["날짜"] <= end_date)]
            
            if not period_df.empty:
                daily_data = period_df.groupby("날짜").agg({
                    col1_select: "sum" if col1_select not in ["지상역률(%)", "진상역률(%)"] else "mean",
                    col2_select: "sum" if col2_select not in ["지상역률(%)", "진상역률(%)"] else "mean"
                }).reset_index()
                
                fig = create_dual_axis_chart(
                    daily_data, "날짜", col1_select, col2_select,
                    f"{start_date} ~ {end_date} 일별 {col1_select} vs {col2_select}",
                    "날짜", col1_select, col2_select
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 기간 요약
                st.markdown("### 📊 선택 기간 요약")
                period_summary = create_summary_table(period_df, "기간")
                st.dataframe(period_summary, use_container_width=True, hide_index=True)

    # === 추가 정보 및 도움말 ===
    with st.expander("📋 데이터 상세 정보"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 전체 데이터 건수", f"{len(df):,}개")
            st.metric("📅 분석 기간", f"{(df['측정일시'].max() - df['측정일시'].min()).days + 1}일")
        
        with col2:
            st.metric("🏭 작업 유형 수", f"{df['작업유형'].nunique()}개")
            st.metric("📈 데이터 완성도", f"{(1 - df.isnull().sum().sum()/(len(df)*len(df.columns)))*100:.1f}%")
        
        with col3:
            st.metric("⚡ 최대 시간당 사용량", f"{df['전력사용량(kWh)'].max():,.1f} kWh")
            st.metric("💰 최대 시간당 요금", f"{df['전기요금(원)'].max():,.0f} 원")

        # 데이터 미리보기
        st.markdown("### 📋 원본 데이터 미리보기")
        preview_cols = ["측정일시", "작업유형", "전력사용량(kWh)", "전기요금(원)", "탄소배출량(tCO2)"]
        available_preview = [col for col in preview_cols if col in df.columns]
        
        st.dataframe(df[available_preview].head(10), use_container_width=True)

    # 하단 정보
    st.markdown("""
    <div class="download-section">
        <h3>🎯 대시보드 주요 기능</h3>
        <p>✅ 실시간 전력 사용량 모니터링 | ✅ 작업유형별 효율성 분석 | ✅ 시간대별 패턴 분석</p>
        <p>✅ 탄소배출량 추적 | ✅ 전문 PDF 보고서 생성 | ✅ 비교 분석 및 트렌드 예측</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()