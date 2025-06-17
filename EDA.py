import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 가져오기
df = pd.read_csv("./data/train.csv")
holiday = pd.read_csv("./data/holiday.csv")

df['측정일시'] = pd.to_datetime(df['측정일시'], errors='raise')
df['날짜'] = df['측정일시'].dt.date

holiday['날짜'] = pd.to_datetime(holiday['날짜']).dt.date

df['is_holiday'] = df['날짜'].isin(holiday['날짜']).astype(int)


# 날짜별 + 공휴일 여부별 평균 전력량
daily_avg = df.groupby(['날짜', 'is_holiday'])['전력사용량(kWh)'].mean().reset_index()

# 공휴일 여부를 문자로 바꿔서 범례 표시 깔끔하게
daily_avg['공휴일여부'] = daily_avg['is_holiday'].map({0: '비공휴일', 1: '공휴일'})

###########
##########
#########

# 하루를 기준으로 만든 "평균적인 전력 사용 패턴"을 보여주는 시각화

# “공휴일과 비공휴일 각각, 하루 중 어떤 시간대에 전기를 얼마나 쓰는 경향이 있는가?”

# 제조공장처럼 반복적인 패턴이 있는 시스템은,
# 하루 단위 시간대별 평균을 비교하는 게 가장 직관적


# 오전 9시~10시에 항상 큰 피크가 있다면 → 출근 후 가동
# 공휴일에는 그 시간대에 피크가 없다면 → 쉬는 날임을 명확히 보여줄 수 있음

# 시간만 추출
df['시간'] = df['측정일시'].dt.strftime('%H:%M')

# 평균 전력량 계산
hourly_avg = df.groupby(['시간', 'is_holiday'])['전력사용량(kWh)'].mean().reset_index()
hourly_avg['공휴일여부'] = hourly_avg['is_holiday'].map({0: '비공휴일', 1: '공휴일'})

# Plotly 선그래프
fig = px.line(hourly_avg, 
              x='시간', 
              y='전력사용량(kWh)', 
              color='공휴일여부',
              title='공휴일 vs 비공휴일 시간대별 전력 사용량 비교',
              color_discrete_map={'공휴일': 'tomato', '비공휴일': 'royalblue'})
fig.update_layout(xaxis_tickangle=45)
fig.show()


















