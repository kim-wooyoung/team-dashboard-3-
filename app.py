import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO

st.set_page_config(page_title="충청본부 팀별 업무일지 분석 대시보드", layout="wide")

# ✅ 로고 base64 인코딩해서 세션에 저장
@st.cache_data
def load_logo_base64(path='로고.jpg'):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        return encoded

if 'logo_base64' not in st.session_state:
    try:
        st.session_state['logo_base64'] = load_logo_base64("로고.jpg")
    except FileNotFoundError:
        st.session_state['logo_base64'] = ""

st.markdown("""
<style>
.metric-box {
  border: 1px solid #e6e6e6;
  border-radius: 10px;
  padding: 15px;
  margin-bottom: 10px;
  background-color: #f9f9f9;
  text-align: center;
}
</style>
""", unsafe_allow_html=True)

def process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['원본작업자'] = df['작업자']
    df['시작일시'] = pd.to_datetime(df['시작일시'])
    df['종료일시'] = pd.to_datetime(df['종료일시'])
    df['작업시간(분)'] = (df['종료일시'] - df['시작일시']).dt.total_seconds() / 60
    df['조구성'] = df['원본작업자'].astype(str).apply(lambda x: '2인 1조' if ',' in x else '1인 1조')
    df['작업자'] = df['작업자'].str.split(',')
    df = df.explode('작업자')
    df['작업자'] = df['작업자'].str.strip()
    df['주차'] = df['시작일시'].apply(lambda x: f"{x.month}월{x.day // 7 + 1}주")
    df['작업일'] = pd.to_datetime(df['시작일시'].dt.date)
    return df

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def show_metric(title, value):
    st.markdown(f"""
    <div class="metric-box">
        <h4>{title}</h4>
        <h2 style='color: #0072C6'>{value}</h2>
    </div>
    """, unsafe_allow_html=True)

def main():
    import os

    col1, col2 = st.columns([8, 1])
    with col1:
        st.markdown("## 📊 충청본부 팀별 업무일지 분석 대시보드")
    with col2:
        try:
            with open("로고.jpg", "rb") as image_file:
                logo_base64 = base64.b64encode(image_file.read()).decode()
        except FileNotFoundError:
            logo_base64 = "iVBORw0KGgoAAAANSUhEUgAAAHgAAAAoCAYAAABqZ0U9AAAACXBIWXMAAA7EAAAOxAGVKw4bAAABKklEQVR4nO3aMU7DQBiG4a/rRFdiC0mEnYVfgm7AeEKiwHoQHe4mF8BcwmOMQhsUv/xub1vMFkcfz36Wc/DMAAAAAAAAAAKADhPXywW16iB3tIE5xk1WKwK07Wrpnbpi7U/o7frHWifXZorulxI60fQfcs77BfTkIcBNvS1s3bVG+ewSPm2nni0+b0Udt+tFb2waXEv2gSYplvkp8BZXVtR3GbvV9mpo5jpu2X7Nyf3HbDK5RTroUVL+IrlJvn/M2hPNYbtX1eLjsKpbVnTwvphbhxwAAAAAAAAAAODfDooMcuPqDCW2AAAAAElFTkSuQmCC"

        st.markdown(f"""
        <div style='display: flex; justify-content: flex-end;'>
            <img src='data:image/jpeg;base64,{logo_base64}' width='180'>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("업무일지를 업로드하고, 팀과 팀원별로 분석 결과를 확인하세요.")

    uploaded_file = st.file_uploader("📁 업무일지 CSV 파일 업로드", type=["csv"])
    st.markdown("""
<div style='padding: 12px; background-color: #f0f8ff; border-left: 5px solid #0072C6; font-weight: bold; font-size: 16px;'>
📥 MOStagram 업무일지에서 파일을 다운로드한 후 그대로 업로드하면 대시보드가 표시됩니다.
</div>
""", unsafe_allow_html=True)

    if uploaded_file:
        df = process_data(uploaded_file)

        with st.sidebar:
            st.header("🔍 검색")
            min_date = df['시작일시'].min().date()
            max_date = df['종료일시'].max().date()
            start_date, end_date = st.date_input("작업 기간 필터", [min_date, max_date], min_value=min_date, max_value=max_date)

        df = df[(df['시작일시'].dt.date >= start_date) & (df['종료일시'].dt.date <= end_date)]

        if '팀' in df.columns:
            team_options = df['팀'].dropna().unique().tolist()
            selected_team = st.sidebar.selectbox("팀 선택", ["전체"] + team_options)
            if selected_team != "전체":
                df = df[df['팀'] == selected_team]

            member_options = df['작업자'].dropna().unique().tolist()
            with st.sidebar.expander("**작업자 선택**", expanded=False):
                selected_members = st.multiselect("작업자 목록", options=member_options, default=member_options)
            df = df[df['작업자'].isin(selected_members)]

        

        # 👤 개인별 누락 현황
        all_workers = df.groupby('작업자')['팀'].first().reset_index()
        date_range = pd.date_range(start=df['작업일'].min(), end=df['작업일'].max(), freq='B')
        all_worker_days = pd.MultiIndex.from_product([all_workers['작업자'], date_range], names=['작업자', '작업일'])
        all_worker_days = pd.DataFrame(index=all_worker_days).reset_index().merge(all_workers, on='작업자')

        actual_logs = df.groupby(['팀', '작업자', '작업일']).size()
        log_df = all_worker_days.merge(
            actual_logs.rename('작성여부').reset_index(),
            on=['팀', '작업자', '작업일'],
            how='left'
        ).fillna({'작성여부': 0})

        st.markdown("## ❗ 개인별 업무일지 누락 현황")
        personal_summary = log_df.groupby(['팀', '작업자'])['작성여부'].mean().reset_index()
        personal_summary = personal_summary[personal_summary['작성여부'] < 1.0]
        personal_summary['누락률(%)'] = (1 - personal_summary['작성여부']) * 100
        personal_summary = personal_summary.sort_values('누락률(%)', ascending=False)
        styled_df = personal_summary[['팀', '작업자', '누락률(%)']].round(1).style.apply(
            lambda x: ['background-color: #ffcccc' if v > 30 else '' for v in x], subset=['누락률(%)']
        )
        st.dataframe(styled_df, use_container_width=True)

        
        
        st.markdown("## 🗓️ 운용팀 일별 작성현황")
        daily_count = df.groupby([df['작업일'].dt.date, '팀']).size().unstack(fill_value=0)
        daily_count.loc['합계'] = daily_count.sum()
        st.dataframe(daily_count, use_container_width=True)

        

        st.markdown("## 📉 팀 주차별 가동율 (이동시간 제외)")
        df_no_move = df[df['업무종류'] != '이동업무']
        total_by_week = df_no_move.groupby(['주차']).agg(전체작업시간_분=('작업시간(분)', 'sum')).reset_index()
        df_weekly = df_no_move.groupby(['팀', '주차']).agg(팀작업시간_분=('작업시간(분)', 'sum')).reset_index()
        df_weekly = df_weekly.merge(total_by_week, on='주차')
        주별_작업자수 = df_no_move.groupby(['팀', '주차'])['작업자'].nunique().reset_index(name='작업자수')
        df_weekly = df_weekly.merge(주별_작업자수, on=['팀', '주차'])
        df_weekly['가동율(%)'] = df_weekly['팀작업시간_분'] / (df_weekly['작업자수'] * 2400)
        df_weekly['가동율(%)'] = df_weekly['가동율(%)'].clip(upper=1.0)
        team_count = df['팀'].nunique()

        fig_util = px.bar(
            df_weekly,
            x='팀',
            y='가동율(%)',
            color='주차',
            barmode='group',
            title='팀 주차별 가동율 (이동시간 제외)',
            labels={'가동율(%)': '가동율', '팀': '팀'}
        )
        fig_util.update_layout(
            yaxis_tickformat='.0%',
            yaxis_range=[0, 1],
            legend_title_text='주차',
            legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center')
        )
        fig_util.add_shape(type="line", x0=-0.5, x1=team_count - 0.5, y0=0.68, y1=0.68, line=dict(color="red", width=2, dash="dot"))
        fig_util.add_annotation(x=team_count - 0.5, y=0.68, text="기준: 68%", showarrow=False, yshift=10, font=dict(color="red"))
        st.plotly_chart(fig_util, use_container_width=True)

        st.markdown("## 📊 일별 평균 작업시간 (이동시간 제외)")
        df_no_move['작업일'] = pd.to_datetime(df_no_move['작업일'])
        daily_sum = df_no_move.groupby(['작업일', '팀'])['작업시간(분)'].sum().reset_index()
        daily_workers = df_no_move.groupby(['작업일', '팀'])['작업자'].nunique().reset_index(name='작업자수')
        daily_avg = pd.merge(daily_sum, daily_workers, on=['작업일', '팀'])
        daily_avg['평균작업시간(시간)'] = (daily_avg['작업시간(분)'] / daily_avg['작업자수']) / 60

        fig_daily = px.bar(
            daily_avg,
            x='팀',
            y='평균작업시간(시간)',
            color='작업일',
            barmode='group',
            title='일별 평균 작업시간 (이동시간 제외)',
            labels={'평균작업시간(시간)': '평균 작업시간(시간)', '작업일': '날짜'}
        )
        fig_daily.update_layout(
            yaxis_range=[0, 10],
            legend=dict(
                orientation='h',
                y=-0.25,
                x=0.5,
                xanchor='center'
            )
        )
        fig_daily.add_shape(
            type="line",
            x0=-0.5,
            x1=len(daily_avg['팀'].unique()) - 0.5,
            y0=6.2,
            y1=6.2,
            line=dict(color="red", width=2, dash="dot")
        )
        fig_daily.add_annotation(
            x=len(daily_avg['팀'].unique()) - 0.5,
            y=6.2,
            text="기준: 6.2시간",
            showarrow=False,
            yshift=10,
            font=dict(color="red")
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("## 🧮 팀별 운용조 현황")
        crew_ratio = df[['팀', '작업자', '조구성']].drop_duplicates().copy()
        crew_summary = crew_ratio.groupby(['팀', '조구성']).size().unstack(fill_value=0)
        crew_summary = crew_summary.T
        crew_summary = crew_summary.div(crew_summary.sum(axis=1), axis=0).fillna(0).round(4) * 100
        crew_summary = crew_summary.rename(columns=lambda x: f"{x}")

        crew_summary_percent = crew_summary.div(crew_summary.sum(axis=0), axis=1).fillna(0).round(4) * 100
        st.dataframe(crew_summary_percent.style.format("{:.2f}%"), use_container_width=True)

        # 조별 구성 막대 그래프
        crew_summary_reset = crew_summary_percent.T.reset_index().rename(columns={'index': '팀'}).melt(id_vars='팀', var_name='조구성', value_name='비율')
        fig_crew = px.bar(
            crew_summary_reset,
            x='팀',
            y='비율',
            color='조구성',
            barmode='group',
            title='팀별 운용조 현황',
            labels={'비율': '비율(%)'}
        )
        fig_crew.update_layout(
    yaxis_range=[0, 100],
    yaxis_ticksuffix="%",
    legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center')
)
        st.plotly_chart(fig_crew, use_container_width=True)

        # ✅ 업무구분별 인원조 현황
        st.markdown("## 🧮 업무구분별 인원조 현황")
        crew_task = df[['구분', '조구성']].copy()
        crew_task_grouped = crew_task.groupby(['구분', '조구성']).size().unstack(fill_value=0)
        crew_task_ratio = crew_task_grouped.div(crew_task_grouped.sum(axis=1), axis=0).fillna(0).round(4) * 100

        crew_task_reset = crew_task_ratio.reset_index().melt(id_vars='구분', var_name='조구성', value_name='비율')
        fig_crew_task = px.bar(
            crew_task_reset,
            x='구분',
            y='비율',
            color='조구성',
            barmode='group',
            title='업무구분별 인원조 현황',
            labels={'비율': '비율(%)'}
        )
        fig_crew_task.update_layout(
    yaxis_range=[0, 100],
    yaxis_ticksuffix="%",
    legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center')
)
        st.plotly_chart(fig_crew_task, use_container_width=True)

        

        

        
    

if __name__ == '__main__':
    main()

    

