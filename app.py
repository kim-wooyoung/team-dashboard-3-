import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
import re
from io import BytesIO

st.set_page_config(page_title="업무일지 분석 대시보드", layout="wide")

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

def split_workers(worker_string):
    worker_string = re.sub(r'[.,\s]', ',', str(worker_string))  # 쉼표, 마침표, 공백 → 쉼표
    worker_string = re.sub(r'(?<=[가-힣]{2})(?=[가-힣]{2})', ',', worker_string)  # 붙여쓰기된 한글 이름 분리
    return [name.strip() for name in worker_string.split(',') if name.strip()]

def process_data(uploaded_file):
    df_original = pd.read_csv(uploaded_file)
    df = df_original.copy()
    df['원본작업자'] = df['작업자']
    df['시작일시'] = pd.to_datetime(df['시작일시'])
    df['종료일시'] = pd.to_datetime(df['종료일시'])
    df['작업시간(분)'] = (df['종료일시'] - df['시작일시']).dt.total_seconds() / 60
    # ✅ 동일 작업자 + 동일 시간대 중복 제거
    df = df.drop_duplicates(subset=['작업자', '시작일시', '종료일시'])
    df['작업자목록'] = df['작업자'].apply(split_workers)
    df['조구성'] = df['작업자목록'].apply(lambda x: '2인 1조' if len(x) >= 2 else '1인 1조')
    df = df.explode('작업자목록')
    df['작업자'] = df['작업자목록'].astype(str).str.strip()
    df.drop(columns=['작업자목록'], inplace=True)
    df['주차'] = df['시작일시'].apply(lambda x: f"{x.month}월{x.day // 7 + 1}주")
    df['작업일'] = pd.to_datetime(df['시작일시'].dt.date)
    return df, df_original

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
        st.markdown("""
<h1 style='font-size: 50px;'>📊  <span style='color:#d32f2f;'>MOS</span>tagram 분석 대시보드</h1>
""", unsafe_allow_html=True)
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

    st.markdown("""
<p style='font-size: 25px;'>업무일지를 업로드하고, 팀과 팀원별로 분석 결과를 확인하세요.</p>
""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 work_report.csv 파일 업로드", type=["csv"])
    st.markdown("""
<div style='padding: 12px; background-color: #f0f8ff; border-left: 5px solid #0072C6; font-weight: bold; font-size: 16px;'>
📤 MOStagram 에서 파일을 다운로드한 후 업로드하면 대시보드가 표시됩니다.
</div>
""", unsafe_allow_html=True)

    if uploaded_file:
        df, df_original = process_data(uploaded_file)

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

    # ✅ 사이드바 하단에 CSV 저장 버튼
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💾 전체 결과 다운로드")
        csv = convert_df_to_csv(df)
        st.sidebar.download_button(
            label="📥 CSV 파일 저장",
            data=csv,
            file_name="업무일지_분석결과.csv",
            mime="text/csv"
        )

        

        # 👤 개인별 누락 현황
        # ✅ 작업자 분리 기준 추가
        split_df = df.copy()
        split_df['작업자'] = split_df['작업자'].astype(str)
        split_df['작업자'] = split_df['작업자'].str.replace('.', ',', regex=False).str.replace(' ', ',', regex=False).str.split(',')
        split_df = split_df.explode('작업자')
        split_df['작업자'] = split_df['작업자'].str.strip()

        all_workers = split_df.groupby('작업자')['팀'].first().reset_index()
        date_range = pd.date_range(start=df['작업일'].min(), end=df['작업일'].max(), freq='B')
        all_worker_days = pd.MultiIndex.from_product([all_workers['작업자'], date_range], names=['작업자', '작업일'])
        all_worker_days = pd.DataFrame(index=all_worker_days).reset_index().merge(all_workers, on='작업자')

        # ✅ 작업자 두 명 이상인 경우 분리
        df_nul = df.copy()
        df_nul['작업자'] = df_nul['작업자'].astype(str).str.replace('.', ',', regex=False).str.split(',')
        df_nul = df_nul.explode('작업자')
        df_nul['작업자'] = df_nul['작업자'].str.strip()

        # ✅ "개인별 업무일지 누락 현황" 항목만 작업자 2명을 분리
        df_nul = df.copy()
        df_nul['작업자'] = df_nul['작업자'].astype(str)
        df_nul['작업자'] = df_nul['작업자'].str.replace('.', ',', regex=False)
        df_nul['작업자'] = df_nul['작업자'].str.replace(' ', ',', regex=False)
        df_nul['작업자'] = df_nul['작업자'].str.split(',')
        df_nul = df_nul.explode('작업자')
        df_nul['작업자'] = df_nul['작업자'].str.strip()

        actual_logs = df_nul.groupby(['팀', '작업자', '작업일']).size()
        log_df = all_worker_days.merge(
            actual_logs.rename('작성여부').reset_index(),
            on=['팀', '작업자', '작업일'],
            how='left'
        ).fillna({'작성여부': 0})

        st.markdown("## 👷‍ 개인별 누락 현황")
        personal_summary = log_df.groupby(['팀', '작업자'])['작성여부'].agg(['mean', 'count']).reset_index()
        personal_summary = personal_summary[personal_summary['mean'] < 1.0].copy()
        personal_summary['누락일수'] = (1 - personal_summary['mean']) * personal_summary['count']
        personal_summary['누락률(%)'] = (1 - personal_summary['mean']) * 100

        personal_summary = personal_summary.sort_values('누락일수', ascending=False).head(30)
        personal_summary.reset_index(drop=True, inplace=True)
        styled_df = personal_summary[['팀', '작업자', '누락일수', '누락률(%)']]
        styled_df['누락일수'] = styled_df['누락일수'].astype(int)
        styled_df['누락률(%)'] = styled_df['누락률(%)'].astype(int)

        # ✔ TOP 5 누락자 카드 표시
        st.markdown("### ⚠️ TOP 5 누락자")
        top5 = styled_df.head(5)
        cols = st.columns(5)
        for i, row in top5.iterrows():
            cols[i].markdown(f"""
            <div style='background-color:#fff3f3; padding:12px; border-radius:12px; box-shadow:0 2px 8px #ddd;'>
                <div style='font-size:30px;'>👷️</div>
                <div style='font-weight:bold;'>{row['작업자']}</div>
                <div style='font-size:13px; color:#555;'>{row['팀']}</div>
                <div style='margin-top:6px;'>❗누락일수: {row['누락일수']}<br>🗓️ 누락률: {row['누락률(%)']}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ✔ 전체 개인 누락률 테이블
        def bar(row):
            pct = row['누락률(%)']
            # 대비 강화: 낮은 구간은 더 연하게, 높은 구간은 훨씬 진하게
            norm = (pct / 100) ** 2
            color = plt.cm.Reds(norm)
            hex_color = '#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3])
            bars = int(pct // 5) * "█"
            return f'<span style="color:{hex_color}">{bars} {pct}%</span>'

        styled_df['누락률 시각화'] = styled_df.apply(bar, axis=1)
        # 스크롤 가능한 컨테이너에 테이블 표시 (최대 10행 분량 높이)
        # 테이블에 번호(Index) 컬럼 추가
        numbered_df = styled_df[['팀', '작업자', '누락일수', '누락률 시각화']].reset_index().rename(columns={'index':'번호'})
        html_table = numbered_df.to_html(escape=False, index=False)
        # 테이블을 컨테이너 너비에 맞추기 위해 스타일 적용
        html_table = html_table.replace('<table', '<table style="width:100%"')
        # 헤더 중앙 정렬 적용
        html_table = html_table.replace('<th>', '<th style="text-align:center;">')
        st.markdown(
            f"<div style='max-height:300px; overflow-y:auto; width:100%'>{html_table}</div>",
            unsafe_allow_html=True
        )
        



        # ✅ 중복 출동 현황
        st.markdown("## 🔁 중복 출동 현황")
        dup_equipment = df[
            (df['업무종류'] == '무선') &
            (df['구분'] == '장애/알람(AS)') &
            df['장비ID'].notna() &
            (df['장비ID'].astype(str).str.strip() != '') &
            df['장비명'].notna() &
            (df['장비명'].astype(str).str.strip() != '') &
            (~df['장비명'].astype(str).str.contains('민원')) &
            (~df['장비명'].astype(str).str.contains('사무'))
        ]

        duplicated_ids = dup_equipment['장비ID'].value_counts()
        duplicated_ids = duplicated_ids[duplicated_ids >= 3].index
        dup_equipment = dup_equipment[dup_equipment['장비ID'].isin(duplicated_ids)]

        grouped = dup_equipment.groupby(['팀', '장비명', '장비ID', '작업자']).size().reset_index(name='건수')
        grouped['작업자'] = grouped['작업자'] + '(' + grouped['건수'].astype(str) + ')'
        grouped.rename(columns={'작업자': '작업자(출동 횟수)'}, inplace=True)
        grouped = grouped.sort_values(by=['팀', '장비명', '장비ID', '건수'], ascending=[True, True, True, False])
        combined = grouped.groupby(['팀', '장비명', '장비ID'])['작업자(출동 횟수)'].apply(lambda x: ', '.join(x)).reset_index()

        중복건수_df = dup_equipment.drop_duplicates(subset=['팀', '장비명', '장비ID', '시작일시', '종료일시'])
        중복건수_df = 중복건수_df.groupby(['팀', '장비명', '장비ID']).size().reset_index(name='중복건수')

        combined = combined.merge(중복건수_df, on=['팀', '장비명', '장비ID'], how='left')
        combined = combined[combined['중복건수'] >= 3]
        dup_equipment_sorted = combined.sort_values(by='중복건수', ascending=False).reset_index(drop=True)
        st.dataframe(dup_equipment_sorted.rename(columns={'팀': '운용팀'}), use_container_width=True)

        

        
        
        st.markdown("## 🗓️ 운용팀 일별 작성현황")
        daily_count = df_original.groupby([pd.to_datetime(df_original['시작일시']).dt.date, df_original['팀']]).size().unstack(fill_value=0).astype(int)
        daily_count.loc['합계'] = daily_count.sum()
        st.dataframe(daily_count, use_container_width=True)

        

        st.markdown("## 📉 팀 주차별 가동율 (이동시간 제외)")

        # 이동업무 제외
        df_move_filtered = df[df['업무종류'] != '이동업무'].copy()

        # ✅ 작업자 분리 (쉼표, 마침표, 공백 기준)
        df_move_filtered['작업자'] = df_move_filtered['작업자'].astype(str)
        df_move_filtered['작업자'] = df_move_filtered['작업자'].str.replace('.', ',', regex=False)
        df_move_filtered['작업자'] = df_move_filtered['작업자'].str.replace(' ', ',', regex=False)
        df_move_filtered['작업자'] = df_move_filtered['작업자'].str.split(',')
        df_move_filtered = df_move_filtered.explode('작업자')
        df_move_filtered['작업자'] = df_move_filtered['작업자'].str.strip()

        # ✅ 주차와 작업일 생성
        df_move_filtered['작업일'] = df_move_filtered['시작일시'].dt.date
        df_move_filtered['주차'] = df_move_filtered['시작일시'].apply(lambda x: f"{x.month}월{x.day // 7 + 1}주")

        # ✅ 1일 1인당 상한 제한 (600분) 정제 작업시간만 활용
        capped = df_move_filtered.groupby(['작업일', '작업자', '팀', '주차'])['작업시간(분)'].sum().clip(upper=600).reset_index()

        # ✅ 팀-주차별 작업시간 및 가동율 계산
        df_team_time = capped.groupby(['팀', '주차'])['작업시간(분)'].sum().reset_index(name='팀작업시간_분')
        unique_worker_count = capped.groupby(['팀', '주차'])['작업자'].nunique().reset_index(name='작업자수')
        df_weekly = df_team_time.merge(unique_worker_count, on=['팀', '주차'])
        df_weekly['기준시간'] = df_weekly['작업자수'] * 2400
        df_weekly['가동율(%)'] = (df_weekly['팀작업시간_분'] / df_weekly['기준시간']).clip(upper=1.0)

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

        # ✅ capped 데이터 기반 재계산
        daily_sum = capped.groupby(['작업일', '팀'])['작업시간(분)'].sum().reset_index()
        daily_worker_count = capped.groupby(['작업일', '팀'])['작업자'].nunique().reset_index(name='작업자수')
        daily_avg = daily_sum.merge(daily_worker_count, on=['작업일', '팀'])
        daily_avg['평균작업시간(시간)'] = daily_avg['작업시간(분)'] / daily_avg['작업자수'] / 60

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

        st.markdown("## 👷‍👷‍ 팀별 운용조 현황")
        crew_base = df.groupby(['팀', '원본작업자']).first().reset_index()
        crew_base['조구성'] = crew_base['원본작업자'].apply(lambda x: '2인 1조' if len(split_workers(x)) >= 2 else '1인 1조')
        crew_summary = crew_base.groupby(['팀', '조구성']).size().unstack(fill_value=0)
        crew_summary_percent = crew_summary.div(crew_summary.sum(axis=1), axis=0).fillna(0).round(4) * 100
        st.dataframe(crew_summary_percent.T.style.format("{:.2f}%"), use_container_width=True)

        # 조별 구성 막대 그래프
        crew_summary_reset = crew_summary_percent.reset_index().melt(id_vars='팀', var_name='조구성', value_name='비율')
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
        st.markdown("## 👷‍👷‍ 업무구분별 인원조 현황")
        df_taskcrew = df_original.copy()
        df_taskcrew['작업자목록'] = df_taskcrew['작업자'].apply(split_workers)
        df_taskcrew['조구성'] = df_taskcrew['작업자목록'].apply(lambda x: '2인 1조' if len(x) >= 2 else '1인 1조')
        df_taskcrew = df_taskcrew.explode('작업자목록')
        df_taskcrew['작업자목록'] = df_taskcrew['작업자목록'].str.strip()

        crew_task = df_taskcrew[['구분', '조구성']].copy()
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

    































