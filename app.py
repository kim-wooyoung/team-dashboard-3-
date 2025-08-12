import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import re
import streamlit.components.v1 as components

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


# ✅ HSL → HEX 변환 유틸 (matplotlib 제거용)
def hsl_to_hex(h, s, l):
    c = (1 - abs(2*l - 1)) * s
    x = c * (1 - abs((h/60) % 2 - 1))
    m = l - c/2
    if   0 <= h < 60:   r1, g1, b1 = c, x, 0
    elif 60 <= h < 120: r1, g1, b1 = x, c, 0
    elif 120 <= h < 180:r1, g1, b1 = 0, c, x
    elif 180 <= h < 240:r1, g1, b1 = 0, x, c
    elif 240 <= h < 300:r1, g1, b1 = x, 0, c
    else:               r1, g1, b1 = c, 0, x
    r, g, b = int((r1 + m) * 255), int((g1 + m) * 255), int((b1 + m) * 255)
    return f'#{r:02x}{g:02x}{b:02x}'


def main():
    col1, col2 = st.columns([8, 1])
    with col1:
        st.markdown("""
<h1 style='font-size: 50px;'>📊  <span style='color:#d32f2f;'>MOS</span>tagram 분석 대시보드</h1>
""", unsafe_allow_html=True)
    with col2:
        logo_base64 = st.session_state.get('logo_base64', "")
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
        # ✅ 컬럼 존재 검증(업로드 직후)
        cols_df = pd.read_csv(uploaded_file, nrows=0)
        required_cols = ['팀','작업자','시작일시','종료일시','업무종류','구분','장비ID','장비명']
        missing = [c for c in required_cols if c not in cols_df.columns]
        if missing:
            st.error(f"필수 컬럼 누락: {missing} — CSV 컬럼명을 확인해주세요.")
            st.stop()
        uploaded_file.seek(0)

        # ✅ 데이터 가공
        df, _ = process_data(uploaded_file)

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

            member_options = (
                df['작업자']
                  .astype(str).str.strip()
                  .replace({'nan': ''})
                  .replace('', np.nan)
                  .dropna()
                  .unique().tolist()
            )
            with st.sidebar.expander("**작업자 선택**", expanded=False):
                selected_members = st.multiselect(
                    "작업자 목록",
                    options=member_options,
                    default=member_options
                )
            df = df[df['작업자'].isin(selected_members)]
        # 데이터 없음 방지
        if df.empty:
            st.warning("선택한 조건에 해당하는 데이터가 없습니다. 필터를 조정해 주세요.")
            st.stop()

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

        # ─────────────────────────────────────────────────────────
        # ✅ 중복 출동 현황  (먼저 표시)
        st.markdown("## 🔁 중복 출동 현황")
        dup_equipment = df[
            (df['업무종류'] == '무선') &
            (df['구분'] == '장애/알람(AS)') &
            df['장비ID'].notna() &
            (df['장비ID'].astype(str).str.strip() != '') &
            df['장비명'].notna() &
            (df['장비명'].astype(str).str.strip() != '') &
            (~df['장비명'].astype(str).str.contains('민원', regex=False)) &
            (~df['장비명'].astype(str).str.contains('사무', regex=False))
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

        # ─────────────────────────────────────────────────────────
        # 👤 개인별 누락 현황  (다음 표시)
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

        # ✅ "개인별 업무일지 누락 현황" 계산용: 작업자 2명 이상 분리 (중복 제거 버전)
        df_nul = df.copy()
        df_nul['작업자'] = df_nul['작업자'].astype(str).str.replace('.', ',', regex=False).str.replace(' ', ',', regex=False)
        df_nul['작업자'] = df_nul['작업자'].str.split(',')
        df_nul = df_nul.explode('작업자')
        df_nul['작업자'] = df_nul['작업자'].str.strip()

        actual_logs = df_nul.groupby(['팀', '작업자', '작업일']).size()
        log_df = all_worker_days.merge(
            actual_logs.rename('작성여부').reset_index(),
            on=['팀', '작업자', '작업일'],
            how='left'
        ).fillna({'작성여부': 0})

        # ✅ 하루에 여러 건이 있어도 작성여부는 1로 처리(누락률 왜곡 방지)
        log_df['작성여부'] = (log_df['작성여부'] > 0).astype(int)

        st.markdown("## 📋 개인별 누락 현황")
        personal_summary = log_df.groupby(['팀', '작업자'])['작성여부'].agg(['mean', 'count']).reset_index()
        personal_summary = personal_summary[personal_summary['mean'] < 1.0].copy()
        personal_summary['누락일수'] = (1 - personal_summary['mean']) * personal_summary['count']
        personal_summary['누락률(%)'] = (1 - personal_summary['mean']) * 100

        personal_summary = personal_summary.sort_values('누락일수', ascending=False).head(30)
        personal_summary.reset_index(drop=True, inplace=True)
        styled_df = personal_summary[['팀', '작업자', '누락일수', '누락률(%)']]
        styled_df['누락일수'] = styled_df['누락일수'].astype(int)
        styled_df['누락률(%)'] = styled_df['누락률(%)'].astype(int)

        # ✔ 개인별 누락 현황 — 표 형식(중복 출동 현황과 동일 스타일)
        table_df = (
            styled_df
            .sort_values(by=['누락일수', '누락률(%)'], ascending=[False, False])
            .reset_index(drop=True)
            .rename(columns={'팀': '운용팀'})
        )
        st.dataframe(table_df, use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # 🕒 구분별 MTTR / 반복도 (이 섹션만 이동업무 제외)
        st.markdown("## 🕒 구분별 MTTR / 반복도")

        _mttr_keys = ['팀', '원본작업자', '시작일시', '종료일시', '구분', '장비ID']
        mttr_df = df.drop_duplicates(subset=_mttr_keys).copy()

        # 장비ID 정리 및 음수 작업시간 제거
        mttr_df['장비ID'] = mttr_df['장비ID'].astype(str).str.strip()
        mttr_df = mttr_df[mttr_df['작업시간(분)'] >= 0]

        # ✅ '사무업무' 제외
        mttr_df = mttr_df[mttr_df['구분'].astype(str).str.strip() != '사무업무']

        # ✅ '이동업무' 제외 (요청 섹션 한정)
        mttr_df = mttr_df[mttr_df['업무종류'].astype(str).str.strip() != '이동업무']

        if mttr_df.empty:
            st.info("분석 가능한 데이터가 없습니다. (필터 조건을 확인해 주세요)")
        else:
            def _p90(x):
                try:
                    return float(np.percentile(x, 90))
                except Exception:
                    return float(np.nan)

            mttr_tbl = (
                mttr_df
                .groupby('구분', dropna=False)['작업시간(분)']
                .agg(건수='count', MTTR_분='mean', 중앙값_분='median', P90_분=_p90)
                .reset_index()
            )

            rep_src = (
                mttr_df
                .loc[mttr_df['장비ID'].notna() & (mttr_df['장비ID'] != '')]
                .groupby(['구분', '장비ID']).size().reset_index(name='건수')
            )
            rep_sum = rep_src.groupby('구분')['장비ID'].nunique().reset_index(name='고유장비수')
            rep_cnt = rep_src[rep_src['건수'] >= 2].groupby('구분')['장비ID'].nunique().reset_index(name='재발장비수')
            rep_tbl = rep_sum.merge(rep_cnt, on='구분', how='left').fillna({'재발장비수': 0})
            rep_tbl['중복업무 비율(%)'] = (
                (rep_tbl['재발장비수'] / rep_tbl['고유장비수']).replace([np.inf, np.nan], 0) * 100
            )

            result = mttr_tbl.merge(rep_tbl, on='구분', how='left')

            # ✅ 정수 표기(분/개수/비율)
            result['MTTR(분)'] = result['MTTR_분'].round().astype('Int64')
            result['중앙값(분)'] = result['중앙값_분'].round().astype('Int64')
            result['P90(분)'] = result['P90_분'].round().astype('Int64')
            result['고유업무수'] = result['고유장비수'].astype('Int64')
            result['중복업무 수'] = result['재발장비수'].astype('Int64')
            result['중복업무 비율(%)'] = result['중복업무 비율(%)'].round().astype('Int64')

            display_cols = ['구분', '건수', 'MTTR(분)', '중앙값(분)', 'P90(분)', '고유업무수', '중복업무 수', '중복업무 비율(%)']
            result_display = (
                result[display_cols]
                .sort_values(['MTTR(분)'], ascending=[True])
                .reset_index(drop=True)
            )

            fmt_all = {col: '{:.0f}' for col in result_display.columns if pd.api.types.is_numeric_dtype(result_display[col])}
            st.dataframe(result_display.style.format(fmt_all), use_container_width=True)

            result_display['MTTR_label'] = result_display['MTTR(분)'].astype('Int64').astype(str)
            # ✅ 축 상한을 동적으로 맞춰 두 그래프 막대 높이가 비슷하게 보이도록 조정
            try:
                _mttr_max = float(result_display['MTTR(분)'].max())
                _dup_max = float(result_display['중복업무 비율(%)'].max())
            except Exception:
                _mttr_max, _dup_max = 0.0, 0.0
            _mttr_ylim = max(10.0, _mttr_max * 1.1) if _mttr_max > 0 else 10.0
            _dup_ylim = min(100.0, max(10.0, _dup_max * 1.1)) if _dup_max > 0 else 10.0

            # ✅ 구분별 색상 고정 매핑 (두 그래프 공통)
            cats = sorted(result_display['구분'].astype(str).unique())
            palette = px.colors.qualitative.Set2
            color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}

            fig_mttr = px.bar(
                result_display,
                x='구분',
                y='MTTR(분)',
                color='구분',
                color_discrete_map=color_map,
                title='구분별 MTTR(분)',
                labels={'MTTR(분)': 'MTTR(분)', '구분': '구분'},
                text='MTTR_label',
                custom_data=['MTTR_label']
            )
            fig_mttr.update_traces(textposition='outside', hovertemplate='구분: %{x}<br>MTTR(분): %{customdata[0]}<extra></extra>')
            fig_mttr.update_layout(legend_title_text='구분', margin=dict(t=60, b=0), height=420, bargap=0.2, yaxis_range=[0, _mttr_ylim])
            # (2열 배치로 이동)
            pass

            # ▼ 구분별 중복업무 비율(%) 그래프 — MTTR 그래프 바로 아래 동일 형식으로 표시
            result_display['dup_label'] = result_display['중복업무 비율(%)'].fillna(0).astype('Int64').astype(str)
            fig_dup_ratio = px.bar(
                result_display,
                x='구분',
                y='중복업무 비율(%)',
                color='구분',
                color_discrete_map=color_map,
                title='구분별 중복업무 비율(%)',
                labels={'중복업무 비율(%)': '중복업무 비율(%)', '구분': '구분'},
                text='dup_label',
                custom_data=['dup_label']
            )
            fig_dup_ratio.update_traces(
                textposition='outside',
                hovertemplate='구분: %{x}<br>중복업무 비율: %{customdata[0]}%<extra></extra>'
            )
            fig_dup_ratio.update_layout(legend_title_text='구분', yaxis_range=[0, _dup_ylim], margin=dict(t=60, b=0), height=420, bargap=0.2)
            cols_mttr = st.columns(2)
            with cols_mttr[0]:
                st.plotly_chart(fig_mttr, use_container_width=True)
            with cols_mttr[1]:
                st.plotly_chart(fig_dup_ratio, use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # 🗓️ 운용팀 일별 작성현황 (이동업무 포함)
        st.markdown("## 🗓️ 운용팀 일별 작성현황")
        daily_count = df.groupby([pd.to_datetime(df['시작일시']).dt.date, df['팀']]).size().unstack(fill_value=0).astype(int)
        daily_count.loc['합계'] = daily_count.sum()
        st.dataframe(daily_count, use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # 📉 팀 주차별 가동율 (이동업무 포함)
        st.markdown("## 📉 팀 주차별 가동율")

        # ✅ 1일 1인당 상한 제한 (600분)
        capped = df.groupby(['작업일', '작업자', '팀', '주차'])['작업시간(분)'].sum().clip(upper=600).reset_index()

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
            title='팀 주차별 가동율',
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

        # ─────────────────────────────────────────────────────────
        # 📊 일별 평균 작업시간 (이동업무 포함)
        st.markdown("## 📊 일별 평균 작업시간")

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
            title='일별 평균 작업시간',
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

        # ─────────────────────────────────────────────────────────
        # ‍ 👷‍팀별 운용조 현황
        st.markdown("## 👷 팀별 운용조 현황")
        crew_base = df.groupby(['팀', '원본작업자']).first().reset_index()
        crew_base['조구성'] = crew_base['원본작업자'].apply(lambda x: '2인 1조' if len(split_workers(x)) >= 2 else '1인 1조')
        crew_summary = crew_base.groupby(['팀', '조구성']).size().unstack(fill_value=0)
        crew_summary_percent = crew_summary.div(crew_summary.sum(axis=1), axis=0).fillna(0).round(4) * 100

        st.dataframe(
            crew_summary_percent.T.style.format("{:.2f}%"),
            use_container_width=True
        )

        crew_summary_reset = crew_summary_percent.reset_index().melt(id_vars='팀', var_name='조구성', value_name='비율')
        fig_crew = px.bar(
            crew_summary_reset,
            x='팀',
            y='비율',
            color='조구성',
            color_discrete_map={'1인 1조': '#1f77b4', '2인 1조': '#ff7f0e'},
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

        # ─────────────────────────────────────────────────────────
        # ✅ 업무구분별 인원조 현황
        df_taskcrew = df.drop_duplicates(
            subset=['팀', '원본작업자', '시작일시', '종료일시', '구분']
        ).copy()
        df_taskcrew['작업자목록'] = df_taskcrew['원본작업자'].apply(split_workers)
        df_taskcrew['조구성'] = df_taskcrew['작업자목록'].apply(
            lambda x: '2인 1조' if len(x) >= 2 else '1인 1조'
        )
        # 👍 여기서는 explode 불필요 — 작업 1건당 조구성 1개만 반영

        crew_task = df_taskcrew[['구분', '조구성']].copy()
        crew_task_grouped = crew_task.groupby(['구분', '조구성']).size().unstack(fill_value=0)
        crew_task_ratio = crew_task_grouped.div(crew_task_grouped.sum(axis=1), axis=0).fillna(0).round(4) * 100

        crew_task_reset = crew_task_ratio.reset_index().melt(id_vars='구분', var_name='조구성', value_name='비율')
        fig_crew_task = px.bar(
            crew_task_reset,
            x='구분',
            y='비율',
            color='조구성',
            color_discrete_map={'1인 1조': '#1f77b4', '2인 1조': '#ff7f0e'},
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
