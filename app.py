import streamlit as st
import pandas as pd
import io
import streamlit.components.v1 as components
import plotly.express as px

def process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    df['시작일시'] = pd.to_datetime(df['시작일시'])
    df['종료일시'] = pd.to_datetime(df['종료일시'])
    df['작업시간(분)'] = (df['종료일시'] - df['시작일시']).dt.total_seconds() / 60
    df['작업자'] = df['작업자'].str.split(',')
    df = df.explode('작업자')
    df['작업자'] = df['작업자'].str.strip()

    return df

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def main():
    st.set_page_config(page_title="홍성운용팀 업무 대시보드", layout="wide")
    st.markdown("""
        <style>
        @media screen and (max-width: 768px) {
            .block-container {
                padding: 1rem !important;
            }
            .css-18e3th9 {
                padding: 1rem !important;
            }
            .css-1d391kg {
                width: 100% !important;
            }
        }
        .metric-label {
            font-weight: bold;
            color: #4B8BBE;
        }
        .stDataFrame th, .stDataFrame td {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📊 홍성운용팀 팀별 업무 대시보드")
    st.markdown("업무일지를 업로드하고, 팀과 팀원별로 분석 결과를 확인하세요.")

    uploaded_file = st.file_uploader("📁 업무일지 CSV 파일 업로드", type=["csv"])

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
            selected_team = st.sidebar.selectbox("팀 선택", team_options)
            df = df[df['팀'] == selected_team]

            # ✅ 작업자 선택: expander 안에서 이름 보이게
            member_options = df['작업자'].dropna().unique().tolist()
            with st.sidebar.expander("**작업자 선택**", expanded=False):
                selected_members = st.multiselect(
                    label="작업자 목록",
                    options=member_options,
                    default=member_options,
                    format_func=lambda x: x  # 이름 그대로 표시
                )
            df = df[df['작업자'].isin(selected_members)]

        st.markdown("### 📌 전체 요약")
        col1, col2, col3 = st.columns(3)
        total_time = df['작업시간(분)'].sum()
        total_tasks = len(df)
        office_time = df[df['구분'] == '사무업무']['작업시간(분)'].sum()
        col1.metric("총 작업 시간 (분)", f"{int(total_time):,}")
        col2.metric("총 작업 건수", f"{total_tasks:,}")
        col3.metric("사무업무 시간 (분)", f"{int(office_time):,}")

        st.markdown("### 📋 업무종류별 작업 시간 및 건수")
        task_by_type = df.groupby('업무종류').agg(
            총작업시간_분=('작업시간(분)', 'sum'),
            작업건수=('작업시간(분)', 'count'),
            평균작업시간_분=('작업시간(분)', 'mean')
        ).reset_index().sort_values(by='총작업시간_분', ascending=False)
        st.dataframe(task_by_type.style.format({
            '총작업시간_분': '{:,.1f}',
            '작업건수': '{:,}',
            '평균작업시간_분': '{:,.1f}'
        }), use_container_width=True)

        st.download_button(
            label="📥 업무종류별 요약 CSV 다운로드",
            data=convert_df_to_csv(task_by_type),
            file_name="업무종류별_요약.csv",
            mime="text/csv"
        )

        fig_task_type = px.bar(
            task_by_type,
            x='업무종류',
            y='총작업시간_분',
            text='총작업시간_분',
            title='업무종류별 총 작업시간',
            labels={'총작업시간_분': '총 작업시간 (분)', '업무종류': '업무 종류'}
        )
        fig_task_type.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_task_type.update_layout(xaxis_tickangle=0)
        st.plotly_chart(fig_task_type, use_container_width=True)

        st.markdown("### 👤 개인별 작업 요약")
        sort_option = st.selectbox("정렬 기준 선택", ['총작업시간_분', '작업건수', '평균작업시간_분'])
        individual_summary = df.groupby('작업자').agg(
            총작업시간_분=('작업시간(분)', 'sum'),
            작업건수=('작업시간(분)', 'count'),
            평균작업시간_분=('작업시간(분)', 'mean')
        ).reset_index().sort_values(by=sort_option, ascending=False)

        st.dataframe(individual_summary.style.format({
            '총작업시간_분': '{:,.1f}',
            '작업건수': '{:,}',
            '평균작업시간_분': '{:,.1f}'
        }), use_container_width=True)

        st.download_button(
            label="📥 개인별 작업 요약 CSV 다운로드",
            data=convert_df_to_csv(individual_summary),
            file_name="개인별_작업요약.csv",
            mime="text/csv"
        )

        st.markdown("### 📈 <b>작업시간 비교 (Top 10)</b>", unsafe_allow_html=True)
        top10 = individual_summary.head(10)
        fig = px.bar(
            top10,
            x='작업자',
            y='총작업시간_분',
            title='Top 10 작업시간 비교',
            labels={'작업자': '작업자', '총작업시간_분': '총 작업시간 (분)'},
            text='총작업시간_분',
            orientation='v'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(xaxis_title='<b>작업자</b>', yaxis_title='작업시간(분)')
        st.plotly_chart(fig, use_container_width=True)

        if '팀' in df.columns:
            st.markdown("### 🏢 팀별 작업시간 비교 - 홍성운용팀")
            team_summary = df.groupby('팀').agg(총작업시간_분=('작업시간(분)', 'sum')).reset_index()
            fig_team = px.bar(
                team_summary,
                x='팀',
                y='총작업시간_분',
                text='총작업시간_분',
                title='홍성운용팀 팀별 작업시간 비교',
                labels={'총작업시간_분': '총 작업시간 (분)'},
                orientation='v'
            )
            fig_team.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_team.update_layout(xaxis_title='팀', yaxis_title='작업시간(분)')
            st.plotly_chart(fig_team, use_container_width=True)

    else:
        st.info("업무일지 CSV 파일을 업로드하면 대시보드가 표시됩니다. 예시 파일을 사용하려면 관리자에게 문의하세요.")

if __name__ == "__main__":
    main()
