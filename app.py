import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import re
from io import BytesIO, StringIO
import zipfile

# =========================
# PATCH PACK: 공통 설정·유틸 (자동삽입)
# =========================
import numpy as np
import pandas as pd
from io import BytesIO
import zipfile

# === 1) 설정 상수(단일 출처) ===
BASIS_MIN_PER_DAY = 480               # 1일 기준 480분
ABNORMAL_TASK_MIN = 600               # 단일 기록 10시간 초과는 분석에서 제외(진단용 컷)
DEFAULT_ERROR_THRESHOLD_MIN = 480     # 작업시간 오류 임계(기본 8시간, 사이드바에서 변경)
DISPLAY_DT_FMT = '%Y-%m-%d %H:%M'     # 표시용 날짜 포맷

# === 2) 문자열 표준화 + 범주형 캐싱 ===
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['팀', '작업자', '구분', '장비ID', '장비명']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if '구분' in df.columns:
        df['구분'] = df['구분'].replace({
            '장애/알람(AS)': '장애/알람',
            '사무업무 ': '사무업무',
        })
    for col in ['시작일시','종료일시']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    for c in ['팀', '구분', '작업자']:
        if c in df.columns:
            df[c] = df[c].astype('category')
    return df

# === 3) 표시용 날짜 포맷 일원화 ===
def format_dt_display(df: pd.DataFrame) -> pd.DataFrame:
    disp = df.copy()
    for col in ('시작일시','종료일시'):
        if col in disp.columns:
            disp[col] = pd.to_datetime(disp[col], errors='coerce').dt.strftime(DISPLAY_DT_FMT)
    return disp

# === 4) 겹침 병합(Union) — 안전 루프 버전 ===
def merge_union_minutes(group: pd.DataFrame):
    gg = group[['시작일시','종료일시']].copy()
    gg['시작일시'] = pd.to_datetime(gg['시작일시'], errors='coerce')
    gg['종료일시'] = pd.to_datetime(gg['종료일시'], errors='coerce')
    gg = gg.dropna().sort_values('시작일시')
    if gg.empty:
        return pd.Series({'병합작업시간(분)': 0.0})
    total = pd.Timedelta(0)
    cur_s = gg.iloc[0]['시작일시']
    cur_e = gg.iloc[0]['종료일시']
    for _, row in gg.iloc[1:].iterrows():
        s, e = row['시작일시'], row['종료일시']
        if s <= cur_e:
            if e > cur_e: cur_e = e
        else:
            total += (cur_e - cur_s)
            cur_s, cur_e = s, e
    total += (cur_e - cur_s)
    return pd.Series({'병합작업시간(분)': total.total_seconds()/60})

# === 4-ALT) 겹침 병합 — 벡터화(대용량 최적화) ===
def merge_union_minutes_fast(group: pd.DataFrame):
    gg = group[['시작일시','종료일시']].copy()
    gg['시작일시'] = pd.to_datetime(gg['시작일시'], errors='coerce')
    gg['종료일시'] = pd.to_datetime(gg['종료일시'], errors='coerce')
    gg = gg.dropna().sort_values('시작일시')
    if gg.empty:
        return pd.Series({'병합작업시간(분)': 0.0})
    s = gg['시작일시'].values.astype('datetime64[ns]').astype('int64')
    e = gg['종료일시' ].values.astype('datetime64[ns]').astype('int64')
    e_cummax = np.maximum.accumulate(e)
    is_new = np.empty(len(s), dtype=bool)
    is_new[0] = True
    is_new[1:] = s[1:] > e_cummax[:-1]
    grp = np.cumsum(is_new) - 1
    minutes = (pd.Series(e).groupby(grp).max() - pd.Series(s).groupby(grp).min()).sum() / 1e9 / 60.0
    return pd.Series({'병합작업시간(분)': float(minutes)})

# === 5) 누락현황 기초(중간 산출물) — 캐시 ===
try:
    import streamlit as st
except Exception:
    class _Dummy: 
        def __getattr__(self, k): 
            def _f(*a, **kw): 
                return None
            return _f
    st = _Dummy()

@st.cache_data(show_spinner=False)
def build_presence_table(df: pd.DataFrame) -> pd.DataFrame:
    src = df.copy()
    if '작업일' not in src.columns:
        src['작업일'] = pd.to_datetime(src['시작일시'], errors='coerce').dt.date
    workers = src[['팀','작업자']].dropna().drop_duplicates()
    date_range = pd.date_range(start=pd.to_datetime(src['작업일']).min(),
                               end  =pd.to_datetime(src['작업일']).max(),
                               freq='B').date
    idx = pd.MultiIndex.from_product([workers['작업자'], date_range], names=['작업자','작업일'])
    all_rows = pd.DataFrame(index=idx).reset_index().merge(workers, on='작업자', how='left')
    actual = src.groupby(['팀','작업자','작업일']).size().rename('작성여부').reset_index()
    actual['작성여부'] = 1
    pres = all_rows.merge(actual, on=['팀','작업자','작업일'], how='left').fillna({'작성여부':0})
    return pres

# === 6) 다운로드(엑셀 엔진 폴백 + 로깅) ===
def df_to_download_bytes(df: pd.DataFrame, sheet_name: str = 'Sheet1'):
    buf = BytesIO()
    try:
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        buf.seek(0)
        return buf.getvalue(), 'xlsxwriter'
    except Exception as e1:
        try:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
            buf.seek(0)
            return buf.getvalue(), 'openpyxl'
        except Exception as e2:
            st.info(f"엑셀 엔진 사용 실패로 CSV(zip)으로 대체합니다. 원인: {type(e1).__name__} / {type(e2).__name__}")
            zbuf = BytesIO()
            with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('data.csv', df.to_csv(index=False, encoding='utf-8-sig'))
            zbuf.seek(0)
            return zbuf.getvalue(), 'zip'

# === 7) 사이드바·안내 배너(환경 의존 대안) ===
def ensure_sidebar_and_captions():
    try:
        with st.sidebar:
            st.markdown("### ⚙️ 분석 옵션")
            if 'error_threshold_min' not in st.session_state:
                st.session_state['error_threshold_min'] = DEFAULT_ERROR_THRESHOLD_MIN
            new_v = st.number_input(
                "작업시간 오류 임계값(분)",
                min_value=60, max_value=1440,
                value=int(st.session_state.get('error_threshold_min', DEFAULT_ERROR_THRESHOLD_MIN)), step=10,
                help="품질 점검(오류 탐지)용 임계값입니다. 분석 제외 기준(10시간 컷)과 다릅니다."
            )
            st.session_state['error_threshold_min'] = int(new_v)
        st.info("왼쪽 사이드바에서 기준/임계값 조정이 가능합니다.")
    except Exception:
        pass

# (자동 호출 시 메인 코드 위에 있어도 무해)
# (패치) ensure_sidebar_and_captions()는 set_page_config 이후에 호출됩니다.
# 공통 문자열 처리 유틸(중복 제거용)
def sstr(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

# ⚙️ Page config — MUST be the first Streamlit command
st.set_page_config(page_title="업무일지 분석 대시보드", layout="wide", initial_sidebar_state="collapsed")


# (패치) set_page_config 직후에 사이드바/캡션 표시
try:
    ensure_sidebar_and_captions()
except Exception:
    pass
# ✅ Query params helpers (replace deprecated experimental_* with st.query_params)
#    - 읽기: st.query_params.get("key", default)
#    - 쓰기: st.query_params["key"] = "value"  (문자열)

def ensure_sidebar_open_once():
    """URL 쿼리파라미터 sb=1을 설정하고 1회 rerun하여, 이후 새로고침 시 '초기 펼침'이 적용되도록 유도.
    구버전 Streamlit에서는 set_page_config를 두 번 호출할 수 없으므로, 안전하게 파라미터만 설정합니다.
    """
    try:
        if st.query_params.get("sb", "0") != "1":
            st.query_params["sb"] = "1"
            st.rerun()
    except Exception:
        pass

# (선택) 최신 버전에서만 동작: 쿼리파라미터로 확장 요청 시 시도
try:
    if st.query_params.get("sb", "0") == "1":
        pass  # auto-open sidebar handled via ensure_sidebar_open_once + rerun
except Exception:
    pass

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
    worker_string = re.sub(r'[.,/;·\s／]+', ',', str(worker_string))  # 쉼표, 마침표, 공백 → 쉼표
    worker_string = re.sub(r'(?<=[가-힣]{2})(?=[가-힣]{2})', ',', worker_string)  # 붙여쓰기된 한글 이름 분리
    return [name.strip() for name in worker_string.split(',') if name.strip()]


def process_data(uploaded_file, dayfirst=False):
    # CSV 읽기: 날짜 컬럼 즉시 파싱 + 인코딩 폴백
    try:
        df_original = pd.read_csv(
            uploaded_file,
            parse_dates=['시작일시','종료일시'],
            dayfirst=dayfirst
        )
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df_original = pd.read_csv(
            uploaded_file,
            parse_dates=['시작일시','종료일시'],
            dayfirst=dayfirst,
            encoding='cp949'
        )

    # 데이터 가공
    df = df_original.copy()
    df['원본작업자'] = df['작업자']
    df['작업시간(분)'] = (df['종료일시'] - df['시작일시']).dt.total_seconds() / 60
    # ✅ 동일 작업자 + 동일 시간대 중복 제거
    df = df.drop_duplicates(subset=['작업자', '시작일시', '종료일시'])
    df['작업자목록'] = df['작업자'].apply(split_workers)
    df['조구성'] = df['작업자목록'].apply(lambda x: '2인 1조' if len(x) >= 2 else '1인 1조')
    df = df.explode('작업자목록')
    df['작업자'] = df['작업자목록'].astype(str).str.strip()
    df.drop(columns=['작업자목록'], inplace=True)

    # 월 내 단순 주차(정책에 따라 조정 가능)
    df['주차'] = df['시작일시'].apply(lambda x: f"{x.month}월{x.day // 7 + 1}주")
    # ISO 주차도 함께 생성(사이드바에서 선택 적용)
    df['ISO주차'] = df['시작일시'].dt.strftime('%G-W%V')

    # 날짜 타입 통일: datetime64[ns]로 보정해 병합 오류 방지
    df['작업일'] = df['시작일시'].dt.normalize()
    return df, df_original


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# 📦 캐시 처리: 파일 바이트 + dayfirst를 키로 사용
@st.cache_data
def cached_process(file_bytes, dayfirst):
    buf = BytesIO(file_bytes)
    buf.seek(0)
    return process_data(buf, dayfirst)

# 📦 멀티시트 엑셀 변환
def to_excel(sheets: dict):
    """멀티시트 XLSX 생성. 엔진 미설치 시 ZIP(CSV 묶음)으로 폴백.
    Returns: (bytes, fmt) where fmt in {"xlsx", "zip"}
    """
    # 1) 가능한 엑셀 엔진 시도(openpyxl → xlsxwriter)
    for engine in ("openpyxl", "xlsxwriter"):
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine=engine) as writer:
                for name, d in sheets.items():
                    if hasattr(d, 'data'):
                        d = d.data
                    d.to_excel(writer, index=False, sheet_name=name[:31])
            return output.getvalue(), "xlsx"
        except ModuleNotFoundError:
            continue
        except Exception:
            # 다른 예외는 다음 옵션으로 폴백
            continue

    # 2) 엑셀 엔진이 없으면 ZIP으로 CSV 묶음 제공
    zbuf = BytesIO()
    with zipfile.ZipFile(zbuf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name, d in sheets.items():
            if hasattr(d, 'data'):
                d = d.data
            safe = f"{name[:31]}.csv"
            zf.writestr(safe, d.to_csv(index=False, encoding='utf-8-sig'))
    return zbuf.getvalue(), "zip"




# 표시용: 시작/종료 일시에서 초 단위 제거(해당 컬럼이 있을 때만)
def format_dt_display(df: pd.DataFrame) -> pd.DataFrame:
    disp = df.copy()
    for col in ("시작일시", "종료일시"):
        if col in disp.columns:
            if pd.api.types.is_datetime64_any_dtype(disp[col]):
                disp[col] = disp[col].dt.strftime('%Y-%m-%d %H:%M')
            else:
                disp[col] = pd.to_datetime(disp[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
    return disp


def main():
    col1, col2 = st.columns([8, 1])
    with col1:
        st.markdown("""
<h1 style='font-size: 50px;'>📊  <span style='color:#d32f2f;'>MOS</span>tagram 분석 대시보드</h1>
""", unsafe_allow_html=True)
    with col2:
        logo_base64 = st.session_state.get('logo_base64', "")
        if logo_base64:
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
📤 MOStagram에서 업무일지 데이터 파일을 <b>다운로드한 후</b>, 해당 파일을 
<b><span style='color:red;'>Browse files</span></b> 버튼을 통해 업로드하면 자동으로 
<b>분석 대시보드</b>가 렌더링됩니다.
</div>
""", unsafe_allow_html=True)

    # 🧰 샘플 데이터 사용 (파일이 없을 때 UI 체험용)
    with st.expander("🧰 샘플 데이터 사용 (파일이 없을 때 UI 체험용)", expanded=False):
        if st.button("샘플 CSV 불러오기"):
            sample = pd.DataFrame({
                '팀': ['A','A','B','B'],
                '작업자': ['홍길동','김철수','이영희','박영수'],
                '시작일시': pd.to_datetime(['2025-08-10 09:00','2025-08-10 10:00','2025-08-11 09:30','2025-08-11 11:00']),
                '종료일시': pd.to_datetime(['2025-08-10 12:00','2025-08-10 15:00','2025-08-11 13:00','2025-08-11 14:30']),
                '업무종류': ['무선','무선','유선','무선'],
                '구분': ['장애/알람(AS)','사무업무','장애/알람(AS)','장애/알람(AS)'],
                '장비ID': ['E1','E2','E3','E3'],
                '장비명': ['국소1','국소2','국소3','국소3'],
            })
            st.session_state['sample_df'] = sample
            st.success("샘플 데이터가 로드되었습니다. 아래 분석을 확인해보세요.")
            # ▶ 샘플 로드 시에도 사이드바 자동 열림 (1회만 rerun)
            ensure_sidebar_open_once()
    # 📅 파싱 설정(업로드 전에 선택 가능)
    with st.sidebar:
        st.subheader("📅 파싱 설정")
        dayfirst_opt = st.checkbox("날짜가 '일/월/연' 형식(dayfirst)", value=False)

    if uploaded_file or st.session_state.get('sample_df') is not None:
        # 업로드/샘플 분기
        if uploaded_file:
            # ✅ 컬럼 존재 검증(업로드 직후) — 인코딩 대비
            try:
                cols_df = pd.read_csv(uploaded_file, nrows=0)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                cols_df = pd.read_csv(uploaded_file, nrows=0, encoding='cp949')
            except Exception as e:
                st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
                st.stop()
            required_cols = ['팀','작업자','시작일시','종료일시','업무종류','구분','장비ID','장비명']
            missing = [c for c in required_cols if c not in cols_df.columns]
            if missing:
                st.error(f"필수 컬럼 누락: {missing} — CSV 컬럼명을 확인해주세요.")
                st.stop()
            uploaded_file.seek(0)

            # ✅ 데이터 가공
            # ▶ 업로드 직후 사이드바 자동 열림 (1회만 rerun)
            ensure_sidebar_open_once()
            file_bytes = uploaded_file.getvalue()
            with st.spinner("데이터 처리 중..."):
                df, _ = cached_process(file_bytes, dayfirst_opt)
        else:
            # 샘플 데이터 처리
            sample_df = st.session_state.get('sample_df').copy()
            _buf = StringIO()
            sample_df.to_csv(_buf, index=False)
            _buf.seek(0)
            with st.spinner("데이터 처리 중..."):
                df, _ = process_data(_buf, dayfirst_opt)

        # 표준화(띄어쓰기/동의어 정리)
        df['구분'] = df['구분'].astype(str).str.strip().replace({
            '장애/알람(AS)': '장애/알람',
            '사무업무 ': '사무업무'
        })
        df['팀'] = df['팀'].astype(str).str.strip()

        st.success(f"업로드 완료: {len(df):,}행 로드")

        
        with st.sidebar:
            st.header("🔍 검색")
            min_date = df['시작일시'].min().date()
            max_date = df['종료일시'].max().date()
            start_date, end_date = st.date_input("작업 기간 필터", [min_date, max_date], min_value=min_date, max_value=max_date)

            week_mode_options = ["월내주차", "ISO주차"]
            week_mode = st.radio("주차 기준", week_mode_options, index=0, horizontal=True)

            st.subheader("⚙️ 설정")
            daily_avg_threshold_hours = st.number_input("일별 평균작업시간 기준(시간)", min_value=0.0, max_value=24.0, value=6.2, step=0.1, format="%.1f")
            util_threshold = st.number_input("가동률 기준(%)", min_value=0, max_value=100, value=68, step=1)

            # 작업시간 오류 임계값 (8시간 기본)
            error_threshold_h = st.number_input("작업시간 오류 임계값(시간)", min_value=1.0, max_value=24.0, value=8.0, step=0.5, format="%.1f")
            error_threshold_min = int(error_threshold_h * 60)
            st.caption(f"현재 임계값: {error_threshold_h:.1f}시간 = {error_threshold_min}분")

            with st.expander("🔁 중복 출동 기준", expanded=False):
                dup_threshold = st.slider("최소 출동 횟수(건)", 2, 10, 3, 1)
                recent_days = st.slider("최근 N일만 보기", 0, 90, 0, 1)

            # ⭐ 필터 즐겨찾기 (저장/불러오기)
            # 필터 초기화
            if st.button("필터 초기화"):
                st.rerun()

        df = df[(df['시작일시'].dt.date >= start_date) & (df['종료일시'].dt.date <= end_date)]

        # 주차 표기 기준 선택 적용
        if week_mode == "월내주차":
            df['주차_표시'] = df['주차']
        else:
            df['주차_표시'] = df['ISO주차']

        if '팀' in df.columns:
            team_options = df['팀'].dropna().unique().tolist()
            team_list = ["전체"] + team_options
            selected_team = st.sidebar.selectbox("팀 선택", team_list, index=0)
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
            st.session_state['selected_team'] = selected_team
            st.session_state['selected_members'] = selected_members

        # 🔎 텍스트 검색 필터 (장비명/작업자)
        with st.sidebar.expander("🔎 텍스트 검색", expanded=False):
            q = st.text_input("장비명/작업자 검색(부분일치)", value="")
        st.session_state['q_text'] = q
        if q:
            df = df[
                df['장비명'].astype(str).str.contains(q, case=False, regex=False, na=False) |
                df['작업자'].astype(str).str.contains(q, case=False, regex=False, na=False)
            ]
        # 필터 요약
        total_members = len(member_options) if 'member_options' in locals() else 0
        sel_members = st.session_state.get('selected_members', [])
        sel_text = "전체" if (not sel_members or (total_members and len(sel_members) == total_members)) else f"{len(sel_members)}/{total_members}명"
        st.info(f"기간: {start_date}~{end_date} | 팀: {st.session_state.get('selected_team','전체')} | 작업자 선택: {sel_text} | 검색어: {q or '-'} | 임계값: {error_threshold_h:.1f}시간({error_threshold_min}분)")

        # 파일명 태그(팀/기간) 공통 정의

        # 데이터 없음 방지
        if df.empty:
            st.warning("선택한 조건에 해당하는 데이터가 없습니다. 필터를 조정해 주세요.")
            st.markdown("## ⚠️ 작업시간 오류")
            st.write("- 0분 작업시간: **0건**")
            st.write(f"- 팀 결측: **0건**")
            st.info("사이드바에서 '필터 초기화'를 누르거나 기간/검색어를 조정해 주세요.")
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

        # 필터링 한 번만 수행
        dup_equipment = df[
            (df['업무종류'] == '무선') &
            (df['구분'] == '장애/알람') &
            df['장비ID'].notna() & (sstr(df['장비ID']) != '') &
            df['장비명'].notna() & (sstr(df['장비명']) != '') &
            (~sstr(df['장비명']).str.contains('민원', regex=False)) &
            (~sstr(df['장비명']).str.contains('사무', regex=False))
        ].copy()

        if 'recent_days' in locals() and recent_days > 0:
            cutoff = pd.to_datetime(end_date) - pd.Timedelta(days=recent_days)
            dup_equipment = dup_equipment[dup_equipment['시작일시'] >= cutoff]

        # 방문(실제 출동) 단위 기준으로 계산
        # - 중복건수: 팀/장비명/장비ID/시작일시/종료일시(=1회 방문) 유니크 개수
        # - 작업자 출동횟수: 방문-작업자 유니크 조합 수
        visit_keys = ['팀','장비명','장비ID','시작일시','종료일시']
        visits = dup_equipment[visit_keys].drop_duplicates()

        # 장비별 중복건수 = 방문수
        count_tbl = (
            visits
            .groupby(['팀','장비명','장비ID'], dropna=False)
            .size()
            .reset_index(name='중복건수')
        )

        # 작업자별 출동 횟수 = 방문-작업자 유니크 조합 수
        worker_cnt = (
            dup_equipment[['팀','장비명','장비ID','작업자','시작일시','종료일시']]
            .drop_duplicates()
            .groupby(['팀','장비명','장비ID','작업자'])
            .size()
            .reset_index(name='방문수')
        )
        worker_list = (
            worker_cnt
            .assign(표시=lambda d: sstr(d['작업자']) + '(' + d['방문수'].astype(int).astype(str) + ')')
            .groupby(['팀','장비명','장비ID'])['표시']
            .apply(lambda s: ', '.join(s))
            .reset_index(name='작업자(출동 횟수)')
        )

        combined = (
            count_tbl.merge(worker_list, on=['팀','장비명','장비ID'], how='left')
            .query('중복건수 >= @dup_threshold')
            .sort_values('중복건수', ascending=False)
            .reset_index(drop=True)
        )

        dup_display = combined.rename(columns={'팀': '운용팀'})

        # 📈 메트릭(중복 출동)
        _dup_cnt = int(len(dup_display))
        _dup_max = int(dup_display['중복건수'].max()) if not dup_display.empty else 0
        _dup_avg = float(dup_display['중복건수'].mean()) if not dup_display.empty else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("중복 장비 수", f"{_dup_cnt}")
        m2.metric("최대 중복 횟수", _dup_max)
        m3.metric("평균 중복 횟수", f"{_dup_avg:.1f}")

        # 파일명 태그(팀/기간)
        date_tag = f"{start_date}_{end_date}"
        team_tag = "전체" if st.session_state.get('selected_team') in [None, "전체"] else st.session_state['selected_team']

        # 🔎 드릴다운: 장비명 선택(메트릭 바로 아래)
        if not dup_display.empty:
            _names = sstr(dup_display['장비명']).unique().tolist()
            _sel = st.selectbox("🔎 장비명 선택(드릴다운)", ["선택 안함"] + _names, index=0)
            if _sel != "선택 안함":
                det = df[sstr(df['장비명']) == str(_sel)].copy()
                det['작업자'] = det['작업자'].astype(str).str.strip()

                visit_keys = ['팀','장비명','장비ID','시작일시','종료일시','구분']
                workers_join = (
                    det.groupby(visit_keys)['작업자']
                      .apply(lambda s: ', '.join(sorted(set(s))))
                      .reset_index()
                )

                # ▶ 작업내용(해시태그) 병합: 후보 컬럼 자동 탐색 후 '작업내용'으로 표시
                _norm = lambda x: ''.join(str(x).split()).lower()
                _colmap = {_norm(c): c for c in det.columns}
                _cands = ['작업내용','해시태그','해시태그(작업내용)','hashtags','hashtag','업무내용','내용']
                _src = None
                for _c in _cands:
                    if _norm(_c) in _colmap:
                        _src = _colmap[_norm(_c)]
                        break
                if _src is not None:
                    _content = (
                        det.groupby(visit_keys)[_src]
                          .apply(lambda s: ', '.join(sorted(set([str(x).strip() for x in s if str(x).strip() and str(x).strip().lower()!='nan']))))
                          .reset_index()
                          .rename(columns={_src:'작업내용'})
                    )
                else:
                    _content = (
                        det.groupby(visit_keys).size().reset_index(name='__tmp').drop(columns='__tmp').assign(작업내용='-')
                    )

                # 동일 방문의 작업시간(분)은 동일하므로 첫 값 사용
                dur = det.groupby(visit_keys)['작업시간(분)'].first().reset_index()

                det_view = workers_join.merge(_content, on=visit_keys, how='left').merge(dur, on=visit_keys, how='left')
                det_view = det_view[['팀','작업자','구분','작업내용','장비명','시작일시','종료일시','작업시간(분)']].sort_values('시작일시')

                st.dataframe(format_dt_display(det_view), use_container_width=True)
                st.download_button(
                    "⬇️ 선택 장비 상세 CSV",
                    data=convert_df_to_csv(det_view),
                    file_name=f"장비상세_{_sel}_{team_tag}_{date_tag}.csv",
                    mime="text/csv"
                )

        st.dataframe(
            format_dt_display(dup_display),
            use_container_width=True,
            column_config={'중복건수': st.column_config.NumberColumn(format="%d")}
        )
        st.download_button(
            "⬇️ 중복 출동 현황 CSV",
            data=convert_df_to_csv(dup_display),
            file_name=f"중복출동현황_{team_tag}_{date_tag}.csv",
            mime="text/csv"
        )

        # ─────────────────────────────────────────────────────────
        # 👤 개인별 누락 현황 — 중복 분리 제거(이미 process_data에서 explode됨)
        st.markdown("## 📋 개인별 누락 현황")
        
        # 모든 작업자 목록 & 영업일 날짜 생성
        workers = df[['팀','작업자']].dropna().drop_duplicates()
        date_range = pd.date_range(start=df['작업일'].min(), end=df['작업일'].max(), freq='B')
        all_worker_days = pd.MultiIndex.from_product([workers['작업자'], date_range], names=['작업자','작업일'])
        all_worker_days = pd.DataFrame(index=all_worker_days).reset_index().merge(workers, on='작업자', how='left')

        # 일자별 작성여부(여러 건→1)
        actual_logs = df.groupby(['팀','작업자','작업일']).size().rename('작성여부').reset_index()
        actual_logs['작성여부'] = 1
        log_df = all_worker_days.merge(actual_logs, on=['팀','작업자','작업일'], how='left').fillna({'작성여부':0})

        # 📈 메트릭(누락 현황)
        _ps_all = log_df.groupby(['팀','작업자'])['작성여부'].agg(['mean','count']).reset_index()
        _ps_all['누락일수'] = (1 - _ps_all['mean']) * _ps_all['count']
        n1, n2, n3 = st.columns(3)
        n1.metric("누락 대상 인원", int((_ps_all['mean'] < 1.0).sum()))
        n2.metric("총 누락 일수", int(_ps_all['누락일수'].sum()))
        n3.metric("평균 누락률(전체)", f"{int((1 - log_df['작성여부'].mean()) * 100)}%")

        # ✔ 개인별 누락 현황 — 표 형식(중복 출동 현황과 동일 스타일)
        personal_summary = _ps_all[_ps_all['mean'] < 1.0].copy()
        personal_summary['누락률(%)'] = (1 - personal_summary['mean']) * 100
        personal_summary['누락일수'] = personal_summary['누락일수'].astype(int)
        personal_summary['누락률(%)'] = personal_summary['누락률(%)'].astype(int)
        table_df = (
            personal_summary[['팀','작업자','누락일수','누락률(%)']]
            .sort_values(by=['누락일수','누락률(%)'], ascending=[False, False])
            .rename(columns={'팀':'운용팀'})
            .reset_index(drop=True)
        )
        st.dataframe(table_df, use_container_width=True, column_config={'누락일수': st.column_config.NumberColumn(format="%d"), '누락률(%)': st.column_config.NumberColumn(format="%d%%")})
        st.download_button("⬇️ 개인별 누락 현황 CSV", data=convert_df_to_csv(table_df), file_name=f"개인별누락현황_{team_tag}_{date_tag}.csv", mime="text/csv")

        # ❗ 데이터 품질 점검 (항상 표시)
        st.markdown("## ⚠️ 작업시간 오류")
        zero_cnt = int((df['작업시간(분)'] == 0).sum())
        null_team = int(df['팀'].isna().sum())
        # 오류 요약 메트릭 카드
        long_cnt = int((df['작업시간(분)'] >= error_threshold_min).sum())
        kz1, kz2, kz3 = st.columns(3)
        kz1.metric("0분 작업", zero_cnt)
        kz2.metric("임계값 초과", long_cnt)
        kz3.metric("팀 결측", null_team)

        # ⬇️ 문제 행 다운로드 (음수/0분/날짜 오류건)
        zero_rows = df[df['작업시간(분)'] == 0][['팀','작업자','구분','장비명','작업시간(분)','시작일시','종료일시']]
        long_rows = df[df['작업시간(분)'] >= error_threshold_min][['팀','작업자','구분','장비명','작업시간(분)','시작일시','종료일시']]
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ 0분 작업시간 CSV", data=convert_df_to_csv(zero_rows), file_name=f"0분_작업시간_{team_tag}_{date_tag}.csv", mime="text/csv")
        with c2:
            st.download_button("⬇️ 작업시간 임계값 초과 CSV", data=convert_df_to_csv(long_rows), file_name=f"작업시간_임계값_초과_{team_tag}_{date_tag}.csv", mime="text/csv")
        # 📋 작업시간 오류 목록
        tab1, tab2 = st.tabs(["임계값 초과 목록", "0분 작업 목록"])
        with tab1:
            keep_sort = st.checkbox("작업시간 내림차순 고정", value=True)
            long_rows_display = long_rows.copy()
            if keep_sort:
                long_rows_display = long_rows_display.sort_values('작업시간(분)', ascending=False)
            st.dataframe(format_dt_display(long_rows_display), use_container_width=True, height=420)
        with tab2:
            recent_first = st.checkbox("최근순 보기", value=False, key="zero_recent_first")
            zero_rows_display = zero_rows.copy()
            if recent_first:
                zero_rows_display = zero_rows_display.sort_values('시작일시', ascending=False)
            else:
                zero_rows_display = zero_rows_display.sort_values(['팀','작업자','시작일시'])
            st.dataframe(format_dt_display(zero_rows_display), use_container_width=True, height=420)

        # 🚨 시간 이상 탐지 — 겹침/역전
        st.markdown("## 🚨 시간 이상 탐지 — 겹침/역전")
        # 시작 > 종료 (역전)
        rev_rows = df[df['시작일시'] > df['종료일시']][['팀','작업자','구분','장비명','시작일시','종료일시','작업시간(분)']].copy()
        rev_cnt = int(len(rev_rows))
        # 같은 작업자의 시간 겹침
        _sorted = df.sort_values(['작업자','시작일시'])
        _prev_end = _sorted.groupby('작업자')['종료일시'].shift()
        overlap_mask = _sorted['시작일시'] < _prev_end
        overlap_rows = _sorted[overlap_mask][['팀','작업자','구분','장비명','시작일시','종료일시']].copy()
        overlap_rows = overlap_rows.assign(이전종료=_prev_end[overlap_mask])
        overlap_cnt = int(len(overlap_rows))
        e1, e2 = st.columns(2)
        with e1:
            st.metric("역전 시간(시작>종료)", rev_cnt)
            if rev_cnt > 0:
                st.download_button("⬇️ 역전 시간 CSV", data=convert_df_to_csv(format_dt_display(rev_rows)), file_name=f"역전시간_{team_tag}_{date_tag}.csv", mime="text/csv")
        with e2:
            st.metric("시간 겹침 건수", overlap_cnt)
            if overlap_cnt > 0:
                st.download_button("⬇️ 시간 겹침 CSV", data=convert_df_to_csv(format_dt_display(overlap_rows)), file_name=f"시간겹침_{team_tag}_{date_tag}.csv", mime="text/csv")

        tab_ov, tab_rev = st.tabs(["시간 겹침 목록", "역전 시간 목록"])
        with tab_ov:
            ov_recent = st.checkbox("최근순 보기", value=False, key="ov_recent_first")
            ov_view = overlap_rows.copy()
            if ov_recent:
                ov_view = ov_view.sort_values('시작일시', ascending=False)
            else:
                ov_view = ov_view.sort_values(['팀','작업자','시작일시'])
            st.dataframe(format_dt_display(ov_view), use_container_width=True, height=420)
        with tab_rev:
            rev_recent = st.checkbox("최근순 보기", value=False, key="rev_recent_first2")
            rev_view = rev_rows.copy()
            if rev_recent:
                rev_view = rev_view.sort_values('시작일시', ascending=False)
            else:
                rev_view = rev_view.sort_values(['팀','작업자','시작일시'])
            st.dataframe(format_dt_display(rev_view), use_container_width=True, height=420)

        st.markdown("## 🕒 구분별 MTTR / 반복도")

        _mttr_keys = ['팀', '원본작업자', '시작일시', '종료일시', '구분', '장비ID']
        mttr_df = df.drop_duplicates(subset=_mttr_keys).copy()

        # 장비ID 정리 및 음수 작업시간 제거
        mttr_df['장비ID'] = mttr_df['장비ID'].astype(str).str.strip()
        # 음수 제외 + 8시간(480분) 초과 건 제외
        mttr_df = mttr_df[(mttr_df['작업시간(분)'] >= 0) & (mttr_df['작업시간(분)'] <= error_threshold_min)]

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
            result['고유 업무 수'] = result['고유장비수'].astype('Int64')
            result['중복업무 수'] = result['재발장비수'].astype('Int64')
            result['중복업무 비율(%)'] = result['중복업무 비율(%)'].round().astype('Int64')

            display_cols = ['구분', '건수', 'MTTR(분)', '중앙값(분)', 'P90(분)', '고유 업무 수', '중복업무 수', '중복업무 비율(%)']
            result_display = (
                result[display_cols]
                .sort_values(['MTTR(분)'], ascending=[True])
                .reset_index(drop=True)
            )
            st.dataframe(result_display, use_container_width=True, column_config={'건수': st.column_config.NumberColumn(format="%d"), 'MTTR(분)': st.column_config.NumberColumn(format="%d"), '중앙값(분)': st.column_config.NumberColumn(format="%d"), 'P90(분)': st.column_config.NumberColumn(format="%d"), '고유 업무 수': st.column_config.NumberColumn(format="%d"), '중복업무 수': st.column_config.NumberColumn(format="%d"), '중복업무 비율(%)': st.column_config.NumberColumn(format="%d%%")})
            st.download_button("⬇️ MTTR/반복도 결과 CSV", data=convert_df_to_csv(result_display), file_name=f"MTTR_반복도_결과_{team_tag}_{date_tag}.csv", mime="text/csv")

            # 📌 요약 KPI
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("총 로그 수", f"{len(df):,}")
            k2.metric("팀 수", int(df['팀'].nunique()))
            try:
                avg_mttr = int(pd.to_numeric(result_display['MTTR(분)'], errors='coerce').dropna().mean())
                p90_dup = int(pd.to_numeric(result_display['중복업무 비율(%)'], errors='coerce').dropna().quantile(0.90))
            except Exception:
                avg_mttr, p90_dup = 0, 0
            k3.metric("평균 MTTR(분)", avg_mttr)
            k4.metric("중복업무 비율 P90(%)", f"{p90_dup}%")

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

            # ✅ 카테고리 순서 고정(두 그래프 일관성)
            order = result_display.sort_values('MTTR(분)')['구분'].astype(str).tolist()
            fig_mttr.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': order})
            fig_dup_ratio.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': order})

            cols_mttr = st.columns(2)
            with cols_mttr[0]:
                st.plotly_chart(fig_mttr, use_container_width=True)
            with cols_mttr[1]:
                st.plotly_chart(fig_dup_ratio, use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # 🗓️ 운용팀 일별 작성현황 (이동업무 포함)
        st.markdown("## 🗓️ 운용팀 일별 작성 현황")
        daily_count = df.groupby([df['시작일시'].dt.date, df['팀']]).size().unstack(fill_value=0).astype(int)
        daily_count.loc['합계'] = daily_count.sum()
        st.dataframe(daily_count, use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # 📉 팀 주차별 가동률 (이동업무 포함)
        st.markdown("## 📉 팀 주차별 가동률")

        # 작업시간 기반(인원=전체, 누락률≥50% 제외)
# (자동정리) 중복 정의 제거:         BASIS_MIN_PER_DAY = 480  # 기준 근무시간(분/일)

        # 1) 단일 기록 기준 '비정상적으로 긴 시간' 제외 — 한 작업당 10시간(600분) 초과 제외
# (자동정리) 중복 정의 제거:         ABNORMAL_TASK_MIN = 600  # 10시간(분)
        util_df = df[(df['작업시간(분)'] >= 0) & (df['작업시간(분)'] < ABNORMAL_TASK_MIN)].copy()

        # 데이터 없으면 안내 후 섹션 종료
        if util_df.empty:
            st.info("가동률을 계산할 수 있는 데이터가 없습니다. (단일 기록 10시간 초과 제외 또는 필터로 인해 공집합)")
        else:
            # 1) 단일 기록 기준 '비정상적으로 긴 시간' 제외 — 한 작업당 10시간(600분) 초과 제외
            # (주의) 아래 한 줄은 이미 전역에 ABNORMAL_TASK_MIN = 600이 정의되어 있으면 중복 정의하지 않습니다.
            #         전역 정의가 없다면, 다음 줄을 활성화하세요(주석 해제).
            # ABNORMAL_TASK_MIN = 600  # 10시간(분)

            util_df = df[(df['작업시간(분)'] >= 0) & (df['작업시간(분)'] < ABNORMAL_TASK_MIN)].copy()

            # 2) 같은 '업무(구분)' 내에서만 시간 겹침 병합 — 완전 벡터화(대용량 최적화)
            grp_keys = ['팀', '작업자', '구분', '작업일', '주차_표시']
            tmp = (
                util_df[grp_keys + ['시작일시', '종료일시']]
                .dropna(subset=['시작일시', '종료일시'])
                .sort_values(grp_keys + ['시작일시'])
                .copy()
            )
            tmp['종료_cummax'] = tmp.groupby(grp_keys)['종료일시'].cummax()
            prev_cummax = tmp.groupby(grp_keys)['종료_cummax'].shift()
            tmp['새구간'] = prev_cummax.isna() | (tmp['시작일시'] > prev_cummax)
            tmp['세그'] = tmp.groupby(grp_keys)['새구간'].cumsum()
            segments = (
                tmp.groupby(grp_keys + ['세그'])
                   .agg(seg_start=('시작일시', 'min'), seg_end=('종료일시', 'max'))
                   .reset_index()
            )
            segments['병합작업시간(분)'] = (segments['seg_end'] - segments['seg_start']).dt.total_seconds() / 60.0
            merged = (
                segments.groupby(grp_keys, as_index=False)['병합작업시간(분)']
                        .sum()
            )

            # 3) 팀×주차별 주간 작업시간 합(분)
            team_time = (
                merged.groupby(['팀','주차_표시'], as_index=False)['병합작업시간(분)']
                      .sum()
                      .rename(columns={'병합작업시간(분)':'주간작업시간_합(분)'})
            )

            # 주차별 영업일수(B, 평일) 계산 — 화면 주차 표시 체계와 동일하게 라벨링
            biz_days = pd.date_range(start=df['작업일'].min(), end=df['작업일'].max(), freq='B')
            cal = pd.DataFrame({'작업일': biz_days})
            if week_mode == "월내주차":
                cal['주차_표시'] = cal['작업일'].apply(lambda x: f"{x.month}월{x.day // 7 + 1}주")
            else:
                cal['주차_표시'] = pd.to_datetime(cal['작업일']).dt.strftime('%G-W%V')
            bdf = cal.groupby('주차_표시')['작업일'].nunique().reset_index(name='영업일수')

            # 팀 인원(전체) — 개인별 누락률 ≥50% 작업자는 제외 (작성여부 mean<=0.5)
            try:
                valid_workers = _ps_all[_ps_all['mean'] > 0.5][['팀','작업자']].dropna().drop_duplicates()
                team_all = valid_workers.groupby('팀')['작업자'].nunique().reset_index(name='팀인원_전체')
            except Exception:
                team_all = (
                    df[['팀','작업자']].dropna().drop_duplicates()
                      .groupby('팀')['작업자'].nunique().reset_index(name='팀인원_전체')
                )

            # 4) 병합 및 가동률(비율) 계산 (0~1, 상한 1.0)
            df_weekly = (
                team_time
                .merge(bdf, on='주차_표시', how='left')
                .merge(team_all, on='팀', how='left')
            )
            denom = (df_weekly['영업일수'] * BASIS_MIN_PER_DAY * df_weekly['팀인원_전체']).replace(0, np.nan)
            df_weekly['가동률(%)'] = (df_weekly['주간작업시간_합(분)'] / denom).clip(upper=1.0)
            df_weekly = df_weekly.sort_values(['팀','주차_표시']).reset_index(drop=True)

            team_count = df['팀'].nunique()
            base_line = util_threshold / 100.0

            # 📈 메트릭(가동률)
            _util_avg = float(df_weekly['가동률(%)'].mean()) if not df_weekly.empty else 0.0
            _team_avg = df_weekly.groupby('팀')['가동률(%)'].mean() if not df_weekly.empty else pd.Series(dtype=float)
            _team_above = int((_team_avg >= base_line).sum()) if not df_weekly.empty else 0
            _vals = (df_weekly['가동률(%)'] * 100).replace([np.inf, -np.inf], np.nan).dropna()
            _util_p90 = int(np.nanpercentile(_vals, 90)) if (not df_weekly.empty and len(_vals) > 0) else 0
            u1, u2, u3 = st.columns(3)
            u1.metric("평균 가동률", f"{int(_util_avg*100)}%")
            u2.metric("기준 이상 팀 수", _team_above)
            u3.metric("가동률 P90", f"{_util_p90}%")

            # 차트
            fig_util = px.bar(
                df_weekly,
                x='팀', y='가동률(%)', color='주차_표시', barmode='group',
                title='팀 주차별 가동률', labels={'가동률(%)': '가동률', '팀': '팀'}
            )
            fig_util.update_layout(
                yaxis_tickformat='.0%', yaxis_range=[0, 1],
                legend_title_text='주차', legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center')
            )
            fig_util.add_shape(
                type="line", x0=-0.5, x1=max(-0.5, team_count - 0.5),
                y0=base_line, y1=base_line, line=dict(color="red", width=2, dash="dot")
            )
            fig_util.add_annotation(
                x=max(-0.5, team_count - 0.5), y=base_line,
                text=f"기준: {util_threshold}%", showarrow=False, yshift=10, font=dict(color="red")
            )
            st.plotly_chart(fig_util, use_container_width=True)
            st.caption("※ 계산식: (같은 '업무(구분)' 내 겹침 병합 후) 주간작업시간합 ÷ (영업일수 × 480분 × 팀 인원(전체, 누락률≥50% 제외)) × 100 · 단일 기록 10시간 초과 제외.")

        # 📊 일별 평균 작업 시간 — 10시간 초과 제외 + 같은 '업무(구분)' 내 겹침 병합 적용
        st.markdown("## 📊 일별 평균 작업 시간")

# (자동정리) 중복 정의 제거:         ABNORMAL_TASK_MIN = 600  # 한 작업당 10시간(분) 초과 제외
        daily_src = df[(df['작업시간(분)'] >= 0) & (df['작업시간(분)'] < ABNORMAL_TASK_MIN)].copy()

        if daily_src.empty:
            st.info("일별 평균 작업 시간을 계산할 데이터가 없습니다. (단일 기록 10시간 초과 제외 등)")
            daily_sum = pd.DataFrame(columns=['작업일', '팀', '작업시간(분)'])
            daily_worker_count = pd.DataFrame(columns=['작업일', '팀', '작업자수'])
        else:
            # 같은 '업무(구분)' 내에서 겹침 병합 — 완전 벡터화(대용량 최적화)
            gkeys = ['팀', '작업자', '구분', '작업일']
            tmp = (
                daily_src[gkeys + ['시작일시', '종료일시']]
                .dropna(subset=['시작일시', '종료일시'])
                .sort_values(gkeys + ['시작일시'])
                .copy()
            )
            # 그룹별 종료시각 누적최댓값과 이전 구간의 누적최댓값
            tmp['종료_cummax'] = tmp.groupby(gkeys)['종료일시'].cummax()
            prev_cummax = tmp.groupby(gkeys)['종료_cummax'].shift()
            # 새 구간 시작 여부(이전 종료 누적최댓값보다 시작이 뒤면 새 구간)
            tmp['새구간'] = prev_cummax.isna() | (tmp['시작일시'] > prev_cummax)
            # 그룹 내 구간 번호
            tmp['세그'] = tmp.groupby(gkeys)['새구간'].cumsum()
            # 각 세그먼트의 [min 시작, max 종료]
            segments = (
                tmp.groupby(gkeys + ['세그'])
                   .agg(seg_start=('시작일시', 'min'), seg_end=('종료일시', 'max'))
                   .reset_index()
            )
            # 세그먼트 길이(분) 계산 후 그룹별 합
            segments['병합작업시간(분)'] = (segments['seg_end'] - segments['seg_start']).dt.total_seconds() / 60.0
            merged_daily = (
                segments.groupby(gkeys, as_index=False)['병합작업시간(분)']
                        .sum()
            )
            # 팀별 일자 합계 (차트/표 입력용 DataFrame)
            daily_sum = (
                merged_daily
                .groupby(['작업일', '팀'], as_index=False)['병합작업시간(분)']
                .sum()
                .rename(columns={'병합작업시간(분)': '작업시간(분)'})
            )

            # 일자별 팀 작업자 수 (unique)
            daily_worker_count = (
                daily_src[['작업일', '팀', '작업자']]
                .dropna()
                .drop_duplicates()
                .groupby(['작업일', '팀'])['작업자']
                .nunique()
                .reset_index(name='작업자수')
            )

        # 평균 계산 및 시각화
        daily_avg = daily_sum.merge(daily_worker_count, on=['작업일', '팀'], how='inner')
        daily_avg = daily_avg.replace([np.inf, -np.inf], np.nan).dropna(subset=['작업자수'])
        daily_avg['평균작업시간(시간)'] = (daily_avg['작업시간(분)'] / daily_avg['작업자수']) / 60

        # 📈 메트릭(일별 평균 작업 시간)
        _mean_hours = float(daily_avg['평균작업시간(시간)'].mean()) if not daily_avg.empty else 0.0
        _exceed_cnt = int((daily_avg['평균작업시간(시간)'] >= daily_avg_threshold_hours).sum()) if not daily_avg.empty else 0
        _max_hours = float(daily_avg['평균작업시간(시간)'].max()) if not daily_avg.empty else 0.0
        d1, d2, d3 = st.columns(3)
        d1.metric("평균(시간)", f"{_mean_hours:.1f}")
        d2.metric("기준 초과 건수", _exceed_cnt)
        d3.metric("최대 평균시간", f"{_max_hours:.1f}")

        fig_daily = px.bar(
            daily_avg,
            x='팀',
            y='평균작업시간(시간)',
            color='작업일',
            barmode='group',
            title='일별 평균 작업 시간',
            labels={'평균작업시간(시간)': '평균 작업 시간(시간)', '작업일': '날짜'}
        )
        ymax = max(10.0, float(daily_avg_threshold_hours) * 1.2)
        fig_daily.update_layout(
            yaxis_range=[0, ymax],
            legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center')
        )
        _x1 = max(-0.5, len(daily_avg['팀'].unique()) - 0.5)
        fig_daily.add_shape(
            type="line",
            x0=-0.5, x1=_x1,
            y0=daily_avg_threshold_hours, y1=daily_avg_threshold_hours,
            line=dict(color="red", width=2, dash="dot")
        )
        fig_daily.add_annotation(
            x=_x1,
            y=daily_avg_threshold_hours,
            text=f"기준: {daily_avg_threshold_hours:.1f}시간",
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
        # 📈 메트릭(팀별 운용조)
        try:
            _avg_one = float(crew_summary_percent['1인 1조'].mean()) if '1인 1조' in crew_summary_percent.columns else 0.0
            _avg_two = float(crew_summary_percent['2인 1조'].mean()) if '2인 1조' in crew_summary_percent.columns else 0.0
        except Exception:
            _avg_one, _avg_two = 0.0, 0.0
        _team_n = int(crew_summary_percent.shape[0])
        c1, c2, c3 = st.columns(3)
        c1.metric("평균 1인 1조 비율", f"{int(round(_avg_one))}%")
        c2.metric("평균 2인 1조 비율", f"{int(round(_avg_two))}%")
        c3.metric("팀 수", _team_n)
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
        # 📈 메트릭(업무구분별 인원조)
        try:
            _avg_two_task = float(crew_task_ratio['2인 1조'].mean()) if '2인 1조' in crew_task_ratio.columns else 0.0
            _avg_one_task = float(crew_task_ratio['1인 1조'].mean()) if '1인 1조' in crew_task_ratio.columns else 0.0
        except Exception:
            _avg_two_task, _avg_one_task = 0.0, 0.0
        _task_n = int(crew_task_ratio.shape[0])
        t1, t2, t3 = st.columns(3)
        t1.metric("평균 2인 1조 비율", f"{int(round(_avg_two_task))}%")
        t2.metric("평균 1인 1조 비율", f"{int(round(_avg_one_task))}%")
        t3.metric("업무 구분 수", f"{_task_n}")
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

        # 📊 결과 묶음 다운로드 (XLSX 또는 ZIP 자동)
        excel_bytes, excel_fmt = to_excel({
            '중복출동': dup_display.reset_index(drop=True),
            '개인별누락': table_df.reset_index(drop=True),
            'MTTR_반복도': result_display.reset_index(drop=True),
            '일별작성현황': daily_count.rename_axis(index='작업일').reset_index(),
            '작업시간오류_임계': long_rows.reset_index(drop=True),
            '작업시간오류_0분': zero_rows.reset_index(drop=True),
        })
        if excel_fmt == "xlsx":
            fname = f"업무일지_분석_모음_{team_tag}_{date_tag}.xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            label = "📊 Excel(모든 시트)"
        else:
            fname = f"업무일지_분석_모음_{team_tag}_{date_tag}.zip"
            mime = "application/zip"
            label = "📦 ZIP(모든 CSV)"
        st.sidebar.download_button(label, data=excel_bytes, file_name=fname, mime=mime)


if __name__ == '__main__':
    main()