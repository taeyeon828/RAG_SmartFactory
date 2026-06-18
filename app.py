# app.py
from main import retrieve_context, ask_rag   # ← main.py RAG 연결
 
import streamlit as st
import pandas as pd
import altair as alt
 
st.set_page_config(
    page_title="스마트공장 도입 컨설턴트 AI",
    page_icon="🏭",
    layout="wide"
)
 
# ——————————————
# CSS
# ——————————————
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap');
 
.stApp { background-color: #ffffff; }
 
.block-container {
    max-width: 1500px;
    padding-top: 3.5rem;
    padding-bottom: 3rem;
    padding-left: 4rem;
    padding-right: 4rem;
}
 
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 10px;
    letter-spacing: -0.03em;
    font-family: 'Noto Sans KR', sans-serif;
}
 
.sub-title {
    font-size: 18px;
    color: #374151;
    margin-bottom: 34px;
    line-height: 1.6;
}
 
.header-banner {
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    margin-top: -3.5rem;
    margin-bottom: 40px;
    background: #ffffff;
    padding: 48px 0;
}
 
.header-banner .banner-title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 12px;
    letter-spacing: -0.03em;
}
 
.header-banner .banner-subtitle {
    font-size: 16px;
    color: #374151;
    line-height: 1.7;
    max-width: 800px;
}
 
.header-inner {
    max-width: 1500px;
    margin: 0 auto;
    padding: 0 4rem;
}
 
.card {
    background-color: #ffffff;
    padding: 32px;
    border-radius: 22px;
    border: 1px solid #d9e2e8;
    box-shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
    margin-bottom: 24px;
    transition: all 0.3s ease;
}
 
.card:hover {
    border-color: #b8d9e8;
    box-shadow: 0 12px 32px rgba(17, 24, 39, 0.1);
}
 
.section-title {
    font-size: 23px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 18px;
}
 
.section-badge {
    display: inline-block;
    width: 36px;
    height: 36px;
    background-color: #cfe8f3;
    border: 2px solid #7db9d3;
    border-radius: 50%;
    text-align: center;
    line-height: 32px;
    font-weight: 800;
    color: #111827;
    margin-right: 10px;
    font-size: 16px;
}
 
.postit-box {
    background-color: #cfe8f3;
    padding: 28px 30px;
    border-radius: 24px;
    border: 1px solid #b8d9e8;
    box-shadow: 8px 10px 0px rgba(184, 217, 232, 0.45);
    margin-top: 18px;
    color: #111827;
}
 
.postit-box h3 { font-size: 24px; font-weight: 800; margin-bottom: 18px; color: #111827; }
.postit-box p, .postit-box li { font-size: 15px; line-height: 1.8; color: #111827; }
 
.postit-warning {
    background-color: rgba(255,255,255,0.65);
    padding: 18px 20px;
    border-radius: 18px;
    margin-top: 24px;
    line-height: 1.7;
    color: #111827;
}
 
.info-box {
    background-color: #cfe8f3;
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #b8d9e8;
    color: #111827;
    margin-bottom: 22px;
    line-height: 1.7;
    font-size: 15px;
}
 
.next-box {
    background-color: #cfe8f3;
    padding: 26px;
    border-radius: 22px;
    border: 1px solid #b8d9e8;
    margin-top: 24px;
    color: #111827;
    line-height: 1.7;
}
 
.stSelectbox label, .stMultiSelect label, .stTextArea label {
    color: #111827 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}
 
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background-color: #f8fbfd !important;
    border: 1px solid #cfdce5 !important;
    border-radius: 14px !important;
    min-height: 45px;
    color: #111827 !important;
}
 
.chat-info-box {
    background-color: #cfe8f3;
    border: 1px solid #b8d9e8;
    border-radius: 18px;
    padding: 18px 22px;
    color: #111827;
    line-height: 1.7;
    margin-bottom: 24px;
}
 
.stTextArea textarea {
    background-color: #f8fbfd !important;
    border: 1px solid #cfdce5 !important;
    border-radius: 14px !important;
    color: #111827 !important;
}
 
.input-label {
    font-size: 15px;
    font-weight: 700;
    color: #111827;
    margin-top: 18px;
    margin-bottom: 10px;
}
 
span[data-baseweb="tag"] {
    background-color: #cfe8f3 !important;
    color: #111827 !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
}
 
.stButton > button {
    height: 50px;
    border-radius: 14px;
    border: 2px solid #9fc8dc;
    background-color: #ffffff;
    color: #111827;
    font-weight: 800;
    font-size: 16px;
}
 
.stButton > button:hover {
    background-color: #b8d9e8;
    color: #111827;
    border: 1px solid #8fbfd6;
}
 
div[data-testid="stForm"] .stButton > button,
div[data-testid="stForm"] .stButton > button:focus {
    background-color: #1d6fa8 !important;
    color: #ffffff !important;
    border: 2px solid #1d6fa8 !important;
}
 
div[data-testid="stForm"] .stButton > button:hover {
    background-color: #155d8e !important;
    color: #ffffff !important;
    border: 2px solid #155d8e !important;
}
 
div[data-testid="stForm"] { border: none; padding: 0; }
 
div[data-testid="stMetric"] {
    background-color: #f8fbfd;
    border: 1px solid #d9e2e8;
    border-radius: 16px;
    padding: 16px;
}
 
div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 14px !important;
    margin-bottom: 18px;
}
 
div[role="radiogroup"] label {
    min-height: 60px !important;
    padding: 14px 18px !important;
    border-radius: 16px !important;
    border: 2px solid #d9e2e8 !important;
    background-color: #f8fbfd !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
 
div[role="radiogroup"] label:hover {
    border-color: #7db9d3 !important;
    background-color: #eff8fb !important;
    box-shadow: 0 4px 12px rgba(125, 185, 211, 0.15) !important;
}
 
div[role="radiogroup"] label:has(input:checked) {
    border: 2px solid #7db9d3 !important;
    background-color: #cfe8f3 !important;
    font-weight: 800 !important;
    color: #111827 !important;
    box-shadow: 0 4px 16px rgba(125, 185, 211, 0.2) !important;
}
 
.summary-card {
    background-color: #ffffff;
    border: 1px solid #d9e2e8;
    border-radius: 22px;
    padding: 30px;
    box-shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
    margin-bottom: 26px;
}
 
.summary-title { font-size: 22px; font-weight: 800; margin-bottom: 22px; color: #111827; }
 
.summary-grid {
    background-color: #f5f8fa;
    border-radius: 18px;
    padding: 24px;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 18px 38px;
}
 
.summary-grid span { display: block; font-size: 13px; color: #6b7280; margin-bottom: 6px; }
.summary-grid b { font-size: 15px; color: #111827; font-weight: 700; }
 
div[role="radiogroup"] label p { font-size: 15px; color: #111827; }
 
.diagnosis-guide {
    background-color: #ffffff;
    border-radius: 22px;
    padding: 26px 30px;
    border: none;
    margin-bottom: 20px;
}
 
header[data-testid="stHeader"] { background: rgba(255, 255, 255, 0.95); }
hr { display: none; }
 
div[role="radiogroup"] input[type="radio"] { display: none; }
 
div[role="radiogroup"] label:has(input:checked) {
    border: 2px solid #7db9d3;
    background-color: #cfe8f3;
    font-weight: 800;
}
 
div[role="radiogroup"] label > div:first-child { display: none !important; }
 
div[role="radiogroup"] label {
    justify-content: center !important;
    text-align: center !important;
}
 
.question-row div[role="radiogroup"] label {
    justify-content: flex-start !important;
    text-align: left !important;
}
 
.question-text {
    font-size: 17px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 10px;
    line-height: 1.6;
}
 
.question-row div[role="radiogroup"] {
    display: inline-flex !important;
    gap: 20px !important;
    margin-left: 12px;
}
 
.question-row div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    min-height: auto !important;
    width: auto !important;
    padding: 0 !important;
    border-radius: 0 !important;
    border: none !important;
    background-color: transparent !important;
    justify-content: flex-start !important;
    text-align: left !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    gap: 6px !important;
    cursor: pointer !important;
}
 
.question-row div[role="radiogroup"] label:hover { background-color: transparent !important; }
 
.question-row div[role="radiogroup"] label:has(input:checked) {
    border: none !important;
    background-color: transparent !important;
    font-weight: 600 !important;
}
 
.question-row div[role="radiogroup"] input[type="radio"] {
    display: inline-block !important;
    width: 18px !important;
    height: 18px !important;
    margin: 0 !important;
    accent-color: #7db9d3 !important;
    cursor: pointer !important;
}
 
.rp-page-header { margin-bottom: 2rem; }
.rp-page-header h1 {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
    margin: 0 0 8px;
    letter-spacing: -0.02em;
    font-family: 'Noto Sans KR', sans-serif;
}
.rp-page-header p { font-size: 15px; color: #6b7280; margin: 0; line-height: 1.7; }
 
.rp-section-label {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #111827;
    margin: 0 0 14px;
}
 
.rp-alert {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #f0faf5;
    border: 1px solid #a7d7bc;
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 1.5rem;
}
.rp-alert-icon { font-size: 20px; color: #1d9e75; flex-shrink: 0; }
.rp-alert-text { font-size: 15px; color: #085041; line-height: 1.6; }
.rp-alert-text strong { font-weight: 800; }
 
.rp-card {
    background: #ffffff;
    border: 1px solid #e5eaef;
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
 
.rp-meta-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}
@media (max-width: 900px) {
    .rp-meta-grid { grid-template-columns: repeat(2, 1fr); }
}
.rp-meta-item {
    background: #f8fbfd;
    border-radius: 14px;
    padding: 14px 16px;
    border: 1px solid #e5eaef;
}
.rp-meta-label { font-size: 11px; color: #6b7280; margin-bottom: 6px; font-weight: 600; letter-spacing: 0.04em; }
.rp-meta-value { font-size: 14px; font-weight: 700; color: #111827; line-height: 1.4; }
 
.rp-rec-list { list-style: none; margin: 0; padding: 0; }
.rp-rec-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px solid #f0f4f8;
}
.rp-rec-item:last-child { border-bottom: none; padding-bottom: 0; }
.rp-rec-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: #f0f7fc;
    border: 1px solid #d9edf7;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 17px;
}
.rp-rec-text { font-size: 14px; color: #374151; line-height: 1.6; }
 
html, body [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
</style>
""", unsafe_allow_html=True)
 
 
# ——————————————
# Session State
# ——————————————
if "page" not in st.session_state:
    st.session_state.page = "company_input"
 
if "company_info" not in st.session_state:
    st.session_state.company_info = {}
 
 
# -----------------------------
# Page 1: 기업 정보 입력
# -----------------------------
def company_input_page():
    st.markdown(
        '''<div class="header-banner">
               <div class="header-inner">
                    <div class="banner-title">스마트공장 의사결정 지원 AI</div>
                    <div class="banner-subtitle">스마트공장 도입이 처음인 중소기업을 대상으로 기업 정보를 기반으로 맞춤형 의사결정을 지원합니다.</div>
                </div>
        </div>''',
        unsafe_allow_html=True
    )
 
    st.markdown('<div class="section-title">사용자의 기업 정보를 입력해주세요</div>', unsafe_allow_html=True)
 
    with st.form("company_form"):
        st.markdown('<div class="input-label">업종을 선택해주세요</div>', unsafe_allow_html=True)
        industry = st.radio(
            "업종",
            ["식품 제조업", "섬유 제조업", "금속 제조업", "통신장비 제조업", "기타"],
            index=None, horizontal=True, label_visibility="collapsed"
        )
 
        st.markdown('<div class="input-label">기업 규모를 선택해주세요</div>', unsafe_allow_html=True)
        company_size = st.radio(
            "기업 규모",
            ["10인 미만", "10~49인", "50~99인", "100인 이상"],
            index=None, horizontal=True, label_visibility="collapsed"
        )
 
        st.markdown('<div class="input-label">연매출을 선택해주세요</div>', unsafe_allow_html=True)
        revenue = st.radio(
            "연매출",
            ["10억 미만", "10~30억", "30~50억", "50~100억", "100억 이상", "잘 모르겠음"],
            index=None, horizontal=True, label_visibility="collapsed"
        )
 
        st.markdown('<div class="input-label">운영 중 가장 고민되는 부분을 선택해주세요</div>', unsafe_allow_html=True)
        concerns = st.multiselect(
            "고민 부분",
            ["품질/불량", "위생/추적성", "납기", "재고", "원가", "생산성", "인력 부족", "기타"],
            placeholder="선택해주세요", label_visibility="collapsed"
        )
 
        st.markdown('<div class="input-label">현재 사용 중인 관리 방식 또는 시스템을 선택해주세요</div>', unsafe_allow_html=True)
        current_system = st.multiselect(
            "관리 방식",
            ["수기 관리", "엑셀", "ERP", "MES", "POP", "바코드/QR", "센서/IoT", "없음"],
            placeholder="선택해주세요", label_visibility="collapsed"
        )
 
        st.markdown('<div class="input-label">현재 상황 또는 추가 설명</div>', unsafe_allow_html=True)
        detail = st.text_area(
            "설명",
            placeholder="예: 생산 실적은 엑셀로 관리하고 있고, 불량 원인 추적이 어렵습니다.",
            height=120, label_visibility="collapsed"
        )
 
        submitted = st.form_submit_button("기업 정보 입력 완료", use_container_width=True)
 
    st.markdown('</div>', unsafe_allow_html=True)
 
    if submitted:
        st.session_state.company_info = {
            "industry": industry,
            "company_size": company_size,
            "revenue": revenue,
            "concerns": concerns,
            "current_system": current_system,
            "detail": detail
        }
        st.session_state.page = "confirm_test"
        st.rerun()
 
 
# -----------------------------
# Page 2: 자가진단 진행 여부 확인
# -----------------------------
def confirm_test_page():
    info = st.session_state.company_info
 
    st.markdown('<div class="summary-title">입력된 기업 정보</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="summary-grid">
        <div><span>업종</span><b>{industry}</b></div>
        <div><span>기업 규모</span><b>{company_size}</b></div>
        <div><span>연매출</span><b>{revenue}</b></div>
        <div><span>주요 고민</span><b>{concerns}</b></div>
        <div><span>현재 시스템</span><b>{current_system}</b></div>
        <div><span>추가 설명</span><b>{detail}</b></div>
    </div>
    </div>
    """.format(
        industry=info.get("industry", "-"),
        company_size=info.get("company_size", "-"),
        revenue=info.get("revenue", "-"),
        concerns=", ".join(info.get("concerns", [])) if info.get("concerns") else "-",
        current_system=", ".join(info.get("current_system", [])) if info.get("current_system") else "-",
        detail=info.get("detail", "-") if info.get("detail") else "-"
    ), unsafe_allow_html=True)
 
    st.markdown("""
    <div class="diagnosis-guide">
         <h3>자가진단 테스트를 진행하시겠습니까?</h3>
         <p>입력한 기업 정보에 더해 생산정보 관리, 실시간 모니터링, 데이터 분석, 자동화 수준 등을 확인하여
         현재 기업이 어느 Level 영역의 특성과 가까운지 참고 수준으로 분석합니다.</p>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 기업 정보 수정하기", use_container_width=True):
            st.session_state.page = "company_input"
            st.rerun()
    with col2:
        if st.button("자가진단 테스트 실행 →", type="primary", use_container_width=True):
            st.session_state.page = "self_check"
            st.rerun()
 
 
# -----------------------------
# Page 3: 자가진단 테스트
# -----------------------------
def self_check_page():
    st.markdown('<div class="main-title">자가진단 기반 기업 특성 분석</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">아래 문항에 응답하면 현재 기업의 스마트공장 수준 영역을 참고 형태로 추정합니다.</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="section-title">자가진단 문항</div>', unsafe_allow_html=True)
 
    questions = [
        {"level": 1, "text": "생산, 입출고, 자재 정보를 주로 수기 또는 엑셀로 관리하고 있다."},
        {"level": 1, "text": "거래처 또는 고객사와의 협업이 전화, 이메일 중심으로 이루어진다."},
        {"level": 2, "text": "생산 실적, 불량률, 입출고 정보를 시스템으로 집계할 수 있다."},
        {"level": 2, "text": "MES, POP, ERP 등 일부 관리 시스템을 사용하고 있다."},
        {"level": 3, "text": "설비, 센서, PLC 등에서 데이터를 자동으로 수집하고 있다."},
        {"level": 3, "text": "생산 현장과 사무실 간 실시간 모니터링이 가능하다."},
        {"level": 4, "text": "공정 데이터를 기반으로 실시간 제어 또는 의사결정 최적화가 가능하다."},
        {"level": 4, "text": "제품 개발, 생산, 구매, 영업 등 주요 기능이 통합적으로 운영된다."},
        {"level": 5, "text": "IoT, CPS, AI 등을 활용하여 생산·제조·운영이 자율적으로 관리된다."},
    ]
 
    answers = {}
 
    with st.form("self_check_form"):
        for i, q in enumerate(questions, start=1):
            st.markdown('<div class="question-row">', unsafe_allow_html=True)
            col1, col2 = st.columns([8, 2])
            with col1:
                st.markdown(f'<div class="question-text">Q{i}. {q["text"]}</div>', unsafe_allow_html=True)
            with col2:
                answers[i] = st.radio(
                    label=f"Q{i}",
                    options=["예", "아니오"],
                    horizontal=True,
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
 
        submitted = st.form_submit_button("진단 결과 확인", type="primary", use_container_width=True)
 
    if submitted:
        scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        yes_questions = []
        for i, q in enumerate(questions, start=1):
            if answers[i] == "예":
                scores[q["level"]] += 1
                yes_questions.append(q["text"])
 
        max_score = max(scores.values())
        likely_levels = [lvl for lvl, score in scores.items() if score == max_score and score > 0]
 
        if not likely_levels:
            result_level = "Level 1 이전"
            result_text  = "현재 응답 기준으로 ICT 미적용 또는 Level 1 이전 단계의 특성이 관찰됩니다."
        elif len(likely_levels) == 1:
            result_level = f"Level {likely_levels[0]}"
            result_text  = f"현재 응답 기준으로 Level {likely_levels[0]} 영역의 특성이 가장 많이 관찰됩니다."
        else:
            result_level = f"Level {likely_levels[0]}~{likely_levels[-1]}"
            result_text  = f"현재 응답 기준으로 Level {likely_levels[0]}~{likely_levels[-1]} 영역의 특성이 함께 관찰됩니다."
 
        st.session_state.diagnosis_result = {
            "scores":        scores,
            "result_text":   result_text,
            "result_level":  result_level,
            "yes_questions": yes_questions,   # ← '예' 답한 문항 저장
        }
        st.session_state.page = "result"
        st.rerun()
 
 
# -----------------------------
# Page 4: 결과 화면
# -----------------------------
def result_page():
    info   = st.session_state.company_info
    result = st.session_state.diagnosis_result
 
    industry      = info.get("industry", "-")
    company_size  = info.get("company_size", "-")
    concerns_list = info.get("concerns", [])
    system_list   = info.get("current_system", [])
    concerns_str  = ", ".join(concerns_list) if concerns_list else "-"
    system_str    = ", ".join(system_list)   if system_list   else "-"
 
    scores       = result["scores"]
    result_text  = result["result_text"]
    max_score    = max(scores.values())
 
    st.markdown("""
    <div class="rp-page-header">
        <h1>진단 결과 및 맞춤형 추천</h1>
        <p>자가진단 결과와 기업 정보를 기반으로 관련 기술·정책·사례 정보를 추천합니다.</p>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown(f"""
    <div class="rp-alert">
        <span class="rp-alert-text">{result_text}</span>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown('<p class="rp-section-label">Level별 응답 분포</p>', unsafe_allow_html=True)
 
    chart_data = pd.DataFrame({
        "Level": ["1", "2", "3", "4", "5"],
        "점수":  list(scores.values()),
        "최고":  [s == max_score and s > 0 for s in scores.values()]
    })
 
    y_max = max(scores.values()) + 1 if max(scores.values()) > 0 else 3
 
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Level:N", axis=alt.Axis(labelAngle=0, title="Level", labelFontSize=13)),
            y=alt.Y("점수:Q", axis=alt.Axis(title="응답 수", tickMinStep=1),
                    scale=alt.Scale(domain=[0, y_max])),
            color=alt.condition(
                alt.datum["최고"] == True,
                alt.value("#378ADD"),
                alt.value("#B5D4F4")
            ),
            tooltip=[
                alt.Tooltip("Level:N", title="Level"),
                alt.Tooltip("점수:Q", title="응답 수")
            ]
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)
 
    st.markdown(f"""
    <div class="rp-card">
        <p class="rp-section-label">기업 특성</p>
        <div class="rp-meta-grid">
            <div class="rp-meta-item">
                <div class="rp-meta-label">업종</div>
                <div class="rp-meta-value">{industry}</div>
            </div>
            <div class="rp-meta-item">
                <div class="rp-meta-label">기업 규모</div>
                <div class="rp-meta-value">{company_size}</div>
            </div>
            <div class="rp-meta-item">
                <div class="rp-meta-label">주요 고민</div>
                <div class="rp-meta-value">{concerns_str}</div>
            </div>
            <div class="rp-meta-item">
                <div class="rp-meta-label">현재 관리 방식</div>
                <div class="rp-meta-value">{system_str}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("""
    <div class="rp-card">
        <p class="rp-section-label">추천 방향</p>
        <ul class="rp-rec-list">
            <li class="rp-rec-item">
                <div class="rp-rec-icon">🗄️</div>
                <div class="rp-rec-text">생산 데이터 자동 수집 체계 구축 검토</div>
            </li>
            <li class="rp-rec-item">
                <div class="rp-rec-icon">🏭</div>
                <div class="rp-rec-text">MES 또는 POP 기반 생산실적 관리 사례 탐색</div>
            </li>
            <li class="rp-rec-item">
                <div class="rp-rec-icon">📊</div>
                <div class="rp-rec-text">품질/불량 이력 데이터 관리 체계 구축</div>
            </li>
            <li class="rp-rec-item">
                <div class="rp-rec-icon">📄</div>
                <div class="rp-rec-text">스마트공장 구축 지원사업 및 수준확인 사업 정보 확인</div>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2 = st.columns([1, 1.6])
    with col1:
        if st.button("← 처음으로 돌아가기", use_container_width=True):
            st.session_state.page = "company_input"
            st.rerun()
    with col2:
        if st.button("RAG 기반 상담 시작하기 →", type="primary", use_container_width=True):
            st.session_state.page = "rag_chat"
            st.rerun()
 
 
# -----------------------------
# Page 5: RAG 챗봇  ← 핵심 연결 부분
# -----------------------------
def rag_chat_page():
    st.markdown('<div class="main-title">스마트공장 도입 지원 챗봇</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">입력한 기업 정보와 자가진단 결과를 기반으로 맞춤형 상담을 제공합니다.</div>',
        unsafe_allow_html=True
    )
 
    company_info     = st.session_state.get("company_info", {})
    diagnosis_result = st.session_state.get("diagnosis_result", {})
 
    st.markdown("""
    <div class="chat-info-box">
        <b>상담에 반영되는 정보</b><br>
        기업 기본정보, 주요 고민, 현재 시스템, 자가진단 결과가 함께 반영됩니다.
    </div>
    """, unsafe_allow_html=True)
 
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
 
    user_question = st.chat_input("스마트공장 도입과 관련해 궁금한 내용을 입력해주세요.")
 
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
 
        with st.chat_message("assistant"):
            with st.spinner("문서를 검색하고 답변을 생성 중..."):
 
                # ── 1~4단계에서 입력한 정보를 profile 로 구성
                profile = {
                    "industry":  company_info.get("industry", "미입력"),
                    "size":      company_info.get("company_size", "미입력"),
                    "pain":      ", ".join(company_info.get("concerns", [])) or "미입력",
                    "process":   company_info.get("detail", "미입력") or "미입력",
                    # 자가진단 결과
                    "level":     diagnosis_result.get("result_level", "미입력"),
                    "level_category":             "",
                    "level_category_explanation": diagnosis_result.get("result_text", ""),
                    "question":  ", ".join(diagnosis_result.get("yes_questions", [])) or "미입력",
                }
 
                # ── RAG 검색 + 답변 생성
                docs   = retrieve_context(user_question)
                answer = ask_rag(user_question, docs, profile=profile)
 
            st.write(answer)
 
            # 근거 발췌문 보기
            with st.expander("근거(발췌문/출처) 보기"):
                if docs:
                    for i, d in enumerate(docs, start=1):
                        meta   = d.metadata or {}
                        source = meta.get("source_file", meta.get("source", "unknown"))
                        st.markdown(f"**[{i}]** {source}")
                        st.write(d.page_content[:800])
                else:
                    st.write("(문서 검색 결과 없음)")
 
        st.session_state.messages.append({"role": "assistant", "content": answer})
 
 
# -----------------------------
# Router
# -----------------------------
if st.session_state.page == "company_input":
    company_input_page()
elif st.session_state.page == "confirm_test":
    confirm_test_page()
elif st.session_state.page == "self_check":
    self_check_page()
elif st.session_state.page == "result":
    result_page()
elif st.session_state.page == "rag_chat":
    rag_chat_page()