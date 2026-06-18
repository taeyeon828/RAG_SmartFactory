"""
main.py  ―  스마트공장 RAG 시스템 (전처리 + 추론 통합)

사용법:
  전처리 (최초 1회 또는 데이터 변경 시):
      python main.py --preprocess

  DB 연결 테스트:
      python main.py --test-db

  앱 실행 (평소):
      streamlit run app.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np

# =====================================
# 경로 설정
# =====================================
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"

PDF_DIR  = DATA_DIR / "raw" / "pdf"
CSV_DIR  = DATA_DIR / "raw" / "csv"
IMG_DIR  = DATA_DIR / "raw" / "images"

PROCESSED_DIR        = DATA_DIR / "processed"
PROCESSED_TEXT_DIR   = PROCESSED_DIR / "text"
PROCESSED_CHUNKS_DIR = PROCESSED_DIR / "chunks"
CHUNKS_JSONL_PATH    = PROCESSED_CHUNKS_DIR / "chunks.jsonl"
REPORT_DIR           = PROCESSED_DIR / "report"
QUALITY_REPORT_PATH  = REPORT_DIR / "quality_report.json"

CHROMA_DIR = BASE_DIR / "data" / "vectorstore" / "faiss_db"

# =====================================
# 청킹 파라미터
# =====================================
CHUNK_SIZE    = 600
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.15)   # 15% 고정

# =====================================
# MMR / Reranker 파라미터
# =====================================
MMR_K       = 5
MMR_FETCH_K = 15
RERANK_TOP  = 3

# =====================================
# 임베딩 / 컬렉션
# =====================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "smart_factory"

# =====================================
# LLM 초기화 (전역) - Google Generative AI
# =====================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0,
)
# =====================================
# LangChain Document
# =====================================
from langchain_core.documents import Document

# =====================================
# 공개 인터페이스 선언
# =====================================
__all__ = ["retrieve_context", "ask_rag", "load_vectorstore", "test_db_connection"]


# =====================================================================
# STEP 0 ― PostgreSQL 연결 테스트
# =====================================================================
def test_db_connection(db_url: str | None = None) -> bool:
    """
    DB 연결 가능 여부만 확인.
    db_url 미전달 시 환경변수 DB_URL 사용.
    """
    from sqlalchemy import create_engine, text

    url = db_url or os.getenv("DB_URL", "")
    if not url:
        print("[DB] DB_URL 환경변수가 설정되지 않았습니다.")
        return False
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[DB] 연결 성공 ✓")
        return True
    except Exception as e:
        print(f"[DB] 연결 실패: {e}")
        return False


# =====================================================================
# STEP 1 ― 전처리 (--preprocess 플래그 시에만 실행)
# =====================================================================

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufeff", "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_csv_value(v) -> str:
    v = str(v).replace("\ufeff", "").strip()
    return re.sub(r"\s+", " ", v)


def _preprocess_pdfs() -> list[Document]:
    """
    PDF → 텍스트 정제 → RecursiveCharacterTextSplitter 청킹
    → chunks.jsonl 저장 → Document 리스트 반환
    --preprocess 시 1회만 실행됨
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "다. ", " "],
    )

    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "files": []}
    docs: list[Document] = []

    with CHUNKS_JSONL_PATH.open("w", encoding="utf-8") as fout:
        for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
            loader = PyPDFLoader(str(pdf_path))
            pages  = loader.load()

            file_stat = {
                "file": pdf_path.name,
                "pages": len(pages),
                "empty_pages": 0,
                "chunks": 0,
            }

            file_text_dir = PROCESSED_TEXT_DIR / pdf_path.stem
            file_text_dir.mkdir(parents=True, exist_ok=True)

            cleaned_pages: list[tuple[int, str]] = []
            for i, page in enumerate(pages, start=1):
                cleaned = _clean_text(page.page_content or "")
                if len(cleaned) < 10:
                    file_stat["empty_pages"] += 1
                (file_text_dir / f"page_{i:03d}.txt").write_text(cleaned, encoding="utf-8")
                cleaned_pages.append((i, cleaned))

            for page_no, text in cleaned_pages:
                if not text:
                    continue
                splits = splitter.create_documents(
                    [text],
                    metadatas=[{
                        "doc_type":    "pdf",
                        "source_type": "pdf",
                        "source_file": pdf_path.name,
                        "source":      pdf_path.name,
                        "page":        page_no,
                    }],
                )
                for j, d in enumerate(splits, start=1):
                    rec = {
                        "chunk_id": f"{pdf_path.stem}_p{page_no:03d}_c{j:03d}",
                        "text":     d.page_content,
                        **d.metadata,
                        "char_len": len(d.page_content),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    file_stat["chunks"] += 1
                    docs.append(Document(page_content=d.page_content, metadata=d.metadata))

            report["files"].append(file_stat)
            print(f"  [PDF] {pdf_path.name}: {file_stat['chunks']} chunks")

    QUALITY_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return docs


def _load_pdf_chunks_from_jsonl() -> list[Document]:
    """전처리 완료된 chunks.jsonl → Document 리스트 (앱 실행 시 사용)"""
    docs: list[Document] = []
    with CHUNKS_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec  = json.loads(line)
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            meta = {k: v for k, v in rec.items() if k != "text"}
            meta.setdefault("doc_type",    "pdf")
            meta.setdefault("source_type", "pdf")
            meta.setdefault("source_file", meta.get("source", "unknown"))
            if isinstance(meta.get("page"), int):
                meta["page_human"] = meta["page"]
            docs.append(Document(page_content=text, metadata=meta))
    return docs


def _load_csvs() -> list[Document]:
    """CSV → Document 리스트 (중복 제거 포함)"""
    docs: list[Document] = []
    seen: set[tuple]     = set()

    for csv_path in sorted(CSV_DIR.glob("*.csv")):
        try:
            f_open = open(csv_path, "r", encoding="utf-8", errors="ignore")
        except Exception:
            f_open = open(csv_path, "r", encoding="cp949", errors="ignore")

        with f_open:
            reader = csv.DictReader(f_open)
            for idx, row in enumerate(reader, start=1):
                cleaned = {
                    k: _clean_csv_value(v)
                    for k, v in row.items()
                    if k and v and _clean_csv_value(v)
                }
                text = "\n".join(f"[{k}] {v}" for k, v in cleaned.items()).strip()
                if not text:
                    continue
                sig = (csv_path.name, text)
                if sig in seen:
                    continue
                seen.add(sig)
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "doc_type":    "csv",
                        "source_type": "csv",
                        "source_file": csv_path.name,
                        "source":      csv_path.name,
                        "row":         idx,
                    },
                ))
    return docs


def _load_images() -> list[Document]:
    """이미지 → EasyOCR → Document 리스트"""
    try:
        import cv2
        import easyocr
    except ImportError:
        print("[WARN] easyocr / opencv 미설치 → 이미지 로딩 생략")
        return []

    reader = easyocr.Reader(["ko", "en"])
    docs: list[Document] = []

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for img_path in sorted(IMG_DIR.glob(ext)):
            try:
                data     = np.fromfile(str(img_path), dtype=np.uint8)
                img      = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                texts    = reader.readtext(img, detail=0)
                ocr_text = "\n".join(texts).strip()
                if not ocr_text:
                    continue
                docs.append(Document(
                    page_content=ocr_text,
                    metadata={
                        "doc_type":    "image",
                        "source_type": "image",
                        "source_file": img_path.name,
                        "source":      img_path.name,
                    },
                ))
            except Exception as e:
                print(f"[WARN] 이미지 처리 실패 {img_path.name}: {e}")
    return docs


def run_preprocess():
    print("=" * 50)
    print("전처리 시작")
    print("=" * 50)

    print("\n[1/3] 데이터 로딩 중...")
    pdf_docs   = _preprocess_pdfs()
    csv_docs   = _load_csvs()
    image_docs = [] # Streamlit Cloud 배포에서는 EasyOCR 생략

    print(f"  PDF chunks : {len(pdf_docs)}")
    print(f"  CSV docs   : {len(csv_docs)}")
    print(f"  Image docs : {len(image_docs)}")

    all_docs = pdf_docs + csv_docs + image_docs
    if not all_docs:
        print("[ERROR] 로딩된 문서가 없습니다. data/raw 경로를 확인하세요.")
        sys.exit(1)

    print("\n[2/3] FAISS 적재 중...")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(str(CHROMA_DIR))

    print(f"\n[3/3] 완료 → {CHROMA_DIR}")
    print(f"      총 {len(all_docs)}개 청크 저장됨")
    print("=" * 50)


# =====================================================================
# STEP 2 ― 벡터스토어 로드 (앱 실행 시 1회)
# =====================================================================

def load_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise RuntimeError(
            f"FAISS DB not found at {CHROMA_DIR}\n"
            "먼저 'python main.py --preprocess' 를 실행하세요."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.load_local(
        str(CHROMA_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# =====================================================================
# STEP 3 ― Query Rewriting (전역 llm 사용)
# =====================================================================

_REWRITE_TEMPLATE = """\
당신은 스마트공장 도메인 전문가입니다.
아래 질문을 벡터DB 검색에 최적화된 형태로 재작성하세요.

우선순위:
1. 질문자의 현재 수준(미적용/기초/중간/고도화)에 맞는 내용을 찾을 수 있도록 재작성하세요.
2. 해당 수준 자료가 부족할 경우 유사 사례나 관련 개념으로 확장하여 재작성하세요.
3. 재작성된 질문만 출력하고 설명은 하지 마세요.

원본 질문: {query}
재작성:"""


def rewrite_query(query: str) -> str:
    """
    전역 llm 으로 구어체 질문 → 도메인 용어 변환.
    예: '위생 좋아져?' → '식품 제조업 HACCP 자동화 위생 추적성 사례'
    """
    try:
        prompt    = _REWRITE_TEMPLATE.format(query=query)
        result    = llm.invoke(prompt)
        rewritten = result.content.strip() if hasattr(result, "content") else str(result).strip()
        print(f"[QueryRewrite] {query!r} → {rewritten!r}")
        return rewritten if rewritten else query
    except Exception as e:
        print(f"[QueryRewrite] 실패, 원본 사용: {e}")
        return query


# =====================================================================
# STEP 4 ― MMR 검색 + Reranker
# =====================================================================

def retrieve_context(query: str) -> list[Document]:
    """
    app.py 가 호출하는 공개 인터페이스.

    파이프라인:
      1) Query Rewriting  (전역 llm 사용)
      2) MMR 검색          (fetch_k=MMR_FETCH_K → top k=MMR_K)
      3) Reranker         (상위 RERANK_TOP 개만 LLM 에 전달)

    반환: Document 리스트 (metadata 에 doc_type, source_file 포함)
    """
    # 1) Query Rewriting
    search_query = rewrite_query(query)

    # 2) MMR 검색
    vectorstore   = load_vectorstore()
    mmr_docs: list[Document] = vectorstore.max_marginal_relevance_search(
        search_query,
        k=MMR_K,
        fetch_k=MMR_FETCH_K,
    )

    if not mmr_docs:
        return []

    # 3) Reranker
    try:
        from sentence_transformers import CrossEncoder
        reranker      = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs         = [(query, d.page_content) for d in mmr_docs]
        scores        = reranker.predict(pairs)
        ranked        = sorted(zip(scores, mmr_docs), key=lambda x: x[0], reverse=True)
        reranked_docs = [doc for _, doc in ranked[:RERANK_TOP]]
        print(f"[Reranker] {len(mmr_docs)} → {len(reranked_docs)} 청크")
        return reranked_docs
    except Exception as e:
        print(f"[Reranker] 실패, MMR 결과 그대로 사용: {e}")
        return mmr_docs[:RERANK_TOP]


# =====================================================================
# STEP 5 ― 프롬프트 + LLM 답변 생성
# =====================================================================

def _format_source(meta: dict) -> str:
    stype = meta.get("source_type") or meta.get("doc_type", "")
    if stype == "pdf":
        return f"{meta.get('source_file')} p.{meta.get('page_human', meta.get('page', '?'))}"
    if stype == "csv":
        return f"{meta.get('source_file')} row.{meta.get('row', '?')}"
    if stype == "image":
        return f"{meta.get('source_file')} (image)"
    if stype == "sql":
        return f"DB / {meta.get('table', '')} row.{meta.get('row', '?')}"
    return meta.get("source_file", "unknown")


def _make_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        src = _format_source(d.metadata)
        parts.append(f"{d.page_content}\n(출처: {src})")
    return "\n\n---\n\n".join(parts)


def _pick_mode(docs: list[Document], query: str) -> str:
    q = query.lower()

    csv_terms = ["공급", "공급기업", "공급 기업", "기업", "업종", "제공", "전문 기술", "전문기술", "키워드"]
    if any(t in q for t in csv_terms):
        if any(d.metadata.get("doc_type") == "csv" for d in docs):
            return "csv"

    pdf_terms = ["사례", "성공", "도입", "절차", "단계", "어떻게", "방법", "효과", "개념"]
    if any(t in q for t in pdf_terms):
        if any(d.metadata.get("doc_type") == "pdf" for d in docs):
            return "pdf"

    counts: dict[str, int] = defaultdict(int)
    for d in docs:
        counts[d.metadata.get("doc_type", "pdf")] += 1
    return max(counts, key=counts.get) if counts else "pdf"


def _build_prompt_pdf(query: str, context: str, profile: dict, suggest_questions: bool = False) -> str:
    industry                   = profile.get("industry", "미입력")
    size                       = profile.get("size", "미입력")
    pain                       = profile.get("pain", "미입력")
    process                    = profile.get("process", "미입력")
    level                      = profile.get("level", "미입력")
    level_category             = profile.get("level_category", "미입력")
    level_category_explanation = profile.get("level_category_explanation", "미입력")
    question                   = profile.get("question", "미입력")

    prompt = f"""
너는 스마트공장 도입을 준비하는 중소기업을 위한 '맞춤형 의사결정 지원 컨설턴트 AI'야.

[ 절대 규칙 - 반드시 먼저 읽고 따를 것 ]
1. [발췌문] 안에 있는 문장만 사용해.
   발췌문 밖의 내용을 단 한 글자도 추가하지 마.
   숫자, 퍼센트, 고유명사, 날짜는 반드시 발췌문에서 그대로 가져와야 한다.
2. 발췌문에 없는 내용이 필요하면 그 부분은 반드시 생략해.
   일반 지식, 상식, 추론으로 보완하는 것은 엄격히 금지한다.
   발췌문에 근거가 전혀 없으면 "제공된 문서에서 해당 내용을 찾지 못했습니다."라고만 답해.
3. 답변의 모든 문장 끝에는 반드시 출처를 표기해. (예: (출처: 파일명 p.쪽))
3. 답변의 모든 문장 끝에는 반드시 출처를 표기해. (예: (출처: 파일명 p.쪽))
4. 전문 용어보다는 중소기업 사장님이 이해하기 쉬운 현장 용어를 사용해.
5. [사용자 기업 정보]의 현재 수준({level})에 맞는 정보만 선별해. (예: Level 0 기업에게 Level 4 기술 제안 금지)

[ 발췌문 ]
{context}

[ 질문 ]
{query}

[ 사용자 기업 정보 ]
- 업종: {industry}
- 규모: {size}
- 주요 고민: {pain}
- 공정 특징: {process}
- 자가진단 결과 수준: {level}
- 해당 수준 분류: {level_category}
- 현재 수준 상세 설명: {level_category_explanation}
- 수준 판정 기준(예라고 답한 항목): {question}

[ 질문 유형별 응답 구조 ]
질문 유형에 따라 아래 구조 중 하나를 선택해.
도입 배경은 현장 상황 중심으로 설명해 (누가, 어디서, 어떤 문제가 반복됐는지).

[ 유형 1: 판단/의사결정 요구 시 ]
- 결론 요약: 질문에 대한 직접적인 답 (현재 {level} 수준에 맞는 결론)
- 근거(왜냐하면): 발췌문 기반의 타당한 이유
- 맞춤형 의사결정 지원: 현재 {level_category} 단계에서 다음 단계로 넘어가기 위해 참고할 정책/지원사업/필요 기술
- 현장 적용 포인트: 우리 공장에 당장 적용할 수 있는 첫걸음

[ 유형 2: 사례/이해 요구 시 ]
- 도입 배경: 구체적 현장 상황 (누가, 어디서, 어떤 문제가 있었는지)
- 변화 과정: 기술적 설명보다 구조 및 프로세스 변화 중심
- 결과: 도입 후의 성과
- 상황 적용: "만약 현재 {level} 수준인 우리 공장에 적용한다면 지금 당장 생각해볼 점" (반드시 이 문장으로 시작)

[ 유형 3: 개념 설명 요구 시 ]
- 개념의 정의: 현장 언어로 쉽게 설명
- 역할: 제조 현장에서 어떤 역할을 하는지
- 기대 효과: 현재 {level} 수준의 기업이 도입했을 때 얻을 수 있는 구체적 이점
""".strip()

    if suggest_questions:
        prompt += f"""

[ 다음 질문 제안 ]
답변을 마친 후, 중소기업 사장님 관점에서 다음에 이어질 법한 질문 2~4개를 제안해.
- 반드시 위 발췌문에 근거할 것
- 질문 유형(개념/사례/절차/의사결정)이 겹치지 않게 할 것
- 스마트공장을 처음 도입하는 중소기업 사장 전제 유지
""".strip()

    return prompt


def _build_prompt_csv(query: str, context: str) -> str:
    return f"""
너는 CSV 근거만으로 답한다.
CONTEXT에 있는 사실만 사용하고, 정의/효과/역할/추측/일반상식 설명은 절대 하지 마라.

[출력 규칙]
1) CONTEXT에 확인된 기술 키워드를 추출하라.
2) 질문에 명시된 업종에 해당하는 정보만 추출하라.
3) 동일/유사 기술은 3~5개 카테고리로 묶어 재정리하라 (새 정보 추가 금지).
4) 각 카테고리 끝에, 해당 키워드가 등장한 출처를 "(출처: ... row.xxx)" 형식으로 표기하라.
5) CONTEXT에 없는 내용은 "제공된 데이터에서 해당 정보를 찾을 수 없습니다."라고 답한다.
6) 마지막에는 Markdown 표 형식으로 요약하라. (열 구성: 회사명 | 적용 업종 | 제공 기술)

답변을 마친 후, 아래 조건을 만족하는 "다음에 할 수 있는 질문 예시"를 2~4개 제안해.
[질문 제안 규칙]
- 반드시 제공된 발췌문(CSV)에 근거해 제안할 것
- 새로운 기술이나 개념을 추가하지 말 것
- 질문은 데이터 조회/비교/필터링 중심으로 구성할 것

질문: {query}
발췌문: {context}
""".strip()



def ask_rag(
    query: str,
    docs: list[Document],
    profile: dict | None = None,
    db_context: str = "",
    suggest_questions: bool = False,  # 평가 시 False 유지 / 앱 배포 후 app.py 호출부에서 True로 변경
) -> str:
    profile  = profile or {}
    mode     = _pick_mode(docs, query)
    use_docs = [d for d in docs if d.metadata.get("doc_type") == mode]
    if not use_docs:
        use_docs = docs

    context = _make_context(use_docs)

    if db_context:
        context = f"[DB 조회 결과]\n{db_context}\n\n---\n\n" + context

    if db_context:
        prompt = _build_prompt_pdf(query, context, profile, suggest_questions)
    elif mode == "csv":
        prompt = _build_prompt_csv(query, context)
    else:
        prompt = _build_prompt_pdf(query, context, profile, suggest_questions)

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# =====================================================================
# CLI 진입점
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="스마트공장 RAG 시스템")
    parser.add_argument("--preprocess", action="store_true", help="전처리 + ChromaDB 적재 (1회성)")
    parser.add_argument("--test-db",    action="store_true", help="PostgreSQL 연결 테스트")
    parser.add_argument("--chat",       action="store_true", help="터미널 채팅 모드")
    args = parser.parse_args()

    if args.test_db:
        test_db_connection()
        sys.exit(0)

    if args.preprocess:
        run_preprocess()
        sys.exit(0)
    
    if args.preprocess:
        run_preprocess()
        sys.exit(0)

    # ↓ 여기에 추가
    if args.chat:
        print("=" * 50)
        print("스마트공장 RAG 채팅 모드")
        print("종료하려면 'quit' 또는 'exit' 입력")
        print("=" * 50)
        while True:
            try:
                query = input("\n질문: ").strip()
            except KeyboardInterrupt:
                print("\n종료합니다.")
                break
            if query.lower() in ("quit", "exit", "종료"):
                print("종료합니다.")
                break
            if not query:
                continue
            docs   = retrieve_context(query)
            answer = ask_rag(query, docs)
            print("\n" + "=" * 50)
            print("답변:")
            print(answer)
            print("=" * 50)

    print("앱을 실행하려면 : streamlit run app.py")  # 기존 코드
    
    

    print("앱을 실행하려면 : streamlit run app.py")
    print("전처리를 실행하려면: python main.py --preprocess")
    print("DB 연결 테스트  : python main.py --test-db")
