import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# 1. 내 RAG 시스템 임포트
# ==========================================
try:
    import main as main_module
    from main import retrieve_context, ask_rag
except ImportError:
    print("[오류] main.py 파일을 찾을 수 없거나 함수를 불러올 수 없습니다.")
    print("이 스크립트를 main.py와 같은 폴더에 위치시켜주세요.")
    exit()

# ==========================================
# 2. RAGAS 평가자(Judge) 모델 및 임베딩 세팅
#    - RAGAS 0.2+ 신버전: LangchainLLMWrapper 사용
#    - 구버전 fallback 지원
# ==========================================
print("\n[System] 로컬 평가 모델(Ollama)을 설정합니다...")

_llm_raw = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0,
    num_ctx=4096,                 # 컨텍스트 길이 제한 (메모리 절약)
    timeout=300,                  # 타임아웃 5분으로 연장
    extra_body={"think": False},  # thinking 모드 비활성화 (속도 개선)
)
_emb_raw = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    evaluator_llm        = LangchainLLMWrapper(_llm_raw)
    evaluator_embeddings = LangchainEmbeddingsWrapper(_emb_raw)
    USE_WRAPPER = True
    print("[System] RAGAS 0.2+ 방식 (LangchainLLMWrapper) 사용")
except ImportError:
    evaluator_llm        = _llm_raw
    evaluator_embeddings = _emb_raw
    faithfulness.llm            = evaluator_llm
    answer_relevancy.llm        = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings
    context_precision.llm       = evaluator_llm
    context_recall.llm          = evaluator_llm
    USE_WRAPPER = False
    print("[System] RAGAS 구버전 방식 (직접 바인딩) 사용")

metrics     = [faithfulness, answer_relevancy, context_precision, context_recall]
metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# ==========================================
# 3. 내 RAG 파이프라인 연동 함수
# ==========================================
def run_my_rag(question: str):
    retrieved_docs = retrieve_context(question)
    if retrieved_docs:
        retrieved_contexts_list = [doc.page_content for doc in retrieved_docs]
    else:
        retrieved_contexts_list = []
    generated_answer = ask_rag(question, retrieved_docs)
    return generated_answer, retrieved_contexts_list


# ==========================================
# 4. 청크 사이즈 1개에 대한 평가 실행 함수
# ==========================================
def evaluate_for_chunk_size(chunk_size: int, df: pd.DataFrame, has_question_type: bool) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f" 🔧 청크 사이즈 {chunk_size} 평가 시작")
    print(f"{'='*55}")

    main_module.CHUNK_SIZE    = chunk_size
    main_module.CHUNK_OVERLAP = int(chunk_size * 0.15)
    main_module.CHROMA_DIR    = Path(f"C:/RAG_DB/faiss_db_{chunk_size}")
    main_module.CHUNKS_JSONL_PATH = (
        main_module.PROCESSED_CHUNKS_DIR / f"chunks_{chunk_size}.jsonl"
    )

    if not main_module.CHROMA_DIR.exists() or not any(main_module.CHROMA_DIR.iterdir()):
        print(f"[System] FAISS DB 없음 → 청크 사이즈 {chunk_size}로 전처리 시작...")
        main_module.run_preprocess()
    else:
        print(f"[System] 기존 FAISS DB 재사용: {main_module.CHROMA_DIR}")

    eval_data: dict = {
        "user_input":         [],
        "retrieved_contexts": [],
        "response":           [],
        "reference":          [],
    }
    if has_question_type:
        eval_data["question_type"] = []

    print(f"[System] RAG 파이프라인 실행 중 (총 {len(df)}개 질문)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        question     = row["question"]
        ground_truth = row["ground_truth"]
        gen_answer, ret_contexts = run_my_rag(question)
        eval_data["user_input"].append(question)
        eval_data["retrieved_contexts"].append(ret_contexts)
        eval_data["response"].append(gen_answer)
        eval_data["reference"].append(ground_truth)
        if has_question_type:
            eval_data["question_type"].append(row["question_type"])

    eval_df    = pd.DataFrame(eval_data)
    hf_dataset = Dataset.from_pandas(eval_df)

    print(f"[System] RAGAS 평가 중 (청크 사이즈 {chunk_size})...")
    if USE_WRAPPER:
        result = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False,
            batch_size=1,
        )
    else:
        result = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            raise_exceptions=False,
            batch_size=1,
        )

    result_df = result.to_pandas()
    if has_question_type:
        result_df["question_type"] = eval_df["question_type"].values
    result_df["chunk_size"] = chunk_size
    return result_df


# ==========================================
# 5. 메인 함수
# ==========================================
def main():
    dataset_path_csv  = r'C:\Users\sysan\OneDrive\바탕 화면\RAG\golden_dataset.csv'
    dataset_path_xlsx = r'C:\Users\sysan\OneDrive\바탕 화면\RAG\golden_dataset.xlsx'
    output_dir        = r'C:\Users\sysan\OneDrive\바탕 화면\RAG\파라미터 비교.csv'

    if os.path.exists(dataset_path_csv):
        print(f"\n[System] {dataset_path_csv} 파일을 로드합니다...")
        df = pd.read_csv(dataset_path_csv)
    elif os.path.exists(dataset_path_xlsx):
        print(f"\n[System] {dataset_path_xlsx} (xlsx) 파일을 로드합니다...")
        df = pd.read_excel(dataset_path_xlsx)
    else:
        print(f"\n[오류] 평가 데이터셋을 찾을 수 없습니다.")
        print(f"  확인한 경로(CSV) : {dataset_path_csv}")
        print(f"  확인한 경로(XLSX): {dataset_path_xlsx}")
        return

    df.columns = df.columns.str.strip()
    print("[System] 로드된 데이터셋 열 목록:", df.columns.tolist())
    has_question_type = "question_type" in df.columns

    chunk_sizes  = [400, 500, 600]
    all_results  = []
    summary_rows = []

    for chunk_size in chunk_sizes:
        result_df = evaluate_for_chunk_size(chunk_size, df, has_question_type)
        all_results.append(result_df)

        detail_path = os.path.join(output_dir, f"ragas_results_chunk{chunk_size}.csv")
        result_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"[System] 저장 완료: {detail_path}")

        overall = {"chunk_size": chunk_size, "구분": "전체 평균"}
        for col in metric_cols:
            overall[col] = result_df[col].mean()
        summary_rows.append(overall)

        if has_question_type and "question_type" in result_df.columns:
            for qtype, grp in result_df.groupby("question_type"):
                row_data = {"chunk_size": chunk_size, "구분": f"유형: {qtype}"}
                for col in metric_cols:
                    row_data[col] = grp[col].mean()
                summary_rows.append(row_data)

    summary_df   = pd.DataFrame(summary_rows, columns=["chunk_size", "구분"] + metric_cols)
    summary_path = os.path.join(output_dir, "ragas_chunk_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[System] 청크 사이즈 비교 요약 저장 완료: {summary_path}")

    print("\n" + "="*60)
    print(" 📊 [청크 사이즈별 종합 성능 비교 (전체 평균)]")
    print("="*60)
    header = f"{'지표':<32}" + "".join(f"  chunk{cs:<6}" for cs in chunk_sizes)
    print(header)
    print("-"*60)
    label_map = {
        "faithfulness":      "Faithfulness (환각 검증)      ",
        "answer_relevancy":  "Answer Relevancy (질문 관련성) ",
        "context_precision": "Context Precision (검색 정밀도)",
        "context_recall":    "Context Recall (검색 재현율)   ",
    }
    for col in metric_cols:
        row_line = label_map[col]
        for result_df in all_results:
            val = result_df[col].mean()
            row_line += f"  {val:.4f}  " if not pd.isna(val) else "  NaN     "
        print(row_line)
    print("="*60)


if __name__ == "__main__":
    main()