import os
import ast
import pandas as pd
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
# 1. RAGAS 평가자(Judge) 모델 및 임베딩 세팅
#    - RAGAS 0.2+ 신버전: LangchainLLMWrapper 사용
#    - 구버전 fallback 지원
# ==========================================
print("\n[System] 로컬 평가 모델(Ollama)을 설정합니다...")

_llm_raw = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0,
    num_ctx=4096,          # 컨텍스트 길이 제한 (메모리 절약)
    timeout=300,           # 타임아웃 5분으로 연장
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
    # 구버전 방식: 지표에 직접 바인딩
    faithfulness.llm          = evaluator_llm
    answer_relevancy.llm      = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings
    context_precision.llm     = evaluator_llm
    context_recall.llm        = evaluator_llm
    USE_WRAPPER = False
    print("[System] RAGAS 구버전 방식 (직접 바인딩) 사용")

metrics = [faithfulness, answer_relevancy, context_precision, context_recall]


# ==========================================
# 2. retrieved_contexts 파싱 함수
#    CSV에 문자열로 저장된 리스트 -> 실제 list[str]
# ==========================================
def parse_contexts(value):
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except Exception:
        return [str(value)]


# ==========================================
# 3. 메인 평가 함수
# ==========================================
def main():
    # ------------------------------------------
    # 이미 생성된 결과 CSV 경로 (경로 수정 금지)
    # ------------------------------------------
    input_csv_path  = r"C:\Users\sysan\OneDrive\바탕 화면\RAG\ragas_final_results.csv"
    output_dir      = r"C:\Users\sysan\OneDrive\바탕 화면\RAG"
    output_detail   = os.path.join(output_dir, "ragas_evaluated_results.csv")
    output_summary  = os.path.join(output_dir, "ragas_summary_scores.csv")

    if not os.path.exists(input_csv_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {input_csv_path}")
        return

    print(f"\n[System] 기존 결과 파일을 로드합니다: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # 열 이름 앞뒤 공백 제거
    df.columns = df.columns.str.strip()
    print("[System] 로드된 CSV 열 목록:", df.columns.tolist())

    # ------------------------------------------
    # RAGAS 0.2+ 는 컬럼명을 그대로 사용:
    #   user_input, retrieved_contexts, response, reference
    # (rename 하지 않음 — rename 하면 RAGAS가 인식 못 함)
    # ------------------------------------------
    eval_df = pd.DataFrame()
    eval_df["user_input"]          = df["user_input"]
    eval_df["response"]            = df["response"]
    eval_df["reference"]           = df["reference"]
    eval_df["retrieved_contexts"]  = df["retrieved_contexts"].apply(parse_contexts)

    if "question_type" in df.columns:
        eval_df["question_type"] = df["question_type"]

    print(f"[System] 총 {len(eval_df)}개 행 로드 완료.")
    print("[System] contexts 샘플 (첫 번째 행):", eval_df["retrieved_contexts"].iloc[0])

    # HuggingFace Dataset 포맷으로 변환
    hf_dataset = Dataset.from_pandas(eval_df)

    print("\n[System] RAGAS 자동 평가를 시작합니다. (모델 추론으로 인해 시간이 소요됩니다)")

    # RAGAS 평가 실행 (버전에 따라 llm/embeddings 전달 방식 분기)
    if USE_WRAPPER:
        result = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False,
            batch_size=1,   # 한 번에 1개씩 처리 (로컬 모델 타임아웃 방지)
        )
    else:
        result = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            raise_exceptions=False,
            batch_size=1,   # 한 번에 1개씩 처리 (로컬 모델 타임아웃 방지)
        )

    result_df = result.to_pandas()

    # question_type 컬럼 복원
    if "question_type" in eval_df.columns:
        result_df["question_type"] = eval_df["question_type"].values

    # ==========================================
    # 4. 상세 결과 CSV 저장
    # ==========================================
    result_df.to_csv(output_detail, index=False, encoding="utf-8-sig")
    print(f"\n[System] 상세 평가 결과 저장 완료: '{output_detail}'")

    # ==========================================
    # 5. 종합 점수 요약 CSV 저장
    # ==========================================
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # 전체 평균
    overall_row = {"구분": "전체 평균"}
    for col in metric_cols:
        overall_row[col] = result_df[col].mean()
    summary_rows = [overall_row]

    # 질문 유형별 평균 (question_type 컬럼이 있을 때만)
    if "question_type" in result_df.columns:
        for qtype, grp in result_df.groupby("question_type"):
            row = {"구분": f"유형: {qtype}"}
            for col in metric_cols:
                row[col] = grp[col].mean()
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows, columns=["구분"] + metric_cols)
    summary_df.to_csv(output_summary, index=False, encoding="utf-8-sig")
    print(f"[System] 종합 점수 요약 저장 완료: '{output_summary}'")

    # ==========================================
    # 6. 콘솔 출력
    # ==========================================
    print("\n" + "="*55)
    print(" 🎯 [RAG 시스템 종합 점수 (전체 평균)]")
    print("="*55)
    for col in metric_cols:
        val = result_df[col].mean()
        label = {
            "faithfulness":      "Faithfulness (환각 검증)     ",
            "answer_relevancy":  "Answer Relevancy (질문 관련성)",
            "context_precision": "Context Precision (검색 정밀도)",
            "context_recall":    "Context Recall (검색 재현율)  ",
        }[col]
        print(f" {label}: {val:.4f}" if not pd.isna(val) else f" {label}: NaN (모델 추론 실패)")

    if "question_type" in result_df.columns:
        print("\n" + "="*55)
        print(" 📊 [질문 유형별(Question Type) 상세 성능 분석]")
        print("="*55)
        type_summary = result_df.groupby("question_type")[metric_cols].mean()
        print(type_summary)


if __name__ == "__main__":
    main()