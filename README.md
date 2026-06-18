# 스마트공장 RAG 기반 의사결정 지원 시스템

## 1. 프로젝트 개요

본 프로젝트는 중소기업의 스마트공장 도입 의사결정을 지원하기 위한 RAG 기반 질의응답 시스템이다.
스마트공장 우수사례집, 공급기업 정보, 스마트공장 수준 자가진단 자료를 기반으로 사용자의 질문에 대해 관련 사례와 기술 정보를 검색하고, 중소기업 관점에서 이해하기 쉬운 답변을 제공하는 것을 목표로 한다.

## 2. 주요 기능

* PDF, CSV 자료 전처리
* 문서 청킹 및 FAISS 벡터 DB 구축
* 사용자 질문 기반 관련 문서 검색
* Query Rewriting을 통한 검색 품질 개선
* MMR 검색 및 Reranker 기반 문서 재정렬
* 스마트공장 수준 자가진단 결과를 반영한 맞춤형 답변 생성
* Streamlit 기반 웹 애플리케이션 실행
* RAGAS 기반 검색 및 답변 품질 평가

## 3. 데이터 구성

본 프로젝트에서 사용한 데이터는 다음 자료를 기반으로 구성하였다.

* 중소기업중앙회 스마트공장 우수사례 자료

  * 2018~2020년 우수사례 자료 활용
* 중소벤처기업부 스마트공장 사업관리시스템 자료

  * 2018~2025년 스마트공장 우수사례집 추가
* 중소기업중앙회 스마트공장 수준 자가진단 자료

  * 스마트공장 수준 자가진단 엑셀 파일 활용

데이터는 `data/raw` 폴더에 PDF, CSV, 이미지 형태로 저장하여 사용한다.

## 4. 프로젝트 구조

```bash
project/
├── app.py
├── main.py
├── evaluate_ragas.py
├── data/
│   ├── raw/
│   │   ├── pdf/
│   │   ├── csv/
│   │   └── images/
│   └── processed/
│       ├── text/
│       ├── chunks/
│       └── report/
└── README.md
```

## 5. 실행 방법

### 5.1 최초 실행 또는 데이터 변경 시

원본 데이터를 전처리하고 벡터 DB에 적재한다.

```bash
python main.py --preprocess
```

이 명령은 PDF, CSV 데이터를 불러온 뒤 청킹 및 임베딩을 수행하고 벡터 DB를 생성한다.

### 5.2 평소 앱 실행

이미 생성된 벡터 DB를 사용하여 Streamlit 앱을 실행한다.

```bash
streamlit run app.py
```

### 5.3 DB 연결 테스트

PostgreSQL 연결 여부를 확인한다.

```bash
python main.py --test-db
```

### 5.4 터미널 채팅 모드 실행

웹 앱 없이 터미널에서 RAG 질의응답을 테스트할 수 있다.

```bash
python main.py --chat
```

## 6. RAG 파이프라인

본 시스템의 RAG 파이프라인은 다음과 같이 구성된다.

1. 원본 데이터 수집
2. PDF, CSV, 이미지 데이터 전처리
3. 텍스트 정제 및 청킹
4. HuggingFace Embedding 모델을 이용한 임베딩 생성
5. FAISS 벡터 DB 저장
6. 사용자 질문 입력
7. Query Rewriting 수행
8. MMR 기반 관련 문서 검색
9. CrossEncoder Reranker를 통한 검색 결과 재정렬
10. LLM 기반 최종 답변 생성

## 7. 평가 방법

RAG 시스템의 성능 평가는 RAGAS를 활용하여 수행한다.

### evaluate_ragas(2)

중간 CSV 파일이 이미 존재하는 경우 사용한다.
기본 청크 사이즈는 400으로 설정되어 있으며, 필요에 따라 조정할 수 있다.

### evaluate_ragas(3)

중간 CSV 파일이 없는 경우 사용한다.
청크 사이즈 400, 500, 600에 대해 각각 평가를 수행하고, 결과를 CSV 파일로 저장한다.

## 8. 사용 기술

* Python
* Streamlit
* LangChain
* FAISS
* HuggingFace Embeddings
* SentenceTransformers CrossEncoder
* Ollama Qwen3
* EasyOCR
* PostgreSQL
* RAGAS

## 9. 기대 효과

본 시스템은 스마트공장 도입을 고민하는 중소기업이 기존 우수사례와 수준진단 자료를 기반으로 자신의 상황에 맞는 정보를 탐색할 수 있도록 돕는다.
특히 단순한 문서 검색이 아니라, 기업의 현재 수준과 주요 고민을 반영하여 의사결정에 활용 가능한 답변을 제공한다는 점에서 의미가 있다.
