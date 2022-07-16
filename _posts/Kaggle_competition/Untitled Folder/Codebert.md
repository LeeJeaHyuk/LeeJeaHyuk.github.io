

```
RANDOM_STATE = 42
MD_MAX_LEN = 64 #마크다운 토큰 최대 64토큰
TOTAL_MAX_LEN = 512 #code context는 512 토큰까지 (23토큰 이하로 구성된 코드 셀 20개까지)
K_FOLDS = 5
FILES_PER_FOLD = 16
LIMIT = 1_000 if os.environ["KAGGLE_KERNEL_RUN_TYPE"] == "Interactive" else None
# KAGGLE_KERNEL_RUN_TYPE 환경 변수가 Interactive이면 앞의 1000개의 노트북만을사용한다
# LIMIT = None # 전체 노트북 사용
MODEL_NAME = "microsoft/codebert-base"
TOKENIZER = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
INPUT_PATH = "../input/AI4Code"
```



```
# n보다 cell들이 많으면 샘플을 뽑고, 그렇지 않으면 그대로 cells를 return
def sample_cells(cells: List[str], n: int) -> List[str]:
    cells = [clean_code(cell) for cell in cells]
    if n >= len(cells):
        return cells
    else:
        results = []
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells): # 전체 샘플 중 step만큼 건너뛴 샘플들을 골라 result에 넣음
            results.append(cells[int(np.round(idx))])
            idx += step
        if cells[-1] not in results:
            results[-1] = cells[-1] # 마지막 셀은 반드시 넣음
        return results

```



```
# total_code(전체 코드), total_md(전체 마크다운), codes(코드에서 샘플 추출) 특성 생성
def get_features(df: pd.DataFrame) -> dict:
    features = {}
    for i, sub_df in tqdm(df.groupby("id"), desc="Features"):
        features[i] = {}
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        codes = sample_cells(code_sub_df.source.values, 20)
        features[i]["total_code"] = total_code
        features[i]["total_md"] = total_md
        features[i]["codes"] = codes
    return features
```



```
# input 값을 준비
def tokenize(df: pd.DataFrame, fts: dict) -> dict:
    input_ids = np.zeros((len(df), TOTAL_MAX_LEN), dtype=np.int32)
    attention_mask = np.zeros((len(df), TOTAL_MAX_LEN), dtype=np.int32)
    features = np.zeros((len(df),), dtype=np.float32)
    labels = np.zeros((len(df),), dtype=np.float32)

    for i, row in tqdm( # tqdm : 진행률을 보여주는 바 생성
        df.reset_index(drop=True).iterrows(), desc="Tokens", total=len(df)
    ): # iterrows : 행번호와 값 동시에 출력, reset_index : 인덱스 초기화
        row_fts = fts[row.id]

        inputs = TOKENIZER.encode_plus( 
            row.source,
            None,
            add_special_tokens=True,
            # 토큰의 시작점에 [CLS] 토큰, 토큰의 마지막에 [SEP] 토큰을 붙인다
            
            max_length=MD_MAX_LEN,
            padding="max_length",
            return_token_type_ids=True, # token type id 생성(0과 1로 문장의 토큰 값 분리)
            truncation=True,
        ) # code context(~512)
        code_inputs = TOKENIZER.batch_encode_plus(
            [str(x) for x in row_fts["codes"]] or [""],
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"] # 토크나이즈 된 MD
        for x in code_inputs["input_ids"]:
            ids.extend(x[:-1]) # 토크나이즈 된 코드를 ids 뒤에 이어붙임
        ids = ids[:TOTAL_MAX_LEN] # 512 토큰까지만 사용
        if len(ids) != TOTAL_MAX_LEN: # 토큰의 길이가 512보다 작으면 그만큼 pad_token_id를 이어붙임
            ids = ids + [
                TOKENIZER.pad_token_id,
            ] * (TOTAL_MAX_LEN - len(ids))

        mask = inputs["attention_mask"] # 토큰의 길이가 512보다 작으면 그만큼 pad_token_id를 이어붙임
        for x in code_inputs["attention_mask"]:
            mask.extend(x[:-1]) # 코드 마스크도 뒤에 이어붙임
        mask = mask[:TOTAL_MAX_LEN]# 512 토큰까지만 사용
        if len(mask) != TOTAL_MAX_LEN:# 토큰의 길이가 512보다 작으면 그만큼 pad_token_id를 이어붙임
            mask = mask + [
                TOKENIZER.pad_token_id,
            ] * (TOTAL_MAX_LEN - len(mask))

        input_ids[i] = ids # 결과 배열에 넣음
        attention_mask[i] = mask
        features[i] = (
            row_fts["total_md"] / (row_fts["total_md"] + row_fts["total_code"]) or 1  # 분모가 1일 때 예외처리?
        ) # 특성은 MD 셀 수 / 전체 셀 수
        labels[i] = row.pct_rank # 레이블은 자료의 순서(rank)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "features": features,
        "labels": labels,
    }

```

