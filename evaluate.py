"""
    python evaluate.py --neo4j_password Ynyspaxl02. --api_key sk-yapyjxwlwpmbhriewgqcilohtnbvfoiymvqinvsihhjpshoy

"""

import argparse
import logging
import os
import json
import time
import re
import threading
import hashlib
import numpy as np
import torch
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import openpyxl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────
VECTORS_DIR        = "D:/RAG/vectors"
INDEX_PATH         = "D:/RAG/faiss.index"
META_PATH          = "D:/RAG/vectors/metadata.jsonl"
TOP_K              = 25            
MAX_TRIPLES        = 30            
EXPAND_RAW_LIMIT   = 80            
SIM_THRESHOLD      = 0.62         
MAX_CONTEXT        = 20           
NEO4J_ENTITY_LIMIT = 10           
RERANKER_MODEL     = "BAAI/bge-reranker-v2-m3" 
DEFAULT_LLM_MODEL  = "deepseek-ai/DeepSeek-V3"  
LLM_MODEL          = DEFAULT_LLM_MODEL
DEFAULT_BASE_URL   = "https://api.siliconflow.cn/v1"
ANSWER_MAX_TOKENS  = 200            
ANSWER_TEMPERATURE = 0.0          
AUX_TEMPERATURE    = 0.0          
TRANSLATION_CACHE  = "translation_cache_v7.json"
CONCURRENCY        = 3            
QUESTIONS_FILE     = "graph_single_choice_v6.jsonl"
# ──────────────────────────────────────────────────

_write_lock       = threading.Lock()
_translate_lock   = threading.Lock()   # 保护翻译缓存的并发读写


def load_translation_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_translation_cache(cache: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _cache_key(text: str) -> str:
    """用 MD5 做缓存 key，避免 key 过长"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def safe_filename(name: str) -> str:
    """把模型名转成适合文件名的形式，避免不同模型结果混在一起。"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def chat_completion(llm_client: OpenAI, messages: list, max_tokens: int,
                    temperature: float = 0.0):
    """
    统一的 LLM 调用入口。
    这样换 DeepSeek/Qwen/其它模型时，只需要改 LLM_MODEL 或 --model。
    """
    return llm_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"enable_thinking": False},
        messages=messages
    )



TRANSLATE_SYSTEM = (
    "You are a professional translator specializing in animal nutrition and feed science. "
    "Translate the following Chinese question stem into concise English, "
    "preserving all technical terms and entity names. "
    "Output only the English translation, nothing else."
)

EXTRACT_ENTITIES_SYSTEM = (
    "You are an expert in animal nutrition and feed science knowledge graphs. "
    "Given a Chinese question about animal nutrition, extract the key scientific entity names "
    "that would appear in an English-language knowledge graph. "
    "Output ONLY a JSON list of English entity names, nothing else. "
    "Use standard scientific nomenclature (e.g., 'Nano ZnO' not 'nano-zinc oxide', "
    "'Phytate' not 'phytic acid salt', 'Fructooligosaccharides' not 'fructo-oligosaccharides'). "
    "Include: substances, additives, enzymes, metabolites, microorganisms, genes, pathways, indicators. "
    "Exclude: generic terms like 'pig', 'diet', 'growth performance'. "
    "Example input: '在断奶仔猪日粮中添加高浓度纳米氧化锌，对腹泻率和血浆锌水平的影响是' "
    'Example output: ["Nano ZnO", "Diarrhea", "Plasma Zinc"]'
)


def translate_query(zh_text: str, llm_client: OpenAI,
                    cache: dict, retries: int = 3) -> str:
    """翻译中文题干为英文（保持原有逻辑，用于向量检索）"""
    key = _cache_key(zh_text)
    with _translate_lock:
        if key in cache:
            return cache[key]

    for attempt in range(retries):
        try:
            resp = chat_completion(
                llm_client,
                max_tokens=256,
                temperature=AUX_TEMPERATURE,
                messages=[
                    {"role": "system", "content": TRANSLATE_SYSTEM},
                    {"role": "user",   "content": zh_text}
                ]
            )
            en_text = resp.choices[0].message.content.strip()
            if en_text:
                with _translate_lock:
                    cache[key] = en_text
                    save_translation_cache(cache, TRANSLATION_CACHE)
                return en_text
        except Exception as e:
            log.warning(f"翻译第{attempt+1}次失败: {e}")
            time.sleep(1)

    log.warning(f"翻译失败，降级为中文原文: {zh_text[:40]}…")
    return zh_text


def extract_graph_entities(zh_text: str, llm_client: OpenAI,
                           cache: dict, retries: int = 2) -> list:
    """
    v7核心新增：用LLM从中文题干提取图谱对齐的英文实体名。
    与简单翻译不同，这里要求LLM输出标准科学命名（与图谱中的实体名一致）。
    """
    cache_key = "entities_" + _cache_key(zh_text)
    with _translate_lock:
        if cache_key in cache:
            try:
                return json.loads(cache[cache_key])
            except Exception:
                pass

    for attempt in range(retries):
        try:
            resp = chat_completion(
                llm_client,
                max_tokens=200,
                temperature=AUX_TEMPERATURE,
                messages=[
                    {"role": "system", "content": EXTRACT_ENTITIES_SYSTEM},
                    {"role": "user",   "content": zh_text}
                ]
            )
            raw = resp.choices[0].message.content.strip()
            # 解析JSON列表
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if m:
                entities = json.loads(m.group())
                if isinstance(entities, list) and entities:
                    # 缓存结果
                    with _translate_lock:
                        cache[cache_key] = json.dumps(entities, ensure_ascii=False)
                        save_translation_cache(cache, TRANSLATION_CACHE)
                    return [str(e).strip() for e in entities if str(e).strip()]
        except Exception as e:
            log.warning(f"实体提取第{attempt+1}次失败: {e}")
            time.sleep(0.5)

    return []




def load_questions(path):
    """支持 .jsonl（v3 题库）和 .xlsx（旧题库）两种格式。"""
    questions = []

    if path.endswith(".jsonl"):
        # jsonl 格式：字段为 question / opa / opb / opc / opd / ans
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                q   = item.get("question", "")
                ans = item.get("ans", "")
                if q and ans:
                    questions.append({
                        "question": str(q),
                        "A": str(item.get("opa", "")),
                        "B": str(item.get("opb", "")),
                        "C": str(item.get("opc", "")),
                        "D": str(item.get("opd", "")),
                        "answer": normalize_answer(ans)
                    })
    else:
        # xlsx 格式：列顺序为 question / A / B / C / D / answer
        wb = openpyxl.load_workbook(path)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            q, a, b, c, d, ans = row
            if q and ans:
                questions.append({
                    "question": str(q),
                    "A": str(a) if a else "",
                    "B": str(b) if b else "",
                    "C": str(c) if c else "",
                    "D": str(d) if d else "",
                    "answer": normalize_answer(ans)
                })

    log.info(f"加载题目 {len(questions)} 道，来源：{path}")
    return questions


def load_done_results(path):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["question"]] = r
    return done


def append_result(path, result):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def format_question(q):
    return (
        f"问题：{q['question']}\n"
        f"A. {q['A']}\n"
        f"B. {q['B']}\n"
        f"C. {q['C']}\n"
        f"D. {q['D']}\n"
    )


def extract_choice(text):
    """
    稳健解析模型输出的 A/B/C/D。
    重点修复：不能用 `if "A" in text` 这种方式，否则 "ANSWER" 会被误判为 A。
    """
    if text is None:
        return "X"

    text = str(text).strip().upper()

    # DeepSeek-R1 / 推理模型可能输出 <think>...</think>，先删除思考内容
    text = re.sub(r"<THINK>.*?</THINK>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 1) 只输出一个字母
    if re.fullmatch(r"[ABCD]", text):
        return text

    # 2) 常见格式：答案是 B / Answer: B / choice B
    m = re.search(r"(?:答案|选择|选项|ANSWER|CHOICE)\s*(?:是|为|:|：)?\s*([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # 3) 括号格式：(B) / （B）
    m = re.search(r"[（(]\s*([ABCD])\s*[）)]", text)
    if m:
        return m.group(1)

    # 4) 独立字母，不能从 ANSWER / BACILLUS 这种单词中抠 A/B/C/D
    m = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])", text)
    if m:
        return m.group(1)

    return "X"


def normalize_answer(ans):
    """清洗题库标准答案，兼容 'A'、'答案：A'、'A. xxx' 等格式。"""
    ans = str(ans).strip().upper()
    m = re.search(r"[ABCD]", ans)
    return m.group(0) if m else "X"





def load_embed_model():
    log.info("加载 bge-m3 embedding 模型…")

    model_path = r"C:\Users\WL\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    log.info(f"模型加载完成，设备: {device} ✓")
    return tokenizer, model



_reranker_model = None
_reranker_tokenizer = None

def load_reranker():
    """加载 bge-reranker-v2-m3 cross-encoder，失败时返回 None"""
    global _reranker_model, _reranker_tokenizer
    try:
        from transformers import AutoModelForSequenceClassification
        log.info(f"加载 reranker 模型: {RERANKER_MODEL}…")
        _reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL, torch_dtype=torch.float16
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker_model = _reranker_model.to(device).eval()
        log.info(f"Reranker 加载完成，设备: {device} ✓")
        return True
    except Exception as e:
        log.warning(f"Reranker 加载失败，将降级到关键词排序: {e}")
        _reranker_model = None
        _reranker_tokenizer = None
        return False


_reranker_lock = threading.Lock()

def rerank_triples(query_text: str, triples: list, top_k: int = MAX_CONTEXT) -> list:
    """
    用 cross-encoder 对 (query, evidence) 对精排。
    triples 是 dict 列表，每个含 'evidence' 字段。
    返回按 reranker 分数降序排列的 top_k 条。
    """
    if _reranker_model is None or not triples:
        return triples[:top_k]

    # 构造 (query, evidence) 对
    pairs = []
    for t in triples:
        ev = t.get("evidence", "") or ""
        # 拼接 head-rel->tail 作为补充上下文
        prefix = f"{t.get('head_name', t.get('head', ''))} [{t.get('rel_type', '')}] {t.get('tail_name', t.get('tail', ''))}: "
        pairs.append([query_text, prefix + ev])

    try:
        with _reranker_lock:
            device = next(_reranker_model.parameters()).device
            inputs = _reranker_tokenizer(
                pairs, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                scores = _reranker_model(**inputs).logits.squeeze(-1).cpu().float().numpy()

        # 附加分数并排序
        scored = list(zip(scores, triples))
        scored.sort(key=lambda x: -x[0])

        result = []
        for score, t in scored[:top_k]:
            t = t.copy()
            t["reranker_score"] = round(float(score), 4)
            result.append(t)
        return result

    except Exception as e:
        log.warning(f"Reranker 推理失败，降级: {e}")
        return triples[:top_k]


ENTITY_SEARCH_QUERY = """
MATCH (n)-[r:RELATION]->(m)
WHERE (toLower(n.name) CONTAINS toLower($entity_name)
       OR toLower(m.name) CONTAINS toLower($entity_name))
  AND r.evidence IS NOT NULL
  AND size(r.evidence) > 40
RETURN
    n.name AS head_name, r.type AS rel_type,
    r.evidence AS evidence, r.source AS source,
    m.name AS tail_name, r.triple_id AS triple_id
LIMIT $limit
"""


def extract_entities_from_question(q, en_query=""):
    """
    v7 大幅增强版实体提取。修复了纯中文题干无法提取实体的问题。
    来源：
    1. 中文题干中的英文词（如 NutriQuest Protect, LPAA, DON）
    2. 中文题干中括号内的术语（如 VFA, G/F）
    3. 选项中的英文专有名词
    4. 英文翻译中的专业术语短语（v7核心新增：不再要求首字母大写）
    5. 中文题干中的关键中文名词（v7核心新增：翻译为英文后检索）
    """
    entities = set()
    zh_text = q.get("question", "")

    # 1. 从中文题干提取英文实体（3+字符的连续英文词组）
    for m in re.finditer(r'[A-Za-z][A-Za-z0-9\-]{2,}(?:\s+[A-Za-z][A-Za-z0-9\-]+)*', zh_text):
        word = m.group().strip()
        if len(word) >= 3 and word.lower() not in {
            "the", "and", "for", "with", "from", "that", "this",
            "are", "was", "were", "not", "has", "have", "been"
        }:
            entities.add(word)

    # 2. 从括号内提取术语
    for m in re.finditer(r'[（(]([A-Za-z][A-Za-z0-9\s\-/]+)[）)]', zh_text):
        term = m.group(1).strip()
        if len(term) >= 2:
            entities.add(term)

    # 3. 从选项提取英文实体（v7放宽：也包括小写开头的术语）
    for key in ["A", "B", "C", "D"]:
        opt_text = q.get(key, "")
        for m in re.finditer(r'[A-Z][a-zA-Z0-9\-]{2,}(?:\s+[A-Z][a-zA-Z0-9\-]+)*', opt_text):
            word = m.group().strip()
            if len(word) >= 3:
                entities.add(word)
        # v7: 也提取括号内的术语（选项中常有）
        for m in re.finditer(r'[（(]([A-Za-z][A-Za-z0-9\s\-/]+)[）)]', opt_text):
            term = m.group(1).strip()
            if len(term) >= 2:
                entities.add(term)

    # 4. v7核心新增：从英文翻译中用NLP短语提取（不再要求首字母大写）
    if en_query:
        EN_STOPWORDS = {
            "what", "when", "where", "which", "effect", "impact", "effects",
            "influence", "addition", "adding", "diet", "diets", "dietary",
            "pigs", "piglets", "sows", "broiler", "poultry", "swine",
            "the", "and", "for", "with", "from", "that", "this", "how",
            "does", "what", "main", "mechanism", "primary", "complete",
            "causal", "chain", "compared", "comparing", "between",
            "performance", "production", "growth", "body", "weight",
            "level", "levels", "concentration", "content", "rate",
            "activity", "expression", "relative", "abundance",
            "intestinal", "serum", "plasma", "blood", "liver", "kidney",
            "muscle", "tissue", "jejunal", "ileal", "cecal", "fecal",
            "under", "conditions", "following", "during", "after", "before",
            "significantly", "reduced", "increased", "improved", "higher",
            "lower", "total", "average", "daily", "gain", "feed",
            "intake", "ratio", "efficiency", "digestibility",
        }

        en_lower = en_query.lower()

        # 4a. 连字符复合词 (nano-zinc oxide, L-arabinose)
        for m in re.finditer(r'[a-zA-Z]+-[a-zA-Z]+(?:\s+[a-zA-Z]+)?', en_lower):
            term = m.group().strip()
            if len(term) >= 5:
                entities.add(term)

        # 4b. 专业术语模式匹配
        # 化学/生物术语模式：含有特定后缀或模式的词
        TERM_PATTERNS = [
            r'\b[a-z]*(?:ase|ose|ine|ate|ol|ene|ide|iol|one|oid)\b',  # 酶/糖/氨基酸等
            r'\b(?:vitamin|amino acid|fatty acid|organic acid)\s*[a-zA-Z0-9]*\b',
            r'\b(?:nano|micro)\s*-?\s*[a-z]+\s*(?:oxide|particle)?\b',
            r'\b[a-z]*flavon\w*\b',  # 黄酮类
            r'\b[a-z]*phenol\w*\b',  # 酚类
            r'\bcalcium\s+\w+\b',
            r'\bsodium\s+\w+\b',
            r'\bzinc\s+\w+\b',
            r'\bcopper\s+\w+\b',
            r'\biron\s+\w+\b',
        ]
        for pat in TERM_PATTERNS:
            for m in re.finditer(pat, en_lower):
                term = m.group().strip()
                if len(term) >= 4 and term not in EN_STOPWORDS:
                    entities.add(term)

        # 4c. 提取2-3词的名词短语（跳过纯停用词组合）
        words = re.findall(r'[a-zA-Z]{2,}', en_query)
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if (words[i].lower() not in EN_STOPWORDS and
                words[i+1].lower() not in EN_STOPWORDS and
                len(bigram) >= 8):
                entities.add(bigram.lower())

        # 4d. 首字母大写的专有名词（保留原有逻辑）
        for m in re.finditer(r'[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*', en_query):
            word = m.group().strip()
            if len(word) >= 4 and word.lower() not in EN_STOPWORDS:
                entities.add(word)


    ZH_EN_MAP = {
        "纳米氧化锌": "Nano ZnO",
        "氧化锌": "Zinc Oxide",
        "植酸": "Phytate",
        "植酸酶": "Phytase",
        "植酸盐": "Phytate",
        "甜菜粕": "Sugar Beet Pulp",
        "豆粕": "Soybean Meal",
        "菜籽粕": "Rapeseed Meal",
        "棉籽粕": "Cottonseed Meal",
        "鱼粉": "Fish Meal",
        "乳清粉": "Whey Powder",
        "小麦麸": "Wheat Bran",
        "玉米": "Corn",
        "大麦": "Barley",
        "燕麦": "Oat",
        "高粱": "Sorghum",
        "木薯": "Cassava",
        "赖氨酸": "Lysine",
        "蛋氨酸": "Methionine",
        "苏氨酸": "Threonine",
        "色氨酸": "Tryptophan",
        "精氨酸": "Arginine",
        "谷氨酰胺": "Glutamine",
        "丁酸": "Butyrate",
        "丙酸": "Propionate",
        "乳酸": "Lactic Acid",
        "阿魏酸": "Ferulic Acid",
        "柠檬酸": "Citric Acid",
        "苯甲酸": "Benzoic Acid",
        "富马酸": "Fumaric Acid",
        "霉菌毒素": "Mycotoxin",
        "呕吐毒素": "Deoxynivalenol",
        "黄曲霉毒素": "Aflatoxin",
        "玉米赤霉烯酮": "Zearalenone",
        "益生菌": "Probiotic",
        "益生元": "Prebiotic",
        "果寡糖": "Fructooligosaccharide",
        "低聚果糖": "Fructooligosaccharide",
        "甘露寡糖": "Mannan Oligosaccharide",
        "酵母": "Yeast",
        "乳酸杆菌": "Lactobacillus",
        "大肠杆菌": "Escherichia coli",
        "沙门氏菌": "Salmonella",
        "链球菌": "Streptococcus",
        "梭菌": "Clostridium",
        "胰蛋白酶": "Trypsin",
        "淀粉酶": "Amylase",
        "脂肪酶": "Lipase",
        "纤维素酶": "Cellulase",
        "木聚糖酶": "Xylanase",
        "碱性磷酸酶": "Alkaline Phosphatase",
        "超氧化物歧化酶": "Superoxide Dismutase",
        "谷胱甘肽": "Glutathione",
        "免疫球蛋白": "Immunoglobulin",
        "干扰素": "Interferon",
        "白细胞介素": "Interleukin",
        "肿瘤坏死因子": "Tumor Necrosis Factor",
        "胰岛素": "Insulin",
        "生长激素": "Growth Hormone",
        "甲状腺激素": "Thyroid Hormone",
        "雌激素": "Estrogen",
        "孕酮": "Progesterone",
        "维生素": "Vitamin",
        "叶酸": "Folic Acid",
        "烟酰胺": "Niacinamide",
        "胡柚果渣": "Citrus Pomace",
        "膨化日粮": "Extruded Diet",
        "发酵饲料": "Fermented Feed",
        "酒糟": "Distillers Grains",
        "有机微量元素": "Organic Trace Minerals",
        "氯化钙": "Calcium Chloride",
        "碳水化合物酶": "Carbohydrase",
    }
    for zh_term, en_term in ZH_EN_MAP.items():
        if zh_term in zh_text:
            entities.add(en_term)

    # 过滤太短或太泛的实体
    filtered = set()
    TOO_GENERIC = {"pig", "pigs", "sow", "sows", "diet", "feed", "acid", "protein", "fat"}
    for e in entities:
        if len(e) >= 3 and e.lower() not in TOO_GENERIC:
            filtered.add(e)

    return filtered


def neo4j_entity_search(driver, entity_name, database="neo4j",
                        limit=NEO4J_ENTITY_LIMIT):
    """在Neo4j中按实体名模糊匹配，返回相关三元组"""
    try:
        with driver.session(database=database) as session:
            result = session.run(
                ENTITY_SEARCH_QUERY,
                entity_name=entity_name,
                limit=limit
            )
            records = []
            for r in result:
                rec = dict(r)
                rec["score"] = 0.85  # 精确匹配给予较高的基础分
                rec["source_type"] = "neo4j_entity"
                records.append(rec)
            return records
    except Exception as e:
        log.warning(f"Neo4j 实体检索失败({entity_name}): {e}")
        return []


def embed_query(query, tokenizer, model):
    device = next(model.parameters()).device
    encoded = tokenizer([query], padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**encoded)
        vec = output.last_hidden_state[:, 0, :]
        vec = torch.nn.functional.normalize(vec, dim=-1)
        vec = vec.cpu().float().numpy()
    faiss.normalize_L2(vec)
    return vec


def load_faiss_index(index_path):
    with open(index_path, "r") as f:
        shard_info = json.load(f)
    log.info(f"分片索引加载完成，共 {shard_info['num_shards']} 个分片 ✓")
    return shard_info


def load_metadata(meta_path):
    log.info("加载 metadata…")
    records = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"metadata 加载完成，共 {len(records)} 条 ✓")
    return records


_faiss_lock = threading.Lock()

def vector_search(query_vec, shard_info, metadata, top_k=TOP_K):
    shard_dir  = shard_info["shard_dir"]
    num_shards = shard_info["num_shards"]
    all_scores, all_indices = [], []
    with _faiss_lock:
        for shard_id in range(num_shards):
            shard_path = os.path.join(shard_dir, f"shard_{shard_id:03d}.index")
            shard_index = faiss.read_index(shard_path)
            scores, local_indices = shard_index.search(query_vec, top_k)
            del shard_index
            offset = shard_id * 100000
            global_indices = np.where(local_indices >= 0, local_indices + offset, -1)
            all_scores.append(scores[0])
            all_indices.append(global_indices[0])
    all_scores  = np.concatenate(all_scores)
    all_indices = np.concatenate(all_indices)
    valid_mask  = all_indices >= 0
    all_scores  = all_scores[valid_mask]
    all_indices = all_indices[valid_mask]
    top_idx = np.argsort(-all_scores)[:top_k]
    results = []
    for idx in top_idx:
        global_i = int(all_indices[idx])
        if global_i >= len(metadata):
            continue
        record = metadata[global_i].copy()
        record["score"] = float(all_scores[idx])
        results.append(record)
    return results


EXPAND_QUERY = """
UNWIND $triple_ids AS tid
MATCH ()-[seed:RELATION {triple_id: tid}]->()
WITH collect(DISTINCT seed) AS seeds
UNWIND seeds AS s
MATCH (h)-[s]->(t)
WITH collect(DISTINCT h) + collect(DISTINCT t) AS seed_nodes
UNWIND seed_nodes AS n
MATCH (n)-[r:RELATION]->(m)
WHERE r.evidence IS NOT NULL AND size(r.evidence) > 40
RETURN
    n.name AS head, r.type AS rel_type,
    r.evidence AS evidence, r.source AS source,
    m.name AS tail, r.triple_id AS triple_id
LIMIT $limit
"""

def expand_subgraph(driver, triple_ids, database="neo4j", limit=EXPAND_RAW_LIMIT):
    with driver.session(database=database) as session:
        result = session.run(EXPAND_QUERY, triple_ids=triple_ids, limit=limit)
        return [dict(r) for r in result]


def build_query_keywords(query_text: str) -> set:
    """从英文查询文本提取关键词，用于过滤扩展三元组。"""
    stopwords = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for",
        "and", "or", "with", "by", "from", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "that", "this", "it", "its",
        # 中文停用词保留，翻译失败降级时仍可用
        "的", "了", "在", "是", "有", "和", "与", "对", "中", "为",
        "其", "该", "此", "这", "一", "不", "也", "都", "量",
        "猪", "仔猪",
    }
    tokens = set()
    for word in re.findall(r'[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}', query_text):
        if word.lower() not in stopwords:
            tokens.add(word.lower())
    return tokens


def filter_expanded_by_relevance(expanded: list, query_keywords: set,
                                max_keep: int = MAX_TRIPLES) -> list:
    """v5: 按关键词重叠度打分排序，取top-N，而非简单过滤。"""
    if not query_keywords or not expanded:
        return expanded[:max_keep]

    scored = []
    for r in expanded:
        head = (r.get("head", "") or "").lower()
        tail = (r.get("tail", "") or "").lower()
        evidence = (r.get("evidence", "") or "").lower()
        combined = head + " " + tail + " " + evidence

        # 计算关键词命中数（head/tail命中权重更高）
        ht_hits = sum(2 for kw in query_keywords if kw in head or kw in tail)
        ev_hits = sum(1 for kw in query_keywords if kw in evidence)
        score = ht_hits + ev_hits
        scored.append((score, r))

    # 按分数降序排，取 max_keep 条
    scored.sort(key=lambda x: -x[0])
    result = [r for score, r in scored[:max_keep] if score > 0]

    # 如果过滤后为空，保留原始前max_keep条
    return result if result else expanded[:max_keep]


def serialize_context(seeds, expanded):
    """v5: 标注相似度分数，帮助LLM判断证据可信度"""
    lines = ["## 检索到的知识图谱证据（按相关性排序）"]
    for i, r in enumerate(seeds):
        score = r.get("score", 0)
        lines.append(
            f"[seed-{i+1}] (相关度:{score:.2f}) "
            f"({r.get('head_name','')}) "
            f"--[{r.get('rel_type','')}]--> "
            f"({r.get('tail_name','')})"
        )
        if r.get("evidence"):
            lines.append(f"    证据：{r.get('evidence','')}")
    if expanded:
        seen = set()
        lines.append("\n## 图谱扩展信息（与上述证据关联的知识）")
        for r in expanded:
            key = f"{r.get('head')}|{r.get('rel_type')}|{r.get('tail')}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"({r.get('head','')}) "
                f"--[{r.get('rel_type','')}]--> "
                f"({r.get('tail','')})"
            )
            if r.get("evidence"):
                lines.append(f"    证据：{r.get('evidence','')}")
    return "\n".join(lines)



SYSTEM_GRAPH_RAG = """你是猪营养与饲料科学领域的专家。
我会提供从科研文献知识库中检索到的相关证据，以及一道单选题，请你选出正确答案。

作答规则：
1. 仔细审查每条证据的相关性。每条证据标注了相关度分数（0-1），分数越高越相关
2. 只采信与题目直接相关的证据（涉及题干提到的实体和指标）
3. 如果检索到的证据与题目无关或不足以判断，请忽略证据，完全依据自身专业知识作答
4. 不要因为证据中出现了某个关键词就强行关联——要判断证据是否真正回答了题目所问的问题
5. 不要因为不确定就偏向选择某个特定选项——每个选项的先验概率是相等的
6. 只输出一个字母（A、B、C 或 D），不要输出任何解释或其他内容"""

SYSTEM_BARE_LLM = """你是猪营养与饲料科学领域的专家。
请根据你的专业知识回答单选题。
只输出一个字母（A、B、C 或 D），不要解释，不要输出其他内容。"""


def extract_option_entities(q):
    """v5: 从选项中提取关键实体名，用于补充检索。"""
    entities = set()
    for key in ["A", "B", "C", "D"]:
        text = q.get(key, "")
        # 提取英文专有名词（首字母大写的连续词组）
        for m in re.finditer(r'[A-Z][a-zA-Z0-9\-]{2,}(?:\s+[A-Z][a-zA-Z0-9\-]+)*', text):
            entities.add(m.group())
        # 提取括号内的英文术语
        for m in re.finditer(r'[（(]([A-Za-z][A-Za-z0-9\s\-]+)[）)]', text):
            entities.add(m.group(1).strip())
    return entities


def answer_with_graph_rag(q, query_vec, shard_info, metadata, driver, llm_client,
                          database="neo4j", en_query_text="",
                          tokenizer=None, embed_model=None,
                          sim_threshold=SIM_THRESHOLD,
                          graph_entities=None):
    """
    v7 改进版：向量检索 + LLM实体提取精确检索 + 正则实体提取 + Reranker精排
    """
    # ── 第1步：向量检索 ──
    seeds = vector_search(query_vec, shard_info, metadata)

    # ── 第2步：实体名精确检索（v7: 合并LLM提取的实体 + 正则提取的实体） ──
    regex_entities = extract_entities_from_question(q, en_query_text)
    llm_entities = set(graph_entities) if graph_entities else set()

    # 合并两个来源的实体，LLM提取的优先（更贴近图谱命名）
    all_entities = list(llm_entities) + [e for e in regex_entities if e not in llm_entities]

    neo4j_hits = []
    seen_triple_ids = {r.get("triple_id") for r in seeds}

    for entity in all_entities[:12]:  # v7: 最多12个实体
        hits = neo4j_entity_search(driver, entity, database=database)
        for h in hits:
            tid = h.get("triple_id")
            if tid and tid not in seen_triple_ids:
                seen_triple_ids.add(tid)
                neo4j_hits.append(h)

    n_neo4j_hits = len(neo4j_hits)

    # ── 第3步：相似度门控 ──
    seed_score_avg = float(np.mean([r["score"] for r in seeds])) if seeds else 0.0
    seed_top_score = max(r["score"] for r in seeds) if seeds else 0.0

    if seed_top_score < sim_threshold and n_neo4j_hits == 0:
        log.debug("检索质量过低且无精确命中，fallback")
        pred, raw, fallback_prompt = answer_bare_llm(q, llm_client)
        return pred, raw + " [FALLBACK:low_sim]", len(seeds), 0, 0, seed_score_avg, n_neo4j_hits, fallback_prompt

    # ── 第4步：图扩展 ──
    triple_ids = [r["triple_id"] for r in seeds if r.get("triple_id")]
    # 也把 neo4j 精确命中的 triple_id 加入扩展
    for h in neo4j_hits:
        tid = h.get("triple_id")
        if tid and tid not in triple_ids:
            triple_ids.append(tid)

    expanded = expand_subgraph(driver, triple_ids, database=database)

    # ── 第5步：合并所有候选，去重 ──
    all_candidates = []
    seen_keys = set()

    # seeds（向量检索结果，已有score）
    for r in seeds:
        key = r.get("triple_id", f"{r.get('head_name')}|{r.get('rel_type')}|{r.get('tail_name')}")
        if key not in seen_keys:
            seen_keys.add(key)
            all_candidates.append(r)

    # neo4j精确命中（给高分）
    for r in neo4j_hits:
        key = r.get("triple_id", f"{r.get('head_name')}|{r.get('rel_type')}|{r.get('tail_name')}")
        if key not in seen_keys:
            seen_keys.add(key)
            all_candidates.append(r)

    # 扩展结果
    for r in expanded:
        key = r.get("triple_id", f"{r.get('head')}|{r.get('rel_type')}|{r.get('tail')}")
        if key not in seen_keys:
            seen_keys.add(key)
            # 统一字段名
            r["head_name"] = r.get("head_name", r.get("head", ""))
            r["tail_name"] = r.get("tail_name", r.get("tail", ""))
            r["score"] = r.get("score", 0.5)
            all_candidates.append(r)

    # ── 第6步：Reranker 精排（v6核心新增）──
    # 构造查询文本：中英文拼接
    rerank_query = en_query_text if en_query_text else q.get("question", "")
    # 加入选项信息帮助reranker理解题目意图
    opts_text = " ".join(q.get(k, "") for k in ["A", "B", "C", "D"])
    rerank_query = rerank_query + " " + opts_text[:200]

    reranked = rerank_triples(rerank_query, all_candidates, top_k=MAX_CONTEXT)

    # 如果reranker没加载，降级到关键词排序
    if _reranker_model is None and en_query_text:
        keywords = build_query_keywords(en_query_text)
        zh_text = q.get("question", "") + " " + opts_text
        for word in re.findall(r'[\u4e00-\u9fff]{2,}', zh_text):
            keywords.add(word)
        reranked = filter_expanded_by_relevance(all_candidates, keywords, max_keep=MAX_CONTEXT)

    # ── 第7步：构造上下文 + LLM回答 ──
    # 将 reranked 按照 seed 和 expanded 的格式分开
    final_seeds = [r for r in reranked if r.get("score", 0) >= 0.6 or r.get("source_type") == "neo4j_entity"]
    final_expanded = [r for r in reranked if r not in final_seeds]

    context = serialize_context(final_seeds, final_expanded)
    prompt  = f"{context}\n\n{format_question(q)}\n请选择正确答案（只输出字母）："

    response = chat_completion(
        llm_client,
        max_tokens=ANSWER_MAX_TOKENS,
        temperature=ANSWER_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_GRAPH_RAG},
            {"role": "user",   "content": prompt}
        ]
    )
    raw = response.choices[0].message.content
    return (extract_choice(raw), raw, len(seeds), len(expanded), len(reranked),
            seed_score_avg, n_neo4j_hits, prompt)


def answer_bare_llm(q, llm_client):
    prompt   = f"{format_question(q)}\n请选择正确答案（只输出字母）："
    response = chat_completion(
        llm_client,
        max_tokens=ANSWER_MAX_TOKENS,
        temperature=ANSWER_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_BARE_LLM},
            {"role": "user",   "content": prompt}
        ]
    )
    raw = response.choices[0].message.content
    return extract_choice(raw), raw, prompt



_embed_lock = threading.Lock()

def process_one(q, tokenizer, embed_model, shard_info, metadata,
                driver, llm_client, database, results_file,
                translation_cache, sim_threshold=SIM_THRESHOLD):
    try:
        zh_query = q["question"]

        # ── 翻译题干 + 提取图谱对齐的实体名（v7并行） ──
        en_query = translate_query(zh_query, llm_client, translation_cache)
        graph_entities = extract_graph_entities(zh_query, llm_client, translation_cache)
        log.debug(f"翻译: {zh_query[:30]}… → {en_query[:60]}…")
        log.debug(f"提取实体: {graph_entities[:5]}")

        with _embed_lock:
            query_vec = embed_query(en_query, tokenizer, embed_model)

        # 系统A（图RAG）和系统B（裸LLM）并发
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(
                answer_with_graph_rag,
                q, query_vec, shard_info, metadata, driver, llm_client,
                database, en_query,
                tokenizer, embed_model,
                sim_threshold,
                graph_entities             # v7: 传入LLM提取的图谱实体
            )
            fut_b = ex.submit(answer_bare_llm, q, llm_client)
            A_pred, A_raw, n_seeds, n_expanded, n_context, seed_score_avg, n_neo4j_hits, A_context = fut_a.result()
            B_pred, B_raw, B_context = fut_b.result()

        is_fallback = "[FALLBACK" in A_raw

        result = {
            "question":        q["question"],
            "answer":          q["answer"],
            "A_pred":          A_pred,
            "B_pred":          B_pred,
            "A_correct":       A_pred == q["answer"],
            "B_correct":       B_pred == q["answer"],
            "A_raw":           A_raw,
            "B_raw":           B_raw,
            "A_context":       A_context,   # 图RAG完整输入context
            "B_context":       B_context,   # 裸LLM输入context
            "n_seeds":         n_seeds,
            "n_expanded":      n_expanded,      # Neo4j 扩展候选数
            "n_context":       n_context,       # 实际送入 LLM 的三元组数
            "seed_score_avg":  round(seed_score_avg, 4),
            "context_triples": n_context,
            "model_name":      LLM_MODEL,
            "en_query":        en_query,
            "is_fallback":     is_fallback,
            "n_neo4j_hits":    n_neo4j_hits,
            "graph_entities":  graph_entities,  # v7: 记录提取的实体
        }
        append_result(results_file, result)
        return result

    except Exception as e:
        log.error(f"题目处理失败: {q['question'][:40]}… 错误: {e}")
        return None



def generate_report(results, output_path):
    total = len(results)
    if total == 0:
        log.warning("没有结果，无法生成报告")
        return

    A_correct  = sum(1 for r in results if r["A_correct"])
    B_correct  = sum(1 for r in results if r["B_correct"])
    A_wins     = [r for r in results if r["A_correct"] and not r["B_correct"]]
    B_wins     = [r for r in results if r["B_correct"] and not r["A_correct"]]
    both_wrong = [r for r in results if not r["A_correct"] and not r["B_correct"]]
    scores     = [r.get("seed_score_avg", 0) for r in results]
    ctx_sizes  = [r.get("context_triples", 0) for r in results]

    lines = []
    lines.append("=" * 60)
    lines.append("         图RAG vs 裸LLM 测评报告（v7-fixed）")
    lines.append("=" * 60)
    lines.append(f"\n总题数：{total}")
    lines.append(f"\n{'系统':<22} {'正确数':>8} {'正确率':>8}")
    lines.append("-" * 42)
    lines.append(f"{'A: 图RAG系统（v7-fixed）':<22} {A_correct:>8} {A_correct/total*100:>7.1f}%")
    lines.append(f"{('B: ' + LLM_MODEL):<22} {B_correct:>8} {B_correct/total*100:>7.1f}%")
    lines.append(f"\n图RAG提升：{(A_correct-B_correct)/total*100:+.1f}%")

    lines.append("\n\n── 检索质量统计 ──")
    lines.append(f"平均 seed 向量相似度：{np.mean(scores):.4f}  (v3={0.7572:.4f}，越高越好)")
    lines.append(f"平均上下文三元组数：  {np.mean(ctx_sizes):.1f}")

    # v5: fallback统计
    fallback_count = sum(1 for r in results if r.get("is_fallback", False))
    fallback_correct = sum(1 for r in results if r.get("is_fallback", False) and r["A_correct"])
    non_fallback = [r for r in results if not r.get("is_fallback", False)]
    nf_correct = sum(1 for r in non_fallback if r["A_correct"])
    lines.append(f"\n── 相似度门控统计 ──")
    lines.append(f"Fallback到裸LLM：    {fallback_count} 题")
    if fallback_count > 0:
        lines.append(f"  Fallback正确率：   {fallback_correct}/{fallback_count} = {fallback_correct/fallback_count*100:.1f}%")
    if non_fallback:
        lines.append(f"使用图RAG上下文：    {len(non_fallback)} 题")
        lines.append(f"  图RAG正确率：      {nf_correct}/{len(non_fallback)} = {nf_correct/len(non_fallback)*100:.1f}%")

    # v6: 实体精确检索统计
    neo4j_hits_list = [r.get("n_neo4j_hits", 0) for r in results]
    has_neo4j = sum(1 for n in neo4j_hits_list if n > 0)
    lines.append(f"\n── 实体精确检索统计 ──")
    lines.append(f"有Neo4j精确命中的题：{has_neo4j}/{total} ({has_neo4j/total*100:.0f}%)")
    if has_neo4j > 0:
        neo4j_correct = sum(1 for r in results if r.get("n_neo4j_hits", 0) > 0 and r["A_correct"])
        neo4j_total = sum(1 for r in results if r.get("n_neo4j_hits", 0) > 0)
        lines.append(f"  有命中时正确率：   {neo4j_correct}/{neo4j_total} = {neo4j_correct/neo4j_total*100:.1f}%")
    no_neo4j = [r for r in results if r.get("n_neo4j_hits", 0) == 0]
    if no_neo4j:
        nn_correct = sum(1 for r in no_neo4j if r["A_correct"])
        lines.append(f"  无命中时正确率：   {nn_correct}/{len(no_neo4j)} = {nn_correct/len(no_neo4j)*100:.1f}%")
    lines.append(f"平均Neo4j命中数：    {np.mean(neo4j_hits_list):.1f}")

    lines.append("\n\n── 按标准答案分类 ──")
    lines.append(f"{'答案':<6} {'题数':>6} {'图RAG正确率':>12} {'裸LLM正确率':>12}")
    lines.append("-" * 40)
    for ans in ["A", "B", "C", "D"]:
        sub = [r for r in results if r["answer"] == ans]
        if not sub:
            continue
        a_acc = sum(1 for r in sub if r["A_correct"]) / len(sub) * 100
        b_acc = sum(1 for r in sub if r["B_correct"]) / len(sub) * 100
        lines.append(f"{ans:<6} {len(sub):>6} {a_acc:>11.1f}% {b_acc:>11.1f}%")

    lines.append(f"\n\n── 答题情况分布 ──")
    lines.append(f"两者都对：    {sum(1 for r in results if r['A_correct'] and r['B_correct'])} 题")
    lines.append(f"仅图RAG对：   {len(A_wins)} 题  ← 图RAG独有优势")
    lines.append(f"仅裸LLM对：   {len(B_wins)} 题  ← 图RAG引入干扰")
    lines.append(f"两者都错：    {len(both_wrong)} 题")

    lines.append("\n\n── 典型案例：图RAG答对，裸LLM答错（前5题）──")
    for r in A_wins[:5]:
        lines.append(f"\n题目：{r['question'][:60]}…")
        lines.append(f"英文查询：{r.get('en_query','N/A')[:80]}")
        lines.append(f"正确答案：{r['answer']}")
        lines.append(f"图RAG回答：{r['A_pred']}  {'✓' if r['A_correct'] else '✗'}")
        lines.append(f"裸LLM回答：{r['B_pred']}  {'✓' if r['B_correct'] else '✗'}")
        lines.append(f"检索相似度：{r.get('seed_score_avg','N/A')}  上下文三元组：{r.get('context_triples','N/A')}")

    lines.append("\n\n── 典型案例：裸LLM答对，图RAG答错（前5题）──")
    for r in B_wins[:5]:
        lines.append(f"\n题目：{r['question'][:60]}…")
        lines.append(f"英文查询：{r.get('en_query','N/A')[:80]}")
        lines.append(f"正确答案：{r['answer']}")
        lines.append(f"图RAG回答：{r['A_pred']}  {'✓' if r['A_correct'] else '✗'}")
        lines.append(f"裸LLM回答：{r['B_pred']}  {'✓' if r['B_correct'] else '✗'}")
        lines.append(f"检索相似度：{r.get('seed_score_avg','N/A')}  上下文三元组：{r.get('context_triples','N/A')}")

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)
    print(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"报告已保存到 {output_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j_uri",      default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user",     default="neo4j")
    parser.add_argument("--neo4j_password", default="")
    parser.add_argument("--neo4j_database", default="neo4j")
    parser.add_argument("--api_key",        default="")
    parser.add_argument("--base_url",       default=DEFAULT_BASE_URL)
    parser.add_argument("--model",          default=DEFAULT_LLM_MODEL,
                        help="LLM模型名，例如 deepseek-ai/DeepSeek-V3、deepseek-ai/DeepSeek-R1、Qwen/Qwen3.5-9B")
    parser.add_argument("--results_file",   default=None,
                        help="结果jsonl路径；默认按模型名自动生成，避免混用旧结果")
    parser.add_argument("--report_file",    default=None,
                        help="报告txt路径；默认按模型名自动生成")
    parser.add_argument("--questions_file", default=QUESTIONS_FILE)
    parser.add_argument("--max_questions",  type=int, default=None)
    parser.add_argument("--concurrency",    type=int, default=CONCURRENCY,
                        help="并发题目数，默认5，API限流时调小")
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument("--report_only",    action="store_true")
    parser.add_argument("--sim_threshold",  type=float, default=SIM_THRESHOLD,
                        help=f"相似度门控阈值，低于此值fallback到裸LLM（默认{SIM_THRESHOLD}）")
    parser.add_argument("--no_reranker",    action="store_true",
                        help="不加载reranker，降级到关键词排序")
    args = parser.parse_args()

    global LLM_MODEL
    LLM_MODEL = args.model
    model_tag = safe_filename(args.model)
    results_file = args.results_file or f"eval_results_{model_tag}.jsonl"
    report_file = args.report_file or f"eval_report_{model_tag}.txt"
    log.info(f"当前测评模型：{LLM_MODEL}")
    log.info(f"结果文件：{results_file}")
    log.info(f"报告文件：{report_file}")

    if args.report_only:
        done = load_done_results(results_file)
        generate_report(list(done.values()), report_file)
        return

    questions = load_questions(args.questions_file)
    if args.max_questions:
        questions = questions[:args.max_questions]
    log.info(f"共加载 {len(questions)} 道题")

    done = load_done_results(results_file) if args.resume else {}
    if done:
        log.info(f"已完成 {len(done)} 题，将跳过")

    todo = [q for q in questions if q["question"] not in done]
    log.info(f"待测试：{len(todo)} 题，并发数：{args.concurrency}")

    # 加载翻译缓存（重跑时不重复调翻译 API）
    translation_cache = load_translation_cache(TRANSLATION_CACHE)
    log.info(f"翻译缓存已加载，命中条目：{len(translation_cache)}")

    llm_client             = OpenAI(api_key=args.api_key, base_url=args.base_url)
    shard_info             = load_faiss_index(INDEX_PATH)
    metadata               = load_metadata(META_PATH)
    tokenizer, embed_model = load_embed_model()

    # v6: 加载 reranker
    if not args.no_reranker:
        load_reranker()
    else:
        log.info("已跳过 reranker 加载（--no_reranker）")

    driver                 = GraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)
    )
    driver.verify_connectivity()
    log.info("Neo4j 连接成功 ✓")

    all_results = list(done.values())
    start_time  = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                process_one,
                q, tokenizer, embed_model, shard_info, metadata,
                driver, llm_client, args.neo4j_database, results_file,
                translation_cache,
                args.sim_threshold         # v5: 传入门控阈值
            ): q
            for q in todo
        }

        with tqdm(total=len(todo), desc="测评进度") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_results.append(result)
                pbar.update(1)

                done_count = len(all_results)
                if done_count > 0 and done_count % 50 == 0:
                    A_acc = sum(1 for r in all_results if r["A_correct"]) / done_count * 100
                    B_acc = sum(1 for r in all_results if r["B_correct"]) / done_count * 100
                    elapsed = (time.time() - start_time) / 60
                    log.info(
                        f"进度 {done_count}/{len(questions)} | "
                        f"图RAG: {A_acc:.1f}% | 裸LLM: {B_acc:.1f}% | "
                        f"耗时: {elapsed:.1f}min"
                    )

    driver.close()
    generate_report(all_results, report_file)


if __name__ == "__main__":
    main()
