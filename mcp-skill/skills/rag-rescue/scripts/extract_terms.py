# rag-rescue skill 附带资源：从查询中抽取关键术语（jieba TF-IDF）
# 用法: python extract_terms.py "查询文本"    （或从 stdin 读入）
# 输出: JSON 字符串数组，如 ["倒排索引", "BM25"]

import json
import sys

import jieba.analyse


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    if not query:
        print("[]")
        return
    terms = jieba.analyse.extract_tags(query, topK=8)
    print(json.dumps(terms, ensure_ascii=False))


if __name__ == "__main__":
    main()
