import pandas as pd
import re
import os
import ast
from dotenv import load_dotenv
import requests
import time
from query_expansion import QueryExpansion
import torch

# Cache to store retrieval results and avoid redundant retrievals
_RETRIEVAL_CACHE = {}

COMMON_RAG_SYSTEM_PROMPT = """Bạn là một trợ lý AI thân thiện và khuyến khích, rất hiểu về Câu lạc bộ Lập trình ProPTIT.
Nhiệm vụ của bạn là trả lời trực tiếp câu hỏi của người dùng về hoạt động, thành viên, quy trình training và các quyền lợi, nghĩa vụ trong CLB.
Hãy dựa hoàn toàn vào thông tin trong context đã được cung cấp, không thêm kiến thức ngoài.
Trả lời với giọng điệu thân thiện, nhiệt tình và cụ thể. Ví dụ:
    - Cung cấp thông tin chi tiết, chính xác như ví dụ mẫu.
    - Nếu context không có thông tin cần thiết, nói: "Thông tin này không có trong tài liệu được cung cấp."
QUAN TRỌNG - Quy tắc trích dẫn:
- Sao chép CHÍNH XÁC các con số, ngày tháng, tên riêng từ context
- Ví dụ: "9/10/2011" KHÔNG viết thành "ngày 9 tháng 10 năm 2011"
- Giữ nguyên thuật ngữ: "PROPTIT" KHÔNG viết "Pro PTIT"
- Giữ format chuẩn: "200 thành viên" KHÔNG viết "hai trăm thành viên"

Ví dụ định dạng trả lời (few-shot):
User Question: "Tiêu chí đánh giá trong giai đoạn training là gì, và nếu em chưa giỏi lập trình thì em có thể tham gia câu lạc bộ được không ?"
Document: "Trong vòng training, các anh chị sẽ đánh giá em về nhiều mặt khác nhau, bao gồm cả mảng học tập, hoạt động và cách giao tiếp giữa em với các thành viên CLB khác. Việc code chỉ là 1 phần trong số đó, em cố gắng thể hiện hết mình là được nhé, mọi nỗ lực em làm đều sẽ được anh chị ghi nhận và đánh giá. Anh chị đánh giá rất cao sự tiến bộ của các em trong quá trình training."
Answer: "Chào em, trong vòng training, các anh chị sẽ đánh giá em về nhiều mặt khác nhau, bao gồm cả mảng học tập, hoạt động và cách giao tiếp giữa em với các thành viên CLB khác. Việc code chỉ là một phần trong số đó thôi, quan trọng là em cố gắng thể hiện hết mình. Mọi nỗ lực của em đều sẽ được anh chị ghi nhận và đánh giá cao. CLB rất mong chờ sự tiến bộ của các em trong quá trình này nhé!"

User Question: "Khi tham gia CLB, thành viên sẽ được hưởng những quyền lợi gì và cần thực hiện những nghĩa vụ gì?"
Document: "Quyền lợi gồm tham gia hoạt động học tập, dự án, ứng cử – đề cử, và học hỏi kỹ năng. Nghĩa vụ gồm tham gia đầy đủ, chấp hành nội quy, hoàn thành nhiệm vụ, đóng phí đúng hạn và đóng góp ý kiến xây dựng CLB."
Answer: "Khi tham gia CLB, em sẽ có rất nhiều quyền lợi hấp dẫn như được tham gia các hoạt động học tập, dự án, có cơ hội ứng cử - đề cử vào các vị trí lãnh đạo, và được học hỏi thêm nhiều kỹ năng mới. Bên cạnh đó, để CLB ngày càng phát triển, em cũng cần thực hiện một số nghĩa vụ như tham gia đầy đủ các buổi sinh hoạt, chấp hành nội quy, hoàn thành tốt nhiệm vụ được giao, đóng phí đúng hạn và tích cực đóng góp ý kiến xây dựng CLB nhé!"

User Question: "CLB có hoạt động mentoring cho thành viên mới không?"
Document: "Có, mentor sẽ hướng dẫn kỹ thuật, giải đáp thắc mắc và giúp thành viên mới làm quen với dự án."
Answer: "Chào em, CLB có hoạt động mentoring rất chu đáo cho thành viên mới đó! Mentor của CLB sẽ hướng dẫn em về kỹ thuật, giải đáp mọi thắc mắc và giúp em làm quen với các dự án một cách nhanh chóng nhất."
"""

def retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion=True):
    cache_key = (query, k, use_query_expansion, bool(reranker))
    if cache_key in _RETRIEVAL_CACHE:
        return _RETRIEVAL_CACHE[cache_key]

    # Chọn retrieval function
    use_m3 = hasattr(embedding, 'use_colbert') and embedding.use_colbert
    if use_m3:
        from m3_retriever.m3_hybrid_retriever import create_m3_hybrid_retriever
        hybrid = create_m3_hybrid_retriever(embedding, vector_db)
        def retrieve_single(q, limit):
            return hybrid.retrieve_hybrid(q, limit)
    else:
        def retrieve_single(q, limit):
            emb = embedding.encode(q)
            vec = emb.get('dense_vecs') if isinstance(emb, dict) else emb
            return vector_db.query('information', vec, limit=limit, embedding_model=embedding)

    # Mở rộng truy vấn nếu cần
    if use_query_expansion and k > 5:
        expander = QueryExpansion()
        queries = expander.expand_query(query, techniques=['synonym'], max_expansions=1)
    else:
        queries = [query]

    # Gộp kết quả với điểm số kết hợp
    all_results = []
    seen = set()
    for i, q in enumerate(queries):
        weight = 1.0 if i == 0 else max(0.0, 0.7 - (i-1)*0.1)
        limit = (k if i == 0 else max(1, k//5)) * (2 if reranker else 1)
        candidates = retrieve_single(q, limit)
        for j, doc in enumerate(candidates):
            doc_id = doc.get('title', str(hash(doc.get('information', ''))))
            if doc_id in seen:
                continue
            seen.add(doc_id)
            score = doc.get('score', 1.0)
            combined = weight * (1.0 / (j+1)) * score
            doc.update({'combined_score': combined, 'expansion_weight': weight, 'source_query': q})
            all_results.append(doc)
    # Sắp xếp và lấy top-k (hoặc top-2k nếu reranker)
    all_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
    pre_k = all_results[:k*2] if reranker else all_results[:k]

    # Rerank các kết quả nếu có reranker
    if reranker and pre_k:
        passages = [doc['information'] for doc in pre_k]
        _, reranked_passages = reranker(query, passages)
        results = [doc for rp in reranked_passages[:k] for doc in pre_k if doc['information'] == rp]
    else:
        results = pre_k

    _RETRIEVAL_CACHE[cache_key] = results
    return results

load_dotenv()

from llm_config import get_llm_response, get_config_info

config_info = get_config_info()
print(f"🤖 LLM: {config_info['model']} ({config_info['provider']})\n")

# Nên chạy từng hàm từ đoạn này để test

def hit_k(file_clb_proptit, file_train_data_proptit, embedding, vector_db, reranker=None, k=5, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train_data_proptit)

    hits = 0
    total_queries = len(df_train)

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # TODO: Nếu các em dùng Text2SQL RAG hay các phương pháp sử dụng ngôn ngữ truy vấn, có thể bỏ qua biến user_embedding
        # Các em có thể dùng các kĩ thuật để viết lại câu query, Reranking, ... ở đoạn này.
        # Retrieve top-k (with optional reranking and query expansion) using fast retrieval
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion)
         
        # Lấy danh sách tài liệu được truy suất
        retrieved_docs = [int(result['title'].split(' ')[-1]) for result in results]
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits += 1
    return hits / total_queries if total_queries > 0 else 0


# Hàm recall@k
def recall_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    
    ans = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # TODO: Nếu các em dùng Text2SQL RAG hay các phương pháp sử dụng ngôn ngữ truy vấn, có thể bỏ qua biến user_embedding
        # Các em có thể dùng các kĩ thuật để viết lại câu query, Reranking, ... ở đoạn này.
        # Embedding câu query
        # Retrieve top-k (with optional reranking and query expansion) using fast retrieval
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion)

        # Lấy danh sách tài liệu được truy suất
        retrieved_docs = [int(result['title'].split(' ')[-1]) for result in results]
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits = len([doc for doc in ground_truth_docs if doc in retrieved_docs])
        ans += hits / len(ground_truth_docs) 
    return ans / len(df_train)


# Hàm precision@k
def precision_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    ans = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Retrieve top-k (with optional reranking and query expansion)
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion)

        # Lấy danh sách tài liệu được truy suất
        retrieved_docs = [int(result['title'].split(' ')[-1]) for result in results]
        # print(f"Retrieved documents: {retrieved_docs}")
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        # print("Ground truth documents:", ground_truth_docs)
        # Kiểm tra xem có ít nhất một tài liệu đúng trong kết quả tìm kiếm
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits = len([doc for doc in retrieved_docs if doc in ground_truth_docs])
        ans += hits / k 
        # print("Hits / k for this query:", hits / k)
    return ans / len(df_train)


# Hàm f1@k
def f1_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5, use_query_expansion=True):
    precision = precision_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion)
    recall = recall_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Hàm MAP@k

def map_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_map = 0

    for index, row in df_train.iterrows():
        hits = 0
        ap = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # Tạo embedding cho câu hỏi của người dùng
        # Retrieve top-k (with optional reranking and query expansion)
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion)

        # Lấy danh sách tài liệu được truy suất
        retrieved_docs = [int(result['title'].split(' ')[-1]) for result in results]
        # print(f"Retrieved documents: {retrieved_docs}")
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        # print("Ground truth documents:", ground_truth_docs)
        
        # Tính MAP cho 1 truy vấn
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_docs:
                hits += 1
                ap += hits / (i + 1)
        if hits > 0:
            ap /= hits
        # print(f"Average Precision for this query: {ap}")
        total_map += ap 
    return total_map / len(df_train)

# Hàm MRR@k
def mrr_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_mrr = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # Tạo embedding cho câu hỏi của người dùng
        # Retrieve top-k (with optional reranking and query expansion)
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion)

        # Lấy danh sách tài liệu được truy suất
        retrieved_docs = [int(result['title'].split(' ')[-1]) for result in results]
        # print(f"Retrieved documents: {retrieved_docs}")
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        # print("Ground truth documents:", ground_truth_docs)
        
        # Tính MRR cho 1 truy vấn
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_docs:
                total_mrr += 1 / (i + 1)
                break
    return total_mrr / len(df_train) if len(df_train) > 0 else 0

# Hàm NDCG@k
import numpy as np
import torch
def dcg_at_k(relevances, k):
    relevances = np.array(relevances)[:k]
    return np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))

def ndcg_at_k(relevances, k):
    dcg_max = dcg_at_k(sorted(relevances, reverse=True), k)
    if dcg_max == 0:
        return 0.0
    return dcg_at_k(relevances, k) / dcg_max

def similarity(embedding1, embedding2):
    # Use torch cosine similarity on appropriate device for performance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = torch.tensor(embedding1, device=device, dtype=torch.float)
    b = torch.tensor(embedding2, device=device, dtype=torch.float)
    norm1 = torch.norm(a)
    norm2 = torch.norm(b)
    if norm1.item() == 0 or norm2.item() == 0:
        return 0.0
    cos_sim = torch.dot(a, b) / (norm1 * norm2)
    # Normalize to [0,1]
    return ((cos_sim + 1) / 2).item()


def ndcg_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_ndcg = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        query_embeddings = embedding.encode(query)
        if isinstance(query_embeddings, dict) and 'dense_vecs' in query_embeddings:
            user_embedding = query_embeddings['dense_vecs']
        else:
            user_embedding = query_embeddings

        # Retrieve top-k (with optional reranking and query expansion)
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion)


        retrieved_docs = [int(result['title'].split(' ')[-1]) for result in results]

        ground_truth_docs = []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))


        # Nếu điểm tương đồng > 0.9 thì gán 3, nếu > 0.7 thì gán 2, nếu > 0.5 thì gán 1, còn lại thì gán 0 
        relevances = []
        for doc in retrieved_docs:
            if doc in ground_truth_docs:
                # Giả sử ta có một hàm để tính độ tương đồng giữa câu hỏi và tài liệu, doc là số thứ tự của tài liệu trong file CLB_PROPTIT.csv
                doc_result = [r for r in results if int(r['title'].split(' ')[-1]) == doc][0]
                doc_embeddings = embedding.encode(doc_result['information'])
                # Extract dense vectors for similarity calculation
                if isinstance(doc_embeddings, dict) and 'dense_vecs' in doc_embeddings:
                    doc_embedding = doc_embeddings['dense_vecs']
                else:
                    doc_embedding = doc_embeddings
                similarity_score = similarity(user_embedding, doc_embedding)
                if similarity_score > 0.9:
                    relevances.append(3)
                elif similarity_score > 0.7:
                    relevances.append(2)
                elif similarity_score > 0.5:
                    relevances.append(1)
                else:
                    relevances.append(0)
            else:
                relevances.append(0)
        ndcg = ndcg_at_k(relevances, k)
        # print(f"NDCG for this query: {ndcg}")
        total_ndcg += ndcg

    return total_ndcg / len(df_train) if len(df_train) > 0 else 0

# Hàm Context Precision@k (LLM Judged)
def context_precision_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    # Chỉ lấy 30 hàng để test nhanh
    # sample_size = 30
    # df_train = df_train.tail(sample_size)
    # print(f"Testing with {len(df_train)} queries out of total {sample_size * 5} queries")

    total_precision = 0

    for index, row in df_train.iterrows():
        print(f"Processing query {index+1}/{len(df_train)}: {row['Query'][:50]}...")
        # TODO: Tạo ra LLM Answer, các em hãy tự viết phần system prompt
        messages = [
            {
                "role": "system",
                "content": COMMON_RAG_SYSTEM_PROMPT
            }
        ]
        hits = 0
        query = row['Query']

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu với optional reranking/query expansion
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion)
         
        # TODO: viết câu query của người dùng (bao gồm document retrieval và câu query)
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        # Thêm context vào messages
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        reply = get_llm_response(messages)

        system_judge = {
            "role": "system",
            "content": "Bạn là một trợ lý AI chuyên đánh giá độ chính xác của các câu trả lời dựa trên ngữ cảnh được cung cấp. "
                      "Bạn sẽ được cung cấp một câu hỏi, một câu trả lời, và danh sách ngữ cảnh. "
                      "Nhiệm vụ: đánh giá mức độ liên quan của mỗi ngữ cảnh với câu trả lời. "
                      f"Trả về một chuỗi liền mạch gồm {k} ký tự, mỗi ký tự là 1 nếu ngữ cảnh tương ứng liên quan, 0 nếu không, sắp xếp theo đúng thứ tự các ngữ cảnh và giải thích ngắn gọn bằng duy nhất 1 câu ở bên dưới."
        }
        
        context_sections = results
        user_content = f"Câu hỏi: {query}\nCâu trả lời: {reply}\n\nNgữ cảnh:\n"
        for idx, res in enumerate(context_sections, 1):
            user_content += f"{idx}. {res['information']}\n"
            # print(f"Context {idx}: {res['information'][:50]}...")
        user_content += f"\nHãy đánh giá mức độ liên quan cho mỗi ngữ cảnh, trả lời chuỗi gồm {k} ký tự 1 hoặc 0 theo thứ tự trên, không giải thích."
        messages_judged = [system_judge, {"role": "user", "content": user_content}]
        judged_reply = get_llm_response(messages_judged)
        raw = str(judged_reply)
        # Dùng regex để tìm chuỗi đúng định dạng
        match = re.search(rf'[01]{{{k}}}', raw)
        if match:
            flags = match.group(0)
        else:
            # Fallback: điền 0 bù
            tmp = ''.join(c for c in raw if c in '01')
            flags = (tmp + '0' * k)[:k]
        hits = flags.count('1')
        # Debug
        print(f"Judged reply: {judged_reply}")
        print(f"Flags: {flags}, Hits: {hits}")

        precision = hits / k if k > 0 else 0
        total_precision += precision
        time.sleep(1)
    return total_precision / len(df_train) if len(df_train) > 0 else 0


# Hàm Context Recall@k (LLM Judged)
def context_recall_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)
    # Sample a subset for faster testing / stable evaluation (same behavior as context_precision_k)
    # sample_size = 20
    # df_train = df_train.head(sample_size)
    # print(f"Testing context_recall_k with {len(df_train)} queries (sample_size={sample_size})")

    total_recall = 0

    for index, row in df_train.iterrows():
        query = row['Query']

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion)
        reply = row['Ground truth answer']
        

        # Build a single judge prompt that asks for a k-length 0/1 string (no explanation)
        system_judge = {
            "role": "system",
            "content": (
                "Bạn là một trợ lý AI chuyên đánh giá độ chính xác của các câu trả lời dựa trên ngữ cảnh được cung cấp. "
                "Bạn sẽ được cung cấp một câu hỏi, một câu trả lời chính xác (ground-truth), và một danh sách ngữ cảnh. "
                "Nhiệm vụ: cho biết mỗi ngữ cảnh có đủ thông tin (hoặc một phần thông tin) để trả lời câu hỏi dựa trên câu trả lời chính xác hay không. "
                "Trả về một chuỗi gồm chính xác {k} ký tự, mỗi ký tự là 1 nếu ngữ cảnh tương ứng liên quan/useful, 0 nếu không, theo đúng thứ tự các ngữ cảnh. "
                "KHÔNG giải thích thêm, chỉ trả về chuỗi 0/1."
            )
        }

        user_content = f"Câu hỏi: {query}\nCâu trả lời chính xác: {reply}\n\nNgữ cảnh:\n"
        for idx, res in enumerate(results, 1):
            user_content += f"{idx}. {res['information']}\n"
            # print(f"Context {idx}: {res['information'][:50]}...")

        messages_judged = [system_judge, {"role": "user", "content": user_content}]
        judged_reply = get_llm_response(messages_judged)

        raw = str(judged_reply)
        # Try to extract a k-length binary string
        match = re.search(rf'[01]{{{k}}}', raw)
        if match:
            flags = match.group(0)
        else:
            tmp = ''.join(c for c in raw if c in '01')
            flags = (tmp + '0' * k)[:k]

        hits = flags.count('1')
        recall = hits / k if k > 0 else 0
        total_recall += recall
        # debug
        print(f"Query {index+1}/{len(df_train)} - Flags: {flags} - Hits: {hits} - Recall: {recall:.3f}")
        time.sleep(1)

    return total_recall / len(df_train) if len(df_train) > 0 else 0

# Hàm Context Entities Recall@k (LLM Judged)
def context_entities_recall_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)
    # Sample subset for faster testing
    # sample_size = 20
    # df_train = df_train.head(sample_size)
    # print(f"Testing context_entities_recall_k with {len(df_train)} queries (sample_size={sample_size})")

    total_recall = 0
    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion)
        reply = row['Ground truth answer']
        # Trích xuất các thực thể từ Ground truth answer bằng LLM
        # NOTE: Các em có thể thay đổi messages_entities nếu muốn
        messages_entities = [
            {
                "role": "system", 
                "content": """Trích xuất tất cả thực thể quan trọng từ câu trả lời. Thực thể bao gồm: tên tổ chức, tên người, ngày tháng, địa điểm, khái niệm, thuật ngữ, số liệu.
CHỈ TRA VỀ danh sách Python hợp lệ, KHÔNG giải thích gì thêm.

Ví dụ:
Input: Câu lạc bộ Lập Trình PTIT được thành lập ngày 9/10/2011 với slogan "Lập trình từ trái tim".
Output: ["Câu lạc bộ Lập Trình PTIT", "PTIT", "9/10/2011", "Lập trình từ trái tim"]

Input: CLB có 200 thành viên đang học tại Học viện PTIT.
Output: ["CLB", "200 thành viên", "Học viện PTIT"]"""
            }
        ]
        # NOTE: Các em có thể thay đổi content nếu muốn
        messages_entities.append({
            "role": "user",
            "content": f"Câu trả lời: {reply}"
        })
        # Gọi  API để trích xuất các thực thể
        entities_str = get_llm_response(messages_entities)
        try:
            entities = ast.literal_eval(entities_str.strip())
            if not isinstance(entities, list):
                entities = []
        except Exception:
            # Fallback: extract first bracketed list and parse
            match = re.search(r'\[.*?\]', entities_str, re.DOTALL)
            if match:
                try:
                    entities = ast.literal_eval(match.group(0))
                except Exception:
                    entities = []
            else:
                entities = []
        tmp = len(entities)
        for result in results:
            context = result['information']
            for entity in entities[:]:  
                entity_clean = entity.strip()
                if entity_clean.lower() in context.lower() or any(word in context.lower() for word in entity_clean.lower().split()):
                    hits += 1
                    entities.remove(entity.strip())
        total_recall += hits / tmp if tmp > 0 else 0
        print(f"Query {index+1}/{len(df_train)} - Total Recall: {total_recall:.3f}")
        time.sleep(1)

    return total_recall / len(df_train) if len(df_train) > 0 else 0



# Hàm tính toán tất cả metrics liên quan đến Retrieval

def calculate_metrics_retrieval(file_clb_proptit, file_train , embedding, vector_db, train, reranker=None, use_query_expansion=True):
    # Tạo ra 1 bảng csv, cột thứ nhất là K value, các cột còn lại là metrics. Sẽ có 3 hàng tương trưng với k = 3, 5, 7
    k_values = [3, 5, 7]
    metrics = {
        "K": [],
        "hit@k": [],
        "recall@k": [],
        "precision@k": [],
        "f1@k": [],
        "map@k": [],
        "mrr@k": [],
        "ndcg@k": [],
        "context_precision@k": [],
        "context_recall@k": [],
        "context_entities_recall@k": []
    }
    # Lưu 2 chữ số thập phân cho các metrics
    for k in k_values:
        metrics["K"].append(k)
        metrics["hit@k"].append(round(hit_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion), 2))
        metrics["recall@k"].append(round(recall_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion), 2))
        metrics["precision@k"].append(round(precision_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion), 2))
        metrics["f1@k"].append(round(f1_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion), 2))
        metrics["map@k"].append(round(map_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion), 2))
        metrics["mrr@k"].append(round(mrr_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion), 2))
        metrics["ndcg@k"].append(round(ndcg_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion), 2))
        metrics["context_precision@k"].append(round(context_precision_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion), 2))
        metrics["context_recall@k"].append(round(context_recall_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion), 2))
        metrics["context_entities_recall@k"].append(round(context_entities_recall_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k, use_query_expansion), 2))
    # Chuyển đổi metrics thành DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
    filename_suffix = "_with_query_expansion" if use_query_expansion else "_baseline"
    if train:
        metrics_df.to_csv(f"metrics_retrieval_train{filename_suffix}.csv", index=False)
    else:
        metrics_df.to_csv(f"metrics_retrieval_test{filename_suffix}.csv", index=False)
    return metrics_df

# Các hàm đánh giá LLM Answer
def get_contexts(query, embedding, vector_db, reranker=None, use_query_expansion=True, k=5):
    return retrieve_and_rerank(query, embedding, vector_db, reranker, k, use_query_expansion)
# Hàm String Presence

def string_presence_k(file_clb_proptit, file_train, embedding, vector_db, k=5, reranker=None, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_presence = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Retrieve contexts with optional query expansion and reranking
        results = get_contexts(query, embedding, vector_db, reranker, use_query_expansion, k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": COMMON_RAG_SYSTEM_PROMPT
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])
        # Thêm context vào messages
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        response = get_llm_response(messages)
        # Trích xuất các thực thể từ câu trả lời bằng LLM
        # NOTE: Các em có thể thay đổi message_entities nếu muốn
        messages_entities = [
            {
                "role": "system",
                "content": """Trích xuất tất cả thực thể quan trọng từ câu trả lời. Thực thể bao gồm: tên tổ chức, tên người, ngày tháng, địa điểm, khái niệm, thuật ngữ, số liệu. 
CHỈ TRA VỀ danh sách Python hợp lệ, KHÔNG giải thích gì thêm.

Ví dụ:
Input: Câu lạc bộ Lập Trình PTIT được thành lập ngày 9/10/2011 với slogan "Lập trình từ trái tim".
Output: ["Câu lạc bộ Lập Trình PTIT", "PTIT", "9/10/2011", "Lập trình từ trái tim"]

Input: CLB có 200 thành viên đang học tại Học viện PTIT.  
Output: ["CLB", "200 thành viên", "Học viện PTIT"]"""
            }
        ]
        # Thay đổi content nếu muốn
        messages_entities.append({
            "role": "user",
            "content": f"Câu trả lời: {reply}"
        })
        # Gọi  API để trích xuất các thực thể
        entities_str = get_llm_response(messages_entities)
        # Trích xuất danh sách ở đầu phản hồi
        match = re.search(r'\[.*?\]', entities_str)
        if match:
            try:
                entities = ast.literal_eval(match.group())
            except Exception:
                entities = []
        else:
            entities = []
        for entity in entities:
            entity_clean = entity.strip()
            if entity_clean.lower() in response.lower() or any(word in response.lower() for word in entity_clean.lower().split()):
                hits += 1
        
        if len(entities) > 0:
            hits /= len(entities)
        else:
            hits = 0
        
        print(f"Query {index+1}/{len(df_train)} - Entities found: {hits * len(entities) if len(entities) > 0 else 0} / {len(entities)} - Presence: {hits:.3f}")
        total_presence += hits
    return total_presence / len(df_train) if len(df_train) > 0 else 0


 

# Hàm Rouge-L

from rouge import Rouge
def rouge_l_k(file_clb_proptit, file_train, embedding, vector_db, k=5, reranker=None, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    rouge = Rouge()
    total_rouge_l = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        # Retrieve contexts with optional reranking and query expansion
        results = get_contexts(query, embedding, vector_db, reranker, use_query_expansion, k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": COMMON_RAG_SYSTEM_PROMPT
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        # Thêm context vào messages
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi API để lấy câu trả lời
        response = get_llm_response(messages)
        try:
            scores = rouge.get_scores(response, reply)
            rouge_l = scores[0]['rouge-l']['f']
        except ValueError:
            rouge_l = 0.0
            print(f"Query {index+1}/{len(df_train)} - Error computing Rouge-L, set to {rouge_l:.3f}")
        total_rouge_l += rouge_l
        print(f"Query {index+1}/{len(df_train)} - Rouge-L: {rouge_l:.3f}")
    return total_rouge_l / len(df_train) if len(df_train) > 0 else 0

# Hàm BLEU-4
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu_4_k(file_clb_proptit, file_train, embedding, vector_db, k=5, reranker=None, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_bleu_4 = 0
    smoothing_function = SmoothingFunction().method1

    for index, row in df_train.iterrows():
        query = row['Query']
        # Retrieve contexts with optional reranking and query expansion
        results = get_contexts(query, embedding, vector_db, reranker, use_query_expansion, k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": COMMON_RAG_SYSTEM_PROMPT
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        # Sửa content nếu muốn
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        response = get_llm_response(messages)
        reference = reply.split()
        candidate = response.split()
        bleu_4 = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)
        total_bleu_4 += bleu_4
        print(f"Query {index+1}/{len(df_train)} - BLEU-4: {bleu_4:.3f}")
    return total_bleu_4 / len(df_train) if len(df_train) > 0 else 0

# Hàm Groundedness (LLM Answer - Hallucination Detection)

def groundedness_k(file_clb_proptit, file_train, embedding, vector_db, k=5, reranker=None, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_groundedness = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        # Retrieve contexts with optional reranking and query expansion
        results = get_contexts(query, embedding, vector_db, reranker, use_query_expansion, k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": COMMON_RAG_SYSTEM_PROMPT
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        # Thêm context vào messages, sửa content nếu muốn
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        response = get_llm_response(messages)
      
    
        # Tách response thành các câu
        import re
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        for sentence in sentences:
            # Tạo một prompt để kiểm tra tính groundedness của câu
            # Sửa prompt nếu muốn
            messages_groundedness = [
                {
                    "role": "system",
                    "content": """Bạn là chuyên gia đánh giá Groundedness trong hệ thống RAG. Nhiệm vụ: đánh giá từng câu trong response có được hỗ trợ bởi context đã cung cấp hay không.

NGUYÊN TẮC ĐÁNH GIÁ:
- supported: Câu được context hỗ trợ trực tiếp HOẶC có thể suy ra hợp lý từ context
- unsupported: Câu chứa thông tin hoàn toàn không có trong context và không thể suy ra
- contradictory: Câu mâu thuẫn trực tiếp với thông tin trong context  
- no_rag: Câu không cần kiểm tra factual (câu chào hỏi, disclaimer, câu chuyển tiếp thông thường)

QUY TẮC ĐẶC BIỆT:
1. Câu mô tả chung về quy định/chính sách (như "Bạn sẽ được tham gia khi...") → supported nếu context đề cập về điều kiện tham gia
2. Danh sách được liệt kê một phần từ context → supported  
3. Thông tin tổng hợp từ nhiều phần context → supported
4. Câu "Thông tin này không có trong tài liệu" → supported
5. Chỉ đánh giá unsupported khi câu chứa facts sai lệch

Chỉ trả lời một từ: supported/unsupported/contradictory/no_rag"""
                }
            ]
            # Sửa content nếu muốn
            messages_groundedness.append({
                "role": "user",
                "content": f"Question: {query}\n\nContexts: {context}\n\nAnswer: {sentence.strip()}"
            })
            # Gọi  API để đánh giá groundedness
            groundedness_reply = get_llm_response(messages_groundedness).lower().strip()
            words = groundedness_reply.split()

            if "supported" in words:
                hits += 1
                cnt += 1
            elif "unsupported" in words or "contradictory" in words:
                cnt += 1
        total_groundedness += hits / cnt if cnt > 0 else 0
        print(f"Query {index+1}/{len(df_train)} - Groundedness: {hits}/{cnt} - Total: {total_groundedness:.3f}")
    return total_groundedness / len(df_train) if len(df_train) > 0 else 0 

# Hàm Response Relevancy (LLM Answer - Measures relevance)


def generate_related_questions(response, embedding):
    # Sửa system prompt nếu muốn
    messages_related = [
        {
            "role": "system",
            "content": """Bạn là một trợ lý AI chuyên tạo ra các câu hỏi liên quan từ một câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là tạo ra các câu hỏi liên quan đến câu trả lời đó. Hãy tạo ra ít nhất 5 câu hỏi liên quan, mỗi câu hỏi nên ngắn gọn và rõ ràng. Trả lời dưới dạng list các câu hỏi như ở ví dụ dưới. LƯU Ý: Trả lời dưới dạng ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3", ...], bao gồm cả dấu ngoặc vuông.
            Ví dụ:
            Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
            Output của bạn: "["CLB Lập Trình PTIT được thành lập khi nào?", "Slogan của CLB là gì?", "Mục tiêu của CLB là gì?"]"
            Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
            Output của bạn: "["Ngành nào có thể tham gia CLB?", "CLB phù hợp với những ai?", "Trở ngại lớn nhất khi tham gia CLB là gì?"]"""
        }
    ]
    # Sửa content nếu muốn
    messages_related.append({
        "role": "user",
        "content": f"Câu trả lời: {response}"
    })
    # Gọi  API để tạo ra các câu hỏi liên quan
    related_questions = get_llm_response(messages_related)
    return related_questions

def response_relevancy_k(file_clb_proptit, file_train, embedding, vector_db, k=5, reranker=None, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_relevancy = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        # Retrieve contexts with optional reranking and query expansion
        results = get_contexts(query, embedding, vector_db, reranker, use_query_expansion, k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": COMMON_RAG_SYSTEM_PROMPT
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        # Sửa content nếu muốn
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        user_embedding_raw = embedding.encode(query)
        # Extract dense vectors for similarity calculation
        if isinstance(user_embedding_raw, dict) and 'dense_vecs' in user_embedding_raw:
            user_embedding = user_embedding_raw['dense_vecs']
        else:
            user_embedding = user_embedding_raw
            
        response = get_llm_response(messages)

        raw_related = generate_related_questions(response, embedding)
        related_questions = []
        if raw_related:
            raw_str = raw_related.strip()
            if raw_str.startswith('[') and raw_str.endswith('"'):
                raw_str = raw_str[:-1]
            try:
                related_questions = ast.literal_eval(raw_str)
            except Exception:
                related_questions = []
        for question in related_questions:
            question_embedding_raw = embedding.encode(question)
            # Extract dense vectors for similarity calculation
            if isinstance(question_embedding_raw, dict) and 'dense_vecs' in question_embedding_raw:
                question_embedding = question_embedding_raw['dense_vecs']
            else:
                question_embedding = question_embedding_raw
            # Tính score relevancy giữa câu hỏi và query
            score = similarity(user_embedding, question_embedding)
            hits += score
        print(f"Query {index+1}/{len(df_train)} - Related questions generated: {len(related_questions)} - Total relevancy score: {hits:.3f}")
        total_relevancy += hits / len(related_questions) if len(related_questions) > 0 else 0
    return total_relevancy / len(df_train) if len(df_train) > 0 else 0


# Hàm Noise Sensitivity (LLM Answer - Robustness to Hallucination)

def noise_sensitivity_k(file_clb_proptit, file_train, embedding, vector_db, k=5, reranker=None, use_query_expansion=True):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_sensitivity = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        # Retrieve contexts with optional reranking and query expansion
        results = get_contexts(query, embedding, vector_db, reranker, use_query_expansion, k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": COMMON_RAG_SYSTEM_PROMPT
            }
        ]
        context =  "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        # Thêm context vào messages, sửa content nếu muốn
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        response = get_llm_response(messages)

        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        sentences = [s for s in sentences if len(s.strip()) > 5]
        for sentence in sentences:
            # Sửa prompt nếu muốn
            messages_sensitivity = [
                {
                    "role": "system",
                    "content": """Bạn là một chuyên gia đánh giá độ nhạy cảm của câu trả lời trong hệ thống RAG. Nhiệm vụ: đánh giá từng câu của response có được hỗ trợ bởi context hay không.

Quy tắc đánh giá NGHIÊM NGẶT:
- 1: Câu có thể truy vết trực tiếp từ context (từ, cụm từ, ý nghĩa xuất hiện trong context)
- 0: Câu KHÔNG có trong context hoặc không thể suy ra từ context

Lưu ý đặc biệt:
- Câu chào hỏi lịch sự, câu cảm thán đơn giản -> 1
- Câu "Thông tin này không có trong tài liệu được cung cấp." -> 1 
- Bất kỳ thông tin cụ thể nào không xuất hiện trong context -> 0
- Suy luận quá xa so với context -> 0

Chỉ trả lời: 1 hoặc 0"""
                }
            ]
            # Sửa prompt nếu muốn
            messages_sensitivity.append({
                "role": "user",
                "content": f"Question: {query}\n\nContexts: {context}\n\nAnswer: {sentence.strip()}"
            })
            # Gọi  API để đánh giá độ nhạy cảm
            sensitivity_reply = get_llm_response(messages_sensitivity)
            if sensitivity_reply == "0":
                hits += 1
        print(f"Query {index+1}/{len(df_train)} - Non-supported sentences: {hits} / {len(sentences)} - Noise Sensitivity: {hits / len(sentences) if len(sentences) > 0 else 0:.3f}")
        total_sensitivity += hits / len(sentences) if len(sentences) > 0 else 0
    return total_sensitivity / len(df_train) if len(df_train) > 0 else 0


# Hàm để tính toán toàn bộ metrics trong module LLM Answer

def calculate_metrics_llm_answer(file_clb_proptit, file_train, embedding, vector_db, train, reranker=None, use_query_expansion=True):
    # Tạo ra 1 bảng csv, cột thứ nhất là K value, các cột còn lại là metrics. Sẽ có 3 hàng tương trưng với k = 3, 5, 7
    k_values = [3, 5, 7]
    metrics = {
        "K": [],
        "string_presence@k": [],
        "rouge_l@k": [],
        "bleu_4@k": [],
        "groundedness@k": [],
        "response_relevancy@k": [],
        "noise_sensitivity@k": []
    }
    # Lưu 2 chữ số thập phân cho các metrics
    for k in k_values:
        metrics["K"].append(k)
        metrics["string_presence@k"].append(round(string_presence_k(file_clb_proptit, file_train, embedding, vector_db, k, reranker, use_query_expansion), 2))
        metrics["rouge_l@k"].append(round(rouge_l_k(file_clb_proptit, file_train, embedding, vector_db, k, reranker, use_query_expansion), 2))
        metrics["bleu_4@k"].append(round(bleu_4_k(file_clb_proptit, file_train, embedding, vector_db, k, reranker, use_query_expansion), 2))
        metrics["groundedness@k"].append(round(groundedness_k(file_clb_proptit, file_train, embedding, vector_db, k, reranker, use_query_expansion), 2))
        metrics["response_relevancy@k"].append(round(response_relevancy_k(file_clb_proptit, file_train, embedding, vector_db, k, reranker, use_query_expansion), 2))
        metrics["noise_sensitivity@k"].append(round(noise_sensitivity_k(file_clb_proptit, file_train, embedding, vector_db, k, reranker, use_query_expansion), 2))
    # Chuyển đổi metrics thành DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
    if train:
        metrics_df.to_csv("metrics_llm_answer_train.csv", index=False)
    else:
        metrics_df.to_csv("metrics_llm_answer_test.csv", index=False)
    return metrics_df
