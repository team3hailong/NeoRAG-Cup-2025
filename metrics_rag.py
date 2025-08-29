import pandas as pd
import re
import openai
import os
import ast
from dotenv import load_dotenv
import requests
import time
from google import genai
from rerank import Reranker

# Helper to retrieve and optionally rerank results
def retrieve_and_rerank(query, embedding, vector_db, reranker, k):
    user_embedding = embedding.encode(query)
    initial_limit = k * 2 if reranker else k
    results = vector_db.query("information", user_embedding, limit=initial_limit)
    if reranker and results:
        passages = [res['information'] for res in results]
        ranked_scores, ranked_passages = reranker(query, passages)
        reranked_results = []
        for rp in ranked_passages[:k]:
            for res in results:
                if res['information'] == rp:
                    reranked_results.append(res)
                    break
        return reranked_results
    return results[:k]

load_dotenv()

# Option 1: Ollama (local)
# client = openai.OpenAI(
#     base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
#     api_key=os.getenv("OLLAMA_API_KEY", "ollama")
# )
# # Default model name for Ollama; change via environment variable OLLAMA_MODEL if needed.
# MODEL_NAME = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

# Option 2: Gemini Pro 
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# MODEL_NAME = "gemini-1.5-flash"

from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
MODEL_NAME="meta-llama/llama-4-maverick-17b-128e-instruct"

# 🔧 HELPER FUNCTION: Wrapper để hỗ trợ cả OpenAI và Gemini, có thể thay đổi temperature, max_tokens
def get_llm_response(messages, model_name=MODEL_NAME):
    max_retries = 3
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.35,
                max_completion_tokens=848,
                top_p=1,
                stream=False,
                stop=None
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
    return ""

# Nên chạy từng hàm từ đoạn này để test

def hit_k(file_clb_proptit, file_train_data_proptit, embedding, vector_db, reranker=None, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train_data_proptit)

    hits = 0
    total_queries = len(df_train)

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # TODO: Nếu các em dùng Text2SQL RAG hay các phương pháp sử dụng ngôn ngữ truy vấn, có thể bỏ qua biến user_embedding
        # Các em có thể dùng các kĩ thuật để viết lại câu query, Reranking, ... ở đoạn này.
        # Retrieve top-k (with optional reranking)
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k)
         
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
def recall_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5):
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
        # Retrieve top-k (with optional reranking)
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k)

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
def precision_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    ans = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Retrieve top-k (with optional reranking)
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k)

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
def f1_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5):
    precision = precision_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k)
    recall = recall_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Hàm MAP@k

def map_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_map = 0

    for index, row in df_train.iterrows():
        hits = 0
        ap = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # Tạo embedding cho câu hỏi của người dùng
        # Retrieve top-k (with optional reranking)
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k)

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
def mrr_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_mrr = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # Tạo embedding cho câu hỏi của người dùng
        # Retrieve top-k (with optional reranking)
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k)

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


def ndcg_k(file_clb_proptit, file_train, embedding, vector_db, reranker=None, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_ndcg = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Retrieve top-k (with optional reranking)
        results = retrieve_and_rerank(query, embedding, vector_db, reranker, k)


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
                doc_embedding = embedding.encode(doc_result['information'])
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
def context_precision_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)
    
    # Chỉ lấy 20 hàng để test nhanh
    sample_size = 20
    df_train = df_train.head(sample_size)
    print(f"Testing with {len(df_train)} queries out of total {sample_size * 5} queries")

    total_precision = 0

    for index, row in df_train.iterrows():
        print(f"Processing query {index+1}/{len(df_train)}: {row['Query'][:50]}...")
        # TODO: Tạo ra LLM Answer, các em hãy tự viết phần system prompt
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
4. Tuyệt đối không suy đoán hoặc bịa thông tin.
5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

Nhiệm vụ của bạn:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác."""
            }
        ]
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information", user_embedding, limit=k)
        
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
def context_recall_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)
    # Sample a subset for faster testing / stable evaluation (same behavior as context_precision_k)
    sample_size = 20
    df_train = df_train.head(sample_size)
    print(f"Testing context_recall_k with {len(df_train)} queries (sample_size={sample_size})")

    total_recall = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information", user_embedding, limit=k)
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
def context_entities_recall_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)
    # Sample subset for faster testing
    sample_size = 20
    df_train = df_train.head(sample_size)
    print(f"Testing context_entities_recall_k with {len(df_train)} queries (sample_size={sample_size})")

    total_recall = 0
    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information", user_embedding, limit=k)
        reply = row['Ground truth answer']
        # Trích xuất các thực thể từ Ground truth answer bằng LLM
        # NOTE: Các em có thể thay đổi messages_entities nếu muốn
        messages_entities = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên trích xuất các thực thể từ câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là trích xuất các thực thể từ câu trả lời đó. Các thực thể có thể là tên người, địa điểm, tổ chức, sự kiện, v.v. Hãy trả lời dưới dạng một danh sách các thực thể.
                Ví dụ:
                Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
                ["ngành khác", "CLB", "CNTT", "mảng]
                Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
                ["Câu lạc bộ Lập Trình PTIT (Programming PTIT)", "PROPTIT", "9/10/2011", "Chia sẻ để cùng nhau phát triển", "sinh viên", "Học viện", "Lập Trình PTIT - Lập trình từ trái tim"]"""
            }
        ]
        # NOTE: Các em có thể thay đổi content nếu muốn
        messages_entities.append({
            "role": "user",
            "content": f"Câu trả lời: {reply}"
        })
        # Gọi  API để trích xuất các thực thể
        entities = get_llm_response(messages_entities)
        entities = eval(entities) if entities.startswith('[') else []
        tmp = len(entities)
        for result in results:
            context = result['information']
            for entity in entities:
                if entity.strip() in context:
                    hits += 1
                    entities.remove(entity.strip())
        total_recall += hits / tmp if tmp > 0 else 0
        print(f"Query {index+1}/{len(df_train)} - Entities extracted: {tmp} - Hits: {hits}")
        time.sleep(1)

    return total_recall / len(df_train) if len(df_train) > 0 else 0



# Hàm tính toán tất cả metrics liên quan đến Retrieval

def calculate_metrics_retrieval(file_clb_proptit, file_train , embedding, vector_db, train, reranker=None):
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
        metrics["hit@k"].append(round(hit_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k), 2))
        metrics["recall@k"].append(round(recall_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k), 2))
        metrics["precision@k"].append(round(precision_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k), 2))
        metrics["f1@k"].append(round(f1_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k), 2))
        metrics["map@k"].append(round(map_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k), 2))
        metrics["mrr@k"].append(round(mrr_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k), 2))
        metrics["ndcg@k"].append(round(ndcg_k(file_clb_proptit, file_train, embedding, vector_db, reranker, k), 2))
        metrics["context_precision@k"].append(round(context_precision_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["context_recall@k"].append(round(context_recall_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["context_entities_recall@k"].append(round(context_entities_recall_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
    # Chuyển đổi metrics thành DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
    if train:
        metrics_df.to_csv("metrics_retrieval_train.csv", index=False)
    else:
        metrics_df.to_csv("metrics_retrieval_test.csv", index=False)
    return metrics_df

# Các hàm đánh giá LLM Answer

# Hàm String Presence

def string_presence_k(file_clb_proptit, file_train, embedding, vector_db,  k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_presence = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information", user_embedding, limit=k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

Nguyên tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
2. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
3. Tuyệt đối không suy đoán hoặc bịa thông tin.
4. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
5. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

Nhiệm vụ của bạn:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác."""
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
                "content": """Bạn là một trợ lý AI chuyên trích xuất các thực thể từ câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là trích xuất các thực thể từ câu trả lời đó. Các thực thể có thể là tên người, địa điểm, tổ chức, sự kiện, v.v. Hãy trả lời dưới dạng một danh sách các thực thể.
                Ví dụ:
                Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
                ["ngành khác", "CLB", "CNTT", "mảng]
                Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
                ["Câu lạc bộ Lập Trình PTIT (Programming PTIT)", "PROPTIT", "9/10/2011", "Chia sẻ để cùng nhau phát triển", "sinh viên", "Học viện", "Lập Trình PTIT - Lập trình từ trái tim"]"""
            }
        ]
        # Thay đổi content nếu muốn
        messages_entities.append({
            "role": "user",
            "content": f"Câu trả lời: {reply}"
        })
        # Gọi  API để trích xuất các thực thể
        entities = get_llm_response(messages_entities)
        entities = eval(entities) if entities.startswith('[') else []
        for entity in entities:
            if entity.strip() in response:
                hits += 1
                # print(f"Entity '{entity.strip()}' found in response.")
        hits /= len(entities) if len(entities) > 0 else 0
        total_presence += hits
    return total_presence / len(df_train) if len(df_train) > 0 else 0


 

# Hàm Rouge-L

from rouge import Rouge
def rouge_l_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    rouge = Rouge()
    total_rouge_l = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information", user_embedding, limit=k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

Nguyên tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
4. Tuyệt đối không suy đoán hoặc bịa thông tin.
5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

Nhiệm vụ của bạn:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác."""
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
        scores = rouge.get_scores(response, reply)
        rouge_l = scores[0]['rouge-l']['f']
        total_rouge_l += rouge_l
    return total_rouge_l / len(df_train) if len(df_train) > 0 else 0

# Hàm BLEU-4
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu_4_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_bleu_4 = 0
    smoothing_function = SmoothingFunction().method1

    for index, row in df_train.iterrows():
        query = row['Query']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information", user_embedding, limit=k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

Nguyên tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
4. Tuyệt đối không suy đoán hoặc bịa thông tin.
5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

Nhiệm vụ của bạn:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác."""
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
    return total_bleu_4 / len(df_train) if len(df_train) > 0 else 0

# Hàm Groundedness (LLM Answer - Hallucination Detection)\

def groundedness_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_groundedness = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information", user_embedding, limit=k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

Nguyên tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
4. Tuyệt đối không suy đoán hoặc bịa thông tin.
5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

Nhiệm vụ của bạn:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác."""
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
        sentences = response.split('. ')
        for sentence in sentences:
            # Tạo một prompt để kiểm tra tính groundedness của câu
            # Sửa prompt nếu muốn
            messages_groundedness = [
                {
                    "role": "system",
                    "content": """Bạn là một chuyên gia đánh giá Groundedness trong hệ thống RAG, có nhiệm vụ phân loại từng câu của câu trả lời dựa trên ngữ cảnh đã cho.
                    Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trong phần trả lời từ mô hình AI. Nhiệm vụ của bạn là đánh giá câu trả lời đó dựa trên ngữ cảnh và câu hỏi.
                    Input:
                    Question: Câu hỏi của người dùng
                    Contexts: Một hoặc nhiều đoạn văn bản được truy xuất
                    Answer: Chỉ một câu trong đoạn văn bản LLM sinh ra
                    Bạn hãy đánh giá dựa trên các nhãn sau: 
                    supported: Nội dung câu được ngữ cảnh hỗ trợ hoặc suy ra trực tiếp.
                    unsupported: Nội dung câu không được ngữ cảnh hỗ trợ, và không thể suy ra từ đó.
                    contradictory: Nội dung câu trái ngược hoặc mâu thuẫn với ngữ cảnh.
                    no_rad: Câu không yêu cầu kiểm tra thực tế (ví dụ: câu chào hỏi, ý kiến cá nhân, câu hỏi tu từ, disclaimers).
                    Hãy trả lời bằng một trong các nhãn trên, không giải thích gì thêm. Chỉ trả lời một từ duy nhất là nhãn đó.
                    Ví dụ:
                    Question: Bạn có thể cho tôi biết về lịch sử của Câu lạc bộ Lập trình ProPTIT không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011.
                    supported"""
                }
            ]
            # Sửa content nếu muốn
            messages_groundedness.append({
                "role": "user",
                "content": f"Question: {query}\n\nContexts: {context}\n\nAnswer: {sentence.strip()}"
            })
            # Gọi  API để đánh giá groundedness
            groundedness_reply = get_llm_response(messages_groundedness)

            if groundedness_reply == "supported":
                hits += 1
                cnt += 1
            elif groundedness_reply == "unsupported" or groundedness_reply == "contradictory":
                cnt += 1
        total_groundedness += hits / cnt if cnt > 0 else 0
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

def response_relevancy_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_relevancy = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information", user_embedding, limit=k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

Nguyên tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
4. Tuyệt đối không suy đoán hoặc bịa thông tin.
5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

Nhiệm vụ của bạn:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác."""
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

        # Dùng câu trả lời của LLM để sinh ra các câu hỏi liên quan
        related_questions = generate_related_questions(response, embedding) # "["CLB Lập Trình PTIT được thành lập khi nào?", "Slogan của CLB là gì?", "Mục tiêu của CLB là gì?"]"
        related_questions = eval(related_questions) if related_questions else []  # Chuyển đổi chuỗi thành danh sách
        for question in related_questions:
            question_embedding = embedding.encode(question)
            # Tính score relevancy giữa câu hỏi và query
            score = similarity(user_embedding, question_embedding)
            hits += score
        total_relevancy += hits / len(related_questions) if len(related_questions) > 0 else 0
    return total_relevancy / len(df_train) if len(df_train) > 0 else 0


# Hàm Noise Sensitivity (LLM Answer - Robustness to Hallucination)

def noise_sensitivity_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_sensitivity = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information", user_embedding, limit=k)
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

Nguyên tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
4. Tuyệt đối không suy đoán hoặc bịa thông tin.
5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

Nhiệm vụ của bạn:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác."""
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

        sentences = response.split('. ')
        for sentence in sentences:
            # Sửa prompt nếu muốn
            messages_sensitivity = [
                {
                    "role": "system",
                    "content": """Bạn là một chuyên gia đánh giá độ nhạy cảm của câu trả lời trong hệ thống RAG, có nhiệm vụ phân loại từng câu của câu trả lời dựa trên ngữ cảnh đã cho.
                    Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trong phần trả lời từ mô hình AI. Nhiệm vụ của bạn là đánh giá câu trả lời đó dựa trên ngữ cảnh và câu hỏi.
                    Input:
                    Question: Câu hỏi của người dùng
                    Contexts: Một hoặc nhiều đoạn văn bản được truy xuất
                    Answer: Chỉ một câu trong đoạn văn bản LLM sinh ra
                    Bạn hãy đánh giá dựa trên các nhãn sau: 
                    1: Nội dung câu được ngữ cảnh hỗ trợ hoặc suy ra trực tiếp.
                    0: Nội dung câu không được ngữ cảnh hỗ trợ, và không thể suy ra từ đó.
                    Ví dụ:
                    Question: Bạn có thể cho tôi biết về lịch sử của Câu lạc bộ Lập trình ProPTIT không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011.
                    1
                    Question: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011. Bạn có biết ngày cụ thể không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT là CLB thuộc PTIT.
                    0"""
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
        total_sensitivity += hits / len(sentences) if len(sentences) > 0 else 0
    return total_sensitivity / len(df_train) if len(df_train) > 0 else 0


# Hàm để tính toán toàn bộ metrics trong module LLM Answer

def calculate_metrics_llm_answer(file_clb_proptit, file_train, embedding, vector_db, train, reranker=None):
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
        metrics["string_presence@k"].append(round(string_presence_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["rouge_l@k"].append(round(rouge_l_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["bleu_4@k"].append(round(bleu_4_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["groundedness@k"].append(round(groundedness_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["response_relevancy@k"].append(round(response_relevancy_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["noise_sensitivity@k"].append(round(noise_sensitivity_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
    # Chuyển đổi metrics thành DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
    if train:
        metrics_df.to_csv("metrics_llm_answer_train.csv", index=False)
    else:
        metrics_df.to_csv("metrics_llm_answer_test.csv", index=False)
    return metrics_df

