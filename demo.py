import streamlit as st
import pandas as pd
import time
from ingest_utils import build_collection_from_docx
from embeddings import Embeddings
from vector_db import VectorDatabase
from rerank import Reranker
from metrics_rag import (
    hit_k, recall_k, precision_k, context_precision_k, 
    context_recall_k, context_entities_recall_k, ndcg_k,
    retrieve_and_rerank
)
import os
from dotenv import load_dotenv

load_dotenv()
import llm_config
from llm_config import get_llm_response

# Caching functions to optimize retrieval and LLM response speed
@st.cache_data(show_spinner=False)
def cached_retrieve_and_rerank(query, k, use_query_expansion):
    return retrieve_and_rerank(
        query=query,
        embedding=st.session_state.embedding_model,
        vector_db=st.session_state.vector_db,
        reranker=st.session_state.reranker,
        k=k,
        use_query_expansion=use_query_expansion
    )

@st.cache_data(show_spinner=False)
def cached_llm_response(context, user_query, temperature, max_tokens):
    # Build prompt with context and examples
    prompt = f"""
    **Bối cảnh:**
    Bạn là một trợ lý AI chuyên gia về Câu lạc bộ Lập trình ProPTIT. Nhiệm vụ của bạn là cung cấp các câu trả lời chính xác và hữu ích dựa *duy nhất* vào thông tin được cung cấp.

    **Thông tin tham khảo:**
    ---
    {context}
    ---

    **Yêu cầu:**
    Dựa vào **duy nhất** "Thông tin tham khảo" ở trên và học theo phong cách từ các ví dụ, hãy trả lời câu hỏi sau đây của người dùng.

    **Câu hỏi:** "{user_query}"

    **Quy tắc trả lời:**
    1.  **Chính xác và Trung thực:** Chỉ sử dụng thông tin đã cho. Nếu thông tin không có trong tài liệu, hãy trả lời như "Ví dụ 2".
    2.  **Chi tiết và Rõ ràng:** Trả lời đầy đủ, chi tiết. Sử dụng gạch đầu dòng hoặc định dạng phù hợp nếu câu trả lời có nhiều ý hoặc cần liệt kê.
    3.  **Tự nhiên và Thân thiện:** Sử dụng ngôn ngữ tiếng Việt tự nhiên, giọng văn thân thiện như đang trò chuyện.

    **Câu trả lời của bạn (bằng tiếng Việt):**
    """
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý AI chuyên gia về CLB Lập trình ProPTIT, luôn trả lời bằng tiếng Việt dựa trên thông tin được cung cấp."},
        {"role": "user", "content": prompt}
    ]
    return get_llm_response(messages, temperature=temperature, max_tokens=max_tokens)

# Page config
st.set_page_config(
    page_title="NeoRAG Cup 2025 - Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.75rem;
        border-left: 4px solid #9c27b0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        color: #333333;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        transition: background-color 0.3s ease;
    }
    .assistant-message:hover {
        background-color: #f5f0f8;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'reranker' not in st.session_state:
    st.session_state.reranker = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Header
st.title("🚀 NeoRAG Cup 2025 - Demo System")
st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    
    # Default optimal configuration
    st.info("🎯 **Cấu hình mặc định:** ChromaDB + Fine-tune BGE-M3 + Fine-tune ViRanker + Query Expansion")
    
    # LLM
    st.markdown("### 🤖 LLM")
    llm_provider = "nvidia"
    llm_config.LLM_PROVIDER = llm_provider
    llm_model = "writer/palmyra-med-70b"
    llm_config.NVIDIA_MODEL = llm_model
    st.info(f"🤖 LLM Model cố định: {llm_model}")
    # Advanced LLM settings
    with st.expander("🔧 Tham số LLM", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_tokens = st.slider("Max Tokens", 256, 2048, 1024, 128)
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
    
    # Status indicators
    st.markdown("### 📊 Trạng thái hệ thống")
    if st.session_state.initialized:
        st.success("✅ Hệ thống đã sẵn sàng")
        if st.session_state.vector_db:
            try:
                doc_count = st.session_state.vector_db.count_documents("information")
                st.info(f"📚 {doc_count} documents đã được load")
            except:
                pass
    else:
        st.warning("⚠️ Chưa khởi tạo")
    
    # Initialize system button
    init_col1, init_col2 = st.columns([2, 1])
    with init_col1:
        if st.button("🔧 Khởi tạo hệ thống", type="primary", use_container_width=True):
            with st.spinner("Đang khởi tạo hệ thống..."):
                try:
                    db_type = "chromadb"
                    model_name = "halobiron/bge-m3-embedding-PROPTIT-domain-ft"
                    embedding_type = "sentence_transformers"
                    reranker_model = "halobiron/ViRanker-PROPTIT-domain-ft"
                    use_reranker = True
                    use_query_expansion = True
                    
                    # Initialize vector database
                    st.session_state.vector_db = VectorDatabase(db_type=db_type)
                    
                    # Initialize embedding model
                    st.session_state.embedding_model = Embeddings(
                        model_name=model_name,
                        type=embedding_type
                    )
                    
                    # Initialize reranker
                    st.session_state.reranker = Reranker(model_name=reranker_model)
                    
                    # Load documents if not already loaded (reusing ingest utility)
                    current_count = st.session_state.vector_db.count_documents("information")
                    if current_count == 0:
                        progress_bar = st.progress(0)
                        def on_progress(done, total):
                            progress_bar.progress(done / max(total, 1))
                        inserted = build_collection_from_docx(
                            doc_path="CLB_PROPTIT.docx",
                            embedding_model=st.session_state.embedding_model,
                            vector_db=st.session_state.vector_db,
                            collection_name="information",
                            rebuild=False,
                            progress_callback=on_progress,
                        )
                        st.success(f"Đã load {inserted} documents vào database!")
                    else:
                        st.info("Documents đã tồn tại trong database.")
                    
                    st.session_state.initialized = True
                    st.success("Hệ thống đã được khởi tạo thành công!")
                    st.rerun()  # Refresh to update status
                    
                except Exception as e:
                    st.error(f"Lỗi khi khởi tạo hệ thống: {str(e)}")
    
    with init_col2:
        if st.session_state.initialized:
            if st.button("🔄", help="Reset hệ thống"):
                st.session_state.initialized = False
                st.session_state.vector_db = None
                st.session_state.embedding_model = None
                st.session_state.reranker = None
                st.rerun()

# Main content tabs
tab1, tab2 = st.tabs(["💬 Chat Demo", "📈 Performance"])

with tab1:
    st.header("💬 Chat với hệ thống RAG")
    
    if not st.session_state.initialized:
        st.warning("⚠️ Vui lòng khởi tạo hệ thống ở sidebar trước khi sử dụng!")
    else:
        # Chat configuration
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            user_query = st.text_input(
                "Hỏi về CLB ProPTIT:",
                placeholder="Ví dụ: CLB ProPTIT được thành lập khi nào?",
                key="user_input"
            )
        
        with col2:
            top_k = st.selectbox("Top K", [3, 5, 7, 10], index=0)
        
        with col3:
            ask_button = st.button("🚀 Hỏi", type="primary")
        
        if ask_button and user_query:
            with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
                try:
                    # Retrieve relevant documents
                    start_time = time.time()
                    
                    results = cached_retrieve_and_rerank(
                        user_query,
                        top_k,
                        True
                    )
                    
                    retrieval_time = time.time() - start_time
                    
                    # Display retrieved documents
                    st.subheader("📋 Documents được truy xuất:")
                    for i, doc in enumerate(results): 
                        with st.expander(f"Document {i+1}: {doc.get('title', 'N/A')}"):
                            st.write(doc.get('information', 'N/A'))
                            if 'score' in doc:
                                st.write(f"**Score:** {doc['score']:.4f}")
                    
                    # Generate answer using selected LLM provider
                    keywords = [word.strip(',.?!"') for word in user_query.split() if len(word) > 2]
                    filtered_docs = []
                    for doc in results:
                        info = doc.get('information', '')
                        segments = [sent for sent in info.split('.') if any(kw.lower() in sent.lower() for kw in keywords)]
                        filtered = '.'.join(segments) if segments else info
                        filtered_docs.append(filtered)
                    context = "\n\n".join(filtered_docs)
                    max_context_chars = 3000
                    if len(context) > max_context_chars:
                        context = context[:max_context_chars] + "\n\n...(Nội dung đã bị cắt ngắn)..."
                    
                    answer = cached_llm_response(context, user_query, temperature, max_tokens)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "query": user_query,
                        "answer": answer,
                        "retrieval_time": retrieval_time,
                        "num_docs": len(results)
                    })
                    
                    # Display answer
                    st.subheader("🤖 Câu trả lời:")
                    st.markdown(f'<div class="assistant-message">{answer}</div>', unsafe_allow_html=True)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⏱️ Thời gian retrieval", f"{retrieval_time:.2f}s")
                    with col2:
                        st.metric("📊 Số documents", len(results))
                    with col3:
                        st.metric("🔍 Top K", top_k)
                    with col4:
                        st.metric("🤖 Model", llm_model.split('/')[-1][:15] + "...")
                        
                except Exception as e:
                    st.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📜 Lịch sử chat")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['query'][:50]}..."):
                    st.markdown(f"**Câu hỏi:** {chat['query']}")
                    st.markdown(f"**Trả lời:** {chat['answer']}")
                    st.markdown(f"**Thời gian:** {chat['retrieval_time']:.2f}s | **Docs:** {chat['num_docs']}")

with tab2:
    st.header("📈 Performance Analysis")
    
    # Metrics data
    metrics_data = {
        "Train": {
            "Retrieval": {
                'k': [3, 5, 7],
                'hit@k': [0.95, 0.99, 0.99],
                'recall@k': [0.85, 0.94, 0.95],
                'precision@k': [0.55, 0.37, 0.27],
                'f1@k': [0.67, 0.54, 0.43],
                'map@k': [0.85, 0.79, 0.78],
                'mrr@k': [0.86, 0.82, 0.81],
                'ndcg@k': [0.88, 0.85, 0.84],
                'context_precision@k': [0.90, 0.66, 0.66],
                'context_recall@k': [0.68, 0.56, 0.54],
                'context_entities_recall@k': [0.77, 0.83, 0.84]
            },
            "LLM": {
                'k': [3, 5, 7],
                'string_presence@k': [0.71, 0.75, 0.73],
                'rouge_l@k': [0.25, 0.25, 0.26],
                'bleu_4@k': [0.06, 0.05, 0.05],
                'groundedness@k': [0.94, 0.93, 0.95],
                'response_relevancy@k': [0.82, 0.82, 0.82],
                'noise_sensitivity@k': [0.14, 0.18, 0.15]
            }
        },
        "Test": {
            "Retrieval": {
                'k': [3, 5, 7],
                'hit@k': [0.97, 0.97, 0.97],
                'recall@k': [0.82, 0.89, 0.91],
                'precision@k': [0.53, 0.37, 0.28],
                'f1@k': [0.65, 0.52, 0.42],
                'map@k': [0.89, 0.84, 0.82],
                'mrr@k': [0.89, 0.86, 0.85],
                'ndcg@k': [0.91, 0.88, 0.87],
                'context_precision@k': [0.96, 0.75, 0.78],
                'context_recall@k': [0.93, 0.69, 0.72],
                'context_entities_recall@k': [0.93, 0.95, 0.97]
            },
            "LLM": {
                'k': [3, 5, 7],
                'string_presence@k': [0.82, 0.86, 0.84],
                'rouge_l@k': [0.58, 0.62, 0.58],
                'bleu_4@k': [0.38, 0.41, 0.37],
                'groundedness@k': [0.99, 1.00, 1.00],
                'response_relevancy@k': [0.81, 0.81, 0.81],
                'noise_sensitivity@k': [0.00, 0.00, 0.01]
            }
        }
    }
    
    # Display metrics for each dataset
    for dataset in ["Train", "Test"]:
        st.subheader(f"📊 {dataset} Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Retrieval Metrics**")
            df_ret = pd.DataFrame(metrics_data[dataset]["Retrieval"])
            st.dataframe(df_ret, use_container_width=True)
        
        with col2:
            st.markdown("**🤖 LLM Metrics**")
            df_llm = pd.DataFrame(metrics_data[dataset]["LLM"])
            st.dataframe(df_llm, use_container_width=True)
    
    # Individual metric testing
    if st.session_state.initialized:
        with st.expander("🔬 Test Individual Metrics", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                test_k = st.slider("K value", 1, 10, 5)
            with col2:
                metric_type = st.selectbox("Metric", [
                    "hit_k", "recall_k", "precision_k", "context_precision_k", 
                    "context_recall_k", "context_entities_recall_k", "ndcg_k"
                ])
            with col3:
                test_expansion = st.checkbox("Use Query Expansion", value=True)
            
            if st.button("🧪 Test Metric"):
                with st.spinner(f"Testing {metric_type}@{test_k}..."):
                    try:
                        if metric_type == "hit_k":
                            result = hit_k("CLB_PROPTIT.csv", "train_data_proptit.xlsx", 
                                         st.session_state.embedding_model, st.session_state.vector_db, 
                                         k=test_k, reranker=st.session_state.reranker, 
                                         use_query_expansion=test_expansion)
                        elif metric_type == "recall_k":
                            result = recall_k("CLB_PROPTIT.csv", "train_data_proptit.xlsx", 
                                            st.session_state.embedding_model, st.session_state.vector_db, 
                                            k=test_k, reranker=st.session_state.reranker, 
                                            use_query_expansion=test_expansion)
                        elif metric_type == "precision_k":
                            result = precision_k("CLB_PROPTIT.csv", "train_data_proptit.xlsx", 
                                               st.session_state.embedding_model, st.session_state.vector_db, 
                                               k=test_k, reranker=st.session_state.reranker, 
                                               use_query_expansion=test_expansion)
                        elif metric_type == "ndcg_k":
                            result = ndcg_k("CLB_PROPTIT.csv", "train_data_proptit.xlsx", 
                                          st.session_state.embedding_model, st.session_state.vector_db, 
                                          k=test_k, reranker=st.session_state.reranker, 
                                          use_query_expansion=test_expansion)
                        
                        st.success(f"**{metric_type}@{test_k}:** {result:.4f}")
                        
                        # Compare with baseline if available
                        baseline_data = metrics_data["Train"]["Retrieval"] if metric_type in ["hit_k", "recall_k", "precision_k", "ndcg_k"] else {}
                        if test_k in [3, 5, 7] and f"{metric_type}@{test_k}" in [f"{k}@{v}" for k, v in zip(baseline_data.keys(), baseline_data.values()) if k != 'k']:
                            baseline_val = baseline_data[metric_type.replace('_', '@')][baseline_data['k'].index(test_k)]
                            improvement = ((result - baseline_val) / baseline_val) * 100
                            
                            if improvement > 0:
                                st.success(f"🎉 Cải thiện {improvement:.2f}% so với baseline!")
                            else:
                                st.warning(f"📉 Thấp hơn baseline {abs(improvement):.2f}%")
                                
                    except Exception as e:
                        st.error(f"Lỗi khi test metric: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    <p>🚀 <b>NeoRAG Cup 2025</b> - Powered by Hải Long</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
