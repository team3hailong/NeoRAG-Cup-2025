import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from docx import Document
import plotly.express as px
import plotly.graph_objects as go
from embeddings import Embeddings
from vector_db import VectorDatabase
from rerank import Reranker
from metrics_rag import (
    calculate_metrics_retrieval, 
    calculate_metrics_llm_answer,
    hit_k, recall_k, precision_k, context_precision_k, 
    context_recall_k, context_entities_recall_k, ndcg_k,
    retrieve_and_rerank
)
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

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
    
    # Database selection
    db_type = st.selectbox(
        "📊 Vector Database",
        ["chromadb", "mongodb", "qdrant", "supabase"],
        index=0
    )
    
    # Embedding model selection
    embedding_type = st.selectbox(
        "🧠 Embedding Model Type",
        ["sentence_transformers", "openai", "gemini", "ollama"],
        index=0
    )
    
    if embedding_type == "sentence_transformers":
        model_name = st.selectbox(
            "Model Name",
            ["BAAI/bge-m3", "Alibaba-NLP/gte-multilingual-base", "sentence-transformers/all-MiniLM-L6-v2"],
            index=0
        )
    elif embedding_type == "openai":
        model_name = st.selectbox(
            "Model Name",
            ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
            index=0
        )
    else:
        model_name = st.text_input("Model Name", "BAAI/bge-m3")
    
    # Reranker options
    use_reranker = st.checkbox("🔄 Sử dụng Reranker", value=False)
    if use_reranker:
        reranker_model = st.selectbox(
            "Reranker Model",
            ["namdp-ptit/ViRanker", "BAAI/bge-reranker-v2-m3"],
            index=0
        )
    
    # Query expansion
    use_query_expansion = st.checkbox("🔍 Query Expansion", value=True)
    
    # Retrieval parameters
    st.subheader("📋 Tham số Retrieval")
    top_k = st.slider("Top K documents", 1, 20, 5)
    
    # Initialize system button
    if st.button("🔧 Khởi tạo hệ thống", type="primary"):
        with st.spinner("Đang khởi tạo hệ thống..."):
            try:
                # Initialize vector database
                st.session_state.vector_db = VectorDatabase(db_type=db_type)
                
                # Initialize embedding model
                st.session_state.embedding_model = Embeddings(
                    model_name=model_name,
                    type=embedding_type
                )
                
                # Initialize reranker if needed
                if use_reranker:
                    st.session_state.reranker = Reranker(model_name=reranker_model)
                else:
                    st.session_state.reranker = None
                
                # Load documents if not already loaded
                if st.session_state.vector_db.count_documents("information") == 0:
                    doc = Document("CLB_PROPTIT.docx")
                    cnt = 1
                    progress_bar = st.progress(0)
                    total_paras = len([p for p in doc.paragraphs if p.text.strip()])
                    
                    for i, para in enumerate(doc.paragraphs):
                        if para.text.strip():
                            embedding_vector = st.session_state.embedding_model.encode(para.text)
                            st.session_state.vector_db.insert_document(
                                collection_name="information",
                                document={
                                    "title": f"Document {cnt}",
                                    "information": para.text,
                                    "embedding": embedding_vector
                                }
                            )
                            cnt += 1
                            progress_bar.progress((i + 1) / total_paras)
                    
                    st.success(f"Đã load {cnt-1} documents vào database!")
                else:
                    st.info("Documents đã tồn tại trong database.")
                
                st.session_state.initialized = True
                st.success("Hệ thống đã được khởi tạo thành công!")
                
            except Exception as e:
                st.error(f"Lỗi khi khởi tạo hệ thống: {str(e)}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Demo", "📊 Metrics", "📈 Performance", "📄 Dataset Info"])

with tab1:
    st.header("💬 Chat với hệ thống RAG")
    
    if not st.session_state.initialized:
        st.warning("⚠️ Vui lòng khởi tạo hệ thống ở sidebar trước khi sử dụng!")
    else:
        # Chat interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_query = st.text_input(
                "Hỏi về CLB ProPTIT:",
                placeholder="Ví dụ: CLB ProPTIT được thành lập khi nào?",
                key="user_input"
            )
        
        with col2:
            ask_button = st.button("🚀 Hỏi", type="primary")
        
        if ask_button and user_query:
            with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
                try:
                    # Retrieve relevant documents
                    start_time = time.time()
                    
                    results = retrieve_and_rerank(
                        query=user_query,
                        embedding=st.session_state.embedding_model,
                        vector_db=st.session_state.vector_db,
                        reranker=st.session_state.reranker,
                        k=top_k,
                        use_query_expansion=use_query_expansion
                    )
                    
                    retrieval_time = time.time() - start_time
                    
                    # Display retrieved documents
                    st.subheader("📋 Documents được truy xuất:")
                    for i, doc in enumerate(results[:3]):  # Show top 3
                        with st.expander(f"Document {i+1}: {doc.get('title', 'N/A')}"):
                            st.write(doc.get('information', 'N/A'))
                            if 'score' in doc:
                                st.write(f"**Score:** {doc['score']:.4f}")
                    
                    # Generate answer using Groq (if API key available)
                    context = "\n\n".join([doc.get('information', '') for doc in results])
                    
                    if os.getenv("GROQ_API_KEY"):
                        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                        
                        prompt = f"""
                        Dựa vào thông tin sau về CLB ProPTIT, hãy trả lời câu hỏi một cách chính xác và chi tiết:

                        Thông tin tham khảo:
                        {context}

                        Câu hỏi: {user_query}

                        Trả lời bằng tiếng Việt một cách tự nhiên và thân thiện:
                        """
                        
                        response = client.chat.completions.create(
                            model="meta-llama/llama-4-maverick-17b-128e-instruct",
                            messages=[
                                {"role": "system", "content": "Bạn là một trợ lý AI thông minh về CLB ProPTIT. Hãy trả lời dựa trên thông tin được cung cấp."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=500,
                            temperature=0.7
                        )
                        
                        answer = response.choices[0].message.content
                    else:
                        answer = f"Dựa vào thông tin được truy xuất, đây là những thông tin liên quan đến câu hỏi '{user_query}':\n\n{context[:500]}...\n\n(Lưu ý: Vui lòng cấu hình GROQ_API_KEY để sử dụng LLM)"
                    
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
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Thời gian retrieval", f"{retrieval_time:.2f}s")
                    with col2:
                        st.metric("📊 Số documents", len(results))
                    with col3:
                        st.metric("🔍 Top K", top_k)
                        
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
    st.header("📊 Metrics Evaluation")
    
    if not st.session_state.initialized:
        st.warning("⚠️ Vui lòng khởi tạo hệ thống trước!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            eval_type = st.selectbox("Loại đánh giá", ["Retrieval Metrics", "LLM Answer Metrics"])
        
        with col2:
            dataset_type = st.selectbox("Dataset", ["Train", "Test"])
        
        if st.button("🧮 Tính toán Metrics"):
            with st.spinner("Đang tính toán metrics..."):
                try:
                    is_train = dataset_type == "Train"
                    
                    if eval_type == "Retrieval Metrics":
                        df_metrics = calculate_metrics_retrieval(
                            "CLB_PROPTIT.csv",
                            "train_data_proptit.xlsx",
                            st.session_state.embedding_model,
                            st.session_state.vector_db,
                            is_train
                        )
                        
                        st.subheader("📈 Retrieval Metrics Results")
                        st.dataframe(df_metrics)
                        
                        # Visualize metrics
                        if not df_metrics.empty:
                            fig = px.line(
                                df_metrics, 
                                x='k', 
                                y=['hit@k', 'recall@k', 'precision@k', 'f1@k'],
                                title="Retrieval Metrics vs K"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # LLM Answer Metrics
                        df_metrics = calculate_metrics_llm_answer(
                            "CLB_PROPTIT.csv",
                            "train_data_proptit.xlsx",
                            st.session_state.embedding_model,
                            st.session_state.vector_db,
                            is_train,
                            st.session_state.reranker
                        )
                        
                        st.subheader("🤖 LLM Answer Metrics Results")
                        st.dataframe(df_metrics)
                        
                        # Visualize metrics
                        if not df_metrics.empty:
                            fig = px.line(
                                df_metrics,
                                x='k',
                                y=['string_presence@k', 'rouge_l@k', 'bleu_4@k', 'groundedness@k'],
                                title="LLM Answer Metrics vs K"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                except Exception as e:
                    st.error(f"Lỗi khi tính toán metrics: {str(e)}")

with tab3:
    st.header("📈 Performance Analysis")
    
    # Baseline comparison
    st.subheader("🎯 So sánh với Baseline")
    
    # Create baseline data from the instructions
    baseline_retrieval_train = {
        'k': [3, 5, 7],
        'hit@k': [0.31, 0.46, 0.57],
        'recall@k': [0.19, 0.28, 0.35],
        'precision@k': [0.12, 0.10, 0.09],
        'f1@k': [0.15, 0.15, 0.15],
        'map@k': [0.23, 0.23, 0.23],
        'mrr@k': [0.23, 0.27, 0.28],
        'ndcg@k': [0.25, 0.31, 0.35],
        'context_precision@k': [0.63, 0.56, 0.54],
        'context_recall@k': [0.50, 0.44, 0.40]
    }
    
    baseline_llm_train = {
        'k': [3, 5, 7],
        'string_presence@k': [0.35, 0.40, 0.41],
        'rouge_l@k': [0.21, 0.23, 0.22],
        'bleu_4@k': [0.03, 0.03, 0.04],
        'groundedness@k': [0.57, 0.61, 0.64],
        'response_relevancy@k': [0.80, 0.80, 0.80],
        'noise_sensitivity@k': [0.51, 0.53, 0.51]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Baseline Retrieval Metrics (Train)")
        df_baseline_ret = pd.DataFrame(baseline_retrieval_train)
        st.dataframe(df_baseline_ret)
        
        # Plot baseline retrieval
        fig = go.Figure()
        metrics_to_plot = ['hit@k', 'recall@k', 'precision@k', 'ndcg@k']
        for metric in metrics_to_plot:
            fig.add_trace(go.Scatter(
                x=df_baseline_ret['k'],
                y=df_baseline_ret[metric],
                mode='lines+markers',
                name=metric
            ))
        fig.update_layout(title="Baseline Retrieval Metrics", xaxis_title="K", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🤖 Baseline LLM Metrics (Train)")
        df_baseline_llm = pd.DataFrame(baseline_llm_train)
        st.dataframe(df_baseline_llm)
        
        # Plot baseline LLM
        fig = go.Figure()
        metrics_to_plot = ['string_presence@k', 'rouge_l@k', 'groundedness@k', 'response_relevancy@k']
        for metric in metrics_to_plot:
            fig.add_trace(go.Scatter(
                x=df_baseline_llm['k'],
                y=df_baseline_llm[metric],
                mode='lines+markers',
                name=metric
            ))
        fig.update_layout(title="Baseline LLM Metrics", xaxis_title="K", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual metric testing
    if st.session_state.initialized:
        st.subheader("🔬 Test Individual Metrics")
        
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
                    if test_k in [3, 5, 7] and metric_type.replace('_', '@') in baseline_retrieval_train:
                        baseline_val = baseline_retrieval_train[metric_type.replace('_', '@')][baseline_retrieval_train['k'].index(test_k)]
                        improvement = ((result - baseline_val) / baseline_val) * 100
                        
                        if improvement > 0:
                            st.success(f"🎉 Cải thiện {improvement:.2f}% so với baseline!")
                        else:
                            st.warning(f"📉 Thấp hơn baseline {abs(improvement):.2f}%")
                            
                except Exception as e:
                    st.error(f"Lỗi khi test metric: {str(e)}")

with tab4:
    st.header("📄 Dataset Information")
    
    # Dataset overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Document Statistics")
        try:
            doc = Document("CLB_PROPTIT.docx")
            total_paragraphs = len([p for p in doc.paragraphs if p.text.strip()])
            total_chars = sum(len(p.text) for p in doc.paragraphs if p.text.strip())
            avg_para_length = total_chars / total_paragraphs if total_paragraphs > 0 else 0
            
            st.metric("📝 Tổng paragraphs", total_paragraphs)
            st.metric("🔤 Tổng ký tự", f"{total_chars:,}")
            st.metric("📏 Độ dài trung bình", f"{avg_para_length:.0f} chars")
            
        except Exception as e:
            st.error(f"Không thể đọc file DOCX: {str(e)}")
    
    with col2:
        st.subheader("🎯 Query Statistics")
        try:
            df_train = pd.read_excel("train_data_proptit.xlsx")
            st.metric("❓ Tổng queries (train)", len(df_train))
            
            if 'question' in df_train.columns:
                avg_query_length = df_train['question'].str.len().mean()
                st.metric("📏 Độ dài query TB", f"{avg_query_length:.0f} chars")
            
        except Exception as e:
            st.error(f"Không thể đọc file train data: {str(e)}")
    
        # Database status
        if st.session_state.initialized and st.session_state.vector_db:
            st.subheader("💾 Database Status")
            try:
                doc_count = st.session_state.vector_db.count_documents("information")
                status = ( "✅ Sẵn sàng"
                    if doc_count > 0 else
                    "⚠️ Trống"
                )
                st.metric(f"📚 Documents in DB | {status}", f"{doc_count}")
            except Exception as e:
                st.error(f"Lỗi khi kiểm tra database: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    <p>🚀 <b>NeoRAG Cup 2025</b> - Powered by Hải Long</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
