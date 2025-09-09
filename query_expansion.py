import re
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

# Import centralized LLM configuration  
from llm_config import get_llm_response as get_llm_response_global, get_config_info

load_dotenv()

class QueryExpansion:
    """
    Lớp QueryExpansion thực hiện các kỹ thuật mở rộng câu truy vấn để cải thiện hiệu suất retrieval trong RAG.
    
    Các kỹ thuật được áp dụng:
    1. Query Rewriting - Viết lại câu hỏi với nhiều cách diễn đạt khác nhau
    2. Query Decomposition - Phân tách câu hỏi phức tạp thành các câu hỏi con đơn giản
    3. Synonym/Paraphrase Expansion - Mở rộng với từ đồng nghĩa và cách diễn đạt khác
    4. Context-Aware Expansion - Mở rộng dựa trên ngữ cảnh CLB ProPTIT
    5. Multi-Perspective Query - Tạo các góc nhìn khác nhau cho cùng một câu hỏi
    """
    
    def __init__(self):
        # Use centralized LLM configuration
        config_info = get_config_info()
        print(f"🤖 QueryExpansion using {config_info['provider'].upper()} - {config_info['model']}")
        self.llm_available = config_info['client_initialized']
        
        # Domain-specific keywords cho CLB ProPTIT - được trích xuất từ dữ liệu thực tế
        self.proptit_keywords = {
            "clb": [
                'clb', 'club', 'câu lạc bộ'
            ],
            "proptit": [
                'lập trình ptit', 'proptit', 'programming ptit', 'pro'
            ],
            "thanh_vien": [
                'thành viên', 'member', 'sinh viên', 'học viên', 'người tham gia', 
                'bạn', 'em', 'anh', 'chị', 'các em', 'mọi người', 'học sinh', 'mem'
            ],
            "hoat_dong": [
                'hoạt động', 'sự kiện', 'event', 'workshop', 'seminar',
                'training', 'sinh hoạt', 'buổi học', 'đào tạo', 'tổ chức',
                'biggame', 'cuộc thi', 'contest', 'competition', 'tournament'
            ],
            "lap_trinh": [
                'lập trình', 'programming', 'code', 'coding', 'thuật toán', 'algorithm', 'CTDL',
                'phát triển phần mềm', 'software development', 'cấu trúc dữ liệu', 'giải thuật',
                'DSA', 'CTDL&GT', 'oop', 'hướng đối tượng'
            ],
            "ptit": [
                'ptit', 'học viện', 'academy', 'trường', 'university',
                'học viện bưu chính viễn thông', 'học viện công nghệ bcvt',
                'bcvt', 'hv'
            ],
            "team_du_an": [
                'team', 'nhóm', 'dự án', 'project', 'sản phẩm'
            ],
            "ky_nang": [
                'kỹ năng', 'skill', 'năng lực', 'khả năng',
                'làm việc nhóm', 'teamwork', 'thuyết trình',
                'quản lý', 'management', 'lãnh đạo', 'giao tiếp'
            ],
            "hoc_tap": [
                'học tập', 'learning', 'đào tạo', 'training', 'giáo dục', 'education',
                'khóa học', 'course', 'lộ trình', 'lộ trình học', 'roadmap', 'roadmap học tập',
                'khung chương trình lập trình', 'kỹ năng lập trình cơ bản', 'môn học', 'subject'
            ],
            "cuoc_thi_giai_thuong": [
                'cuộc thi', 'contest', 'competition', 'tournament', 'giải thưởng', 'award',
                'giải nhất', 'giải nhì', 'giải ba', 'giải khuyến khích', 'prize',
                'huy chương', 'medal', 'vàng', 'bạc', 'đồng', 'vô địch', 'champion'
            ],
            "su_kien_dac_biet": [
                'biggame', 'spoj tournament', 'procwar', 'code battle', 'game jam',
                'marathon', 'sinh nhật', 'camping', 'picnic', 'dã ngoại',
                'pama cup', 'progapp', 'icpc', 'digital race', 'Olympic'
            ],
            "cong_nghe": [
                'c++', 'java', 'python', 'javascript', 'html', 'css', 'sql',
                'học máy', 'học sâu', 'ML', 'DL', 'trí tuệ nhân tạo',
                'web', 'mobile', 'game', 'backend', 'frontend', 'fullstack',
                'react', 'angular', 'vue', 'nodejs', 'django', 'spring'
            ],
            "van_hoa_clb": [
                'phương châm', 'chia sẻ để cùng nhau phát triển', 'truyền thống',
                'văn hóa', 'gắn kết', 'đoàn kết', 'tương thân tương ái',
                'nội quy', 'quy định', 'nghĩa vụ', 'quyền lợi', 'lập trình từ trái tim',
                'tác phong', 'văn hóa ứng xử', 'quy tắc ứng xử', 'phong cách giao tiếp',
                'nét văn hóa CLB', 'etiquette', 'manners', 'ứng xử có văn hóa',
                'trang phục', 'dress code', 'đồng phục', 'quy định trang phục',
                'sử dụng trụ sở', 'cơ sở vật chất', 'phòng ban', 'phòng họp',
                'thành viên tiêu biểu', 'tiêu chí', 'điều kiện xét chọn', 
                'tiêu chuẩn đánh giá', 'cách thức bầu chọn', 'reward', 'recognition',
                'khen thưởng', 'tuyên dương', 'điểm rèn luyện', 'vi phạm'
            ]
        }
    
    def _get_llm_response(self, messages: List[Dict], max_retries: int = 3) -> str:
        """Helper function để gọi LLM với retry logic sử dụng cấu hình chung"""
        if not self.llm_available:
            print("⚠️  LLM not available, returning empty response")
            return ""
            
        return get_llm_response_global(
            messages=messages,
            temperature=0.4,
            max_tokens=256,
            max_retries=max_retries
        )
    
    def combined_llm_expansion(self, query: str) -> List[str]:
        system_prompt = """Tạo các biến thể câu hỏi cho CLB ProPTIT. Trả về format JSON:
{"rewrites": [...], "decomposed": [...], "context_aware": [...]}

Yêu cầu:
- rewrites: 2 cách viết lại khác nhau
- decomposed: Chia thành câu hỏi con (nếu phức tạp)  
- context_aware: 2 câu hỏi với ngữ cảnh CLB ProPTIT cụ thể

Ngữ cảnh CLB ProPTIT:
- Thành lập 9/10/2011, phương châm "Chia sẻ để cùng phát triển"
- 6 team: AI, Mobile, Data, Game, Web, Backend
- Quy trình: 3 vòng (CV, PV, Training), chỉ tuyển năm 1
- Sự kiện: BigGame, SPOJ, PROCWAR, Code Battle
- Lộ trình: C → C++ → CTDL&GT → OOP → Java"""
        
        user_prompt = f"Câu hỏi: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._get_llm_response(messages)
        
        try:
            import json
            # Tìm JSON trong response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                expanded = [query]
                expanded.extend(result.get("rewrites", []))
                expanded.extend(result.get("decomposed", []))
                expanded.extend(result.get("context_aware", []))
                return expanded
        except:
            pass
        
        # Fallback 
        return self.query_rewriting(query)
    
    def query_rewriting(self, query: str, num_variants: int = 2) -> List[str]:  
        system_prompt = """Viết lại câu hỏi CLB ProPTIT thành 2 cách khác. Format: ["cách 1", "cách 2"]"""
        user_prompt = f"Câu hỏi: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._get_llm_response(messages)
        
        try:
            import ast
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if list_match:
                variants = ast.literal_eval(list_match.group())
                return [query] + variants[:num_variants]
            else:
                lines = [line.strip().strip('"\'') for line in response.split('\n') if line.strip()]
                return [query] + lines[:num_variants]
        except:
            return [query, query.replace("CLB", "Câu lạc bộ"), query.replace("ProPTIT", "Lập trình PTIT")]
    
    def query_decomposition(self, query: str) -> List[str]:
        # Chia nhỏ câu hỏi phức tạp
        if ' và ' in query:
            parts = query.split(' và ')
            if len(parts) == 2:
                return [query] + [part.strip() + '?' for part in parts if part.strip()]
        
        # LLM decomposition cho câu phức tạp
        if len(query.split()) > 10:  # Chỉ dùng LLM cho câu dài
            system_prompt = """Chia câu hỏi phức tạp thành câu con đơn giản. Format: ["câu 1", "câu 2"]"""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Câu hỏi: {query}"}
            ]
            
            response = self._get_llm_response(messages)
            try:
                import ast
                list_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if list_match:
                    sub_queries = ast.literal_eval(list_match.group())
                    return [query] + sub_queries
            except:
                pass
        
        return [query]
    
    def synonym_expansion(self, query: str) -> List[str]:
        expanded_queries = [query]
        
        # Thay thế từ đồng nghĩa
        for category, synonyms in self.proptit_keywords.items():
            for synonym in synonyms[:3]: 
                pattern = re.compile(r'\b' + re.escape(synonym) + r'\b', flags=re.IGNORECASE)
                if pattern.search(query):
                    for alternative in synonyms[:2]:  
                        if alternative.lower() != synonym.lower():
                            new_query = pattern.sub(alternative, query)
                            if new_query != query and new_query not in expanded_queries:
                                expanded_queries.append(new_query)
                                if len(expanded_queries) >= 4:  
                                    return expanded_queries
                    break
        
        return expanded_queries
    
    def context_aware_expansion(self, query: str) -> List[str]:
        expanded = [query]
        query_lower = query.lower()
        
        # Phân tích ngữ cảnh câu hỏi dựa trên intent và keywords
        
        # 1. Câu hỏi về thời gian thành lập, lịch sử
        if any(word in query_lower for word in ["thành lập", "khi nào", "năm nào", "lịch sử"]):
            expanded.extend([
                "CLB ProPTIT được thành lập ngày 9/10/2011",
                "Lịch sử hình thành CLB ProPTIT",
                "Người sáng lập CLB ProPTIT"
            ])
        
        # 2. Câu hỏi về số lượng thành viên
        elif any(word in query_lower for word in ["bao nhiêu thành viên", "số lượng", "có bao nhiều"]):
            expanded.extend([
                "Quy mô thành viên CLB ProPTIT",
                "CLB tuyển 25 thành viên mỗi năm",
                "Tổng số thành viên hiện tại CLB"
            ])
        
        # 3. Câu hỏi về quy trình tham gia
        elif any(word in query_lower for word in ["tham gia", "vào clb", "đăng ký"]):
            expanded.extend([
                "Quy trình 3 vòng tuyển thành viên ProPTIT",
                "Điều kiện tham gia CLB ProPTIT",
                "Cách đăng ký vào CLB ProPTIT"
            ])
        
        # 4. Câu hỏi về hoạt động, sự kiện
        elif any(word in query_lower for word in ["hoạt động", "sự kiện", "event"]):
            expanded.extend([
                "Các sự kiện nổi bật của CLB ProPTIT",
                "BigGame và SPOJ Tournament ProPTIT",
                "Lịch hoạt động hàng năm CLB"
            ])
        
        # 5. Câu hỏi về teams, cơ cấu tổ chức
        elif any(word in query_lower for word in ["team", "nhóm", "phân chia"]) and "thành viên" not in query_lower:
            # Tạo biến thể câu hỏi thay vì trả về đáp án
            expanded.extend([
                "CLB ProPTIT có bao nhiêu team dự án?",
                "Danh sách các team dự án của CLB ProPTIT",
                "Các team dự án trong CLB ProPTIT gồm những nào?"
            ])
        
        # 6. Câu hỏi về học tập, lộ trình
        elif any(word in query_lower for word in ["học", "lộ trình", "training", "đào tạo"]):
            expanded.extend([
                "Lộ trình học tập tại ProPTIT",
                "Chương trình training C++ ProPTIT",
                "CTDL&GT và OOP trong CLB"
            ])
        
        # 7. Câu hỏi về lợi ích, giá trị
        elif any(word in query_lower for word in ["lợi ích", "có gì", "tại sao", "giá trị"]):
            expanded.extend([
                "Lợi ích khi tham gia CLB ProPTIT",
                "Kỹ năng đạt được từ CLB",
                "0.1 điểm xét học bổng ProPTIT"
            ])
        
        return expanded[:4]  
    
    def document_structure_expansion(self, query: str) -> List[str]:
        doc_keywords = {
            "thành lập": ["Lịch sử CLB ProPTIT", "9/10/2011 ProPTIT"],
            "phương châm": ["Chia sẻ để cùng phát triển", "Slogan ProPTIT"],
            "team": ["6 team ProPTIT", "Team AI, Mobile, Data, Game, Web, Backend"],
            "tuyển": ["3 vòng tuyển", "CV, Phỏng vấn, Training"],
            "training": ["Lộ trình C → C++ → CTDL&GT", "OOP Java ProPTIT"],
            "sự kiện": ["BigGame SPOJ PROCWAR", "Code Battle Game C++"]
        }
        
        expanded = [query]
        query_lower = query.lower()
        
        for keyword, expansions in doc_keywords.items():
            if keyword in query_lower:
                expanded.extend(expansions[:2])
                break
        
        return expanded[:3]
    
    def expand_query(self, query: str, techniques: List[str] = None, max_expansions: int = 8) -> List[str]:
        """
        Optimized main expansion function - ưu tiên rule-based, giảm LLM calls
        
        Args:
            query: Câu hỏi gốc
            techniques: Danh sách kỹ thuật muốn sử dụng
            max_expansions: Số lượng tối đa câu hỏi mở rộng (giảm từ 10 xuống 8)
        
        Returns:
            List[str]: Danh sách câu hỏi đã mở rộng
        """
        if techniques is None:
            # Ưu tiên rule-based techniques trước
            techniques = ["rule_based", "synonym", "context", "combined_llm"]
        
        all_expanded = [query]
        
        try:
            # 2. Synonym expansion 
            if "synonym" in techniques:
                synonyms = self.synonym_expansion(query)
                all_expanded.extend(synonyms[1:])
            
            # 3. Context-aware expansion
            if "context" in techniques:
                context_aware = self.context_aware_expansion(query)
                all_expanded.extend(context_aware[1:])
            
            # 4. Combined LLM expansion 
            if "combined_llm" in techniques and len(all_expanded) < max_expansions:
                combined = self.combined_llm_expansion(query)
                all_expanded.extend(combined[1:])
            
            # Legacy techniques
            if "decomposition" in techniques:
                decomposed = self.query_decomposition(query)
                all_expanded.extend(decomposed[1:])
            
            if "document_structure" in techniques:
                structure_aware = self.document_structure_expansion(query)
                all_expanded.extend(structure_aware[1:])
        
        except Exception as e:
            print(f"Error in query expansion: {e}")
        
        unique_expanded = []
        seen = set()
        for q in all_expanded:
            if q.lower() not in seen:
                unique_expanded.append(q)
                seen.add(q.lower())
        
        return unique_expanded[:max_expansions]
    
    def rank_expanded_queries(self, original_query: str, expanded_queries: List[str], embedding_model) -> List[str]:
        """
        Xếp hạng các câu hỏi mở rộng dựa trên độ tương đồng với câu hỏi gốc
        
        Args:
            original_query: Câu hỏi gốc
            expanded_queries: Danh sách câu hỏi đã mở rộng
            embedding_model: Model để tạo embedding
        
        Returns:
            List[str]: Danh sách câu hỏi đã được xếp hạng
        """
        import torch
        
        if len(expanded_queries) <= 1:
            return expanded_queries
        
        try:
            # Tạo embedding cho câu hỏi gốc
            original_embedding = embedding_model.encode(original_query)
            
            # Tính similarity scores
            scores = []
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            original_tensor = torch.tensor(original_embedding, device=device, dtype=torch.float)
            
            for query in expanded_queries:
                if query == original_query:
                    scores.append((query, 1.0))  # Perfect score for original
                else:
                    query_embedding = embedding_model.encode(query)
                    query_tensor = torch.tensor(query_embedding, device=device, dtype=torch.float)
                    
                    # Cosine similarity
                    norm1 = torch.norm(original_tensor)
                    norm2 = torch.norm(query_tensor)
                    if norm1.item() == 0 or norm2.item() == 0:
                        similarity = 0.0
                    else:
                        cos_sim = torch.dot(original_tensor, query_tensor) / (norm1 * norm2)
                        similarity = cos_sim.item()
                    
                    scores.append((query, similarity))
            
            # Sort by similarity (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            return [query for query, _ in scores]
        
        except Exception as e:
            print(f"Error in ranking queries: {e}")
            return expanded_queries


def test_query_expansion():
    """Function để test các kỹ thuật query expansion tối ưu"""
    expander = QueryExpansion()
    
    test_queries = [
        "CLB ProPTIT có những hoạt động gì?",
        "Làm thế nào để tham gia CLB?",
        "Những thành viên nổi bật của CLB là ai?",
        "CLB được thành lập khi nào và có bao nhiêu thành viên?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Original Query: {query}")
        print(f"{'='*50}")
        
        print("\n2. Synonym Expansion (Rule-based):")
        synonyms = expander.synonym_expansion(query)
        for i, syn in enumerate(synonyms):
            print(f"   {i+1}. {syn}")
        
        print("\n3. Context-Aware Expansion (Template-based):")
        context = expander.context_aware_expansion(query)
        for i, ctx in enumerate(context):
            print(f"   {i+1}. {ctx}")
        
        print("\n4. Combined LLM Expansion (1 API call):")
        combined = expander.combined_llm_expansion(query)
        for i, comb in enumerate(combined):
            print(f"   {i+1}. {comb}")
        
        print("\n5. Optimized All Techniques Combined:")
        all_expanded = expander.expand_query(query, max_expansions=8)
        for i, exp in enumerate(all_expanded):
            print(f"   {i+1}. {exp}")
        
        print(f"\nTotal expansions: {len(all_expanded)} (vs old version ~10-15)")


if __name__ == "__main__":
    test_query_expansion()
