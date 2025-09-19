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
    1. LLM-Based Expansion - Sử dụng LLM để tạo các cách hỏi khác nhau (combined_llm_expansion)
    2. Synonym Expansion - Mở rộng với từ đồng nghĩa sử dụng domain-specific keywords (synonym_expansion)
    3. Context-Aware Expansion - Mở rộng dựa trên ngữ cảnh và intent của câu hỏi (context_aware_expansion)
    4. Weighted Query Ranking - Xếp hạng các query mở rộng theo độ tương đồng với query gốc (rank_expanded_queries)
    """
    
    def __init__(self):
        # Use centralized LLM configuration
        config_info = get_config_info()
        # print(f"🤖 QueryExpansion using {config_info['provider'].upper()} - {config_info['model']}")
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
        # Simplified prompt để giảm complexity và tăng success rate
        system_prompt = """Tạo 2 cách hỏi khác nhau cho câu hỏi về CLB ProPTIT. 
Trả về format: ["cách hỏi 1", "cách hỏi 2"]
Chỉ thay đổi cách diễn đạt, giữ nguyên ý nghĩa."""
        
        user_prompt = f"Câu hỏi: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._get_llm_response(messages)
        
        try:
            import ast
            # Tìm list trong response
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if list_match:
                variants = ast.literal_eval(list_match.group())
                if isinstance(variants, list) and len(variants) >= 2:
                    expanded = [query]
                    expanded.extend(variants[:2])  # Chỉ lấy 2 variants
                    return expanded
        except Exception as e:
            print(f"LLM expansion parsing error: {e}")
        
        # Fallback: rule-based simple rewrites
        fallback_variants = [
            query.replace("CLB", "Câu lạc bộ"),
            query.replace("ProPTIT", "Lập trình PTIT"),
        ]
        return [query] + [v for v in fallback_variants if v != query][:2]
    
    def synonym_expansion(self, query: str) -> List[str]:
        expanded_queries = [query]
        query_lower = query.lower()
        
        best_synonyms = []
        
        for category, synonyms in self.proptit_keywords.items():
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    for alternative in synonyms:
                        if (alternative.lower() != synonym.lower() and 
                            len(alternative) >= 3):  # Tránh từ quá ngắn
                            pattern = re.compile(r'\b' + re.escape(synonym) + r'\b', flags=re.IGNORECASE)
                            new_query = pattern.sub(alternative, query)
                            if new_query != query and new_query not in expanded_queries:
                                best_synonyms.append(new_query)
                            break  # Chỉ lấy 1 alternative tốt nhất cho mỗi synonym
                    break  # Chỉ xử lý 1 synonym per category
        
        # Chọn 1-2 synonym expansion tốt nhất
        expanded_queries.extend(best_synonyms[:2])
        
        return expanded_queries
    
    def context_aware_expansion(self, query: str) -> List[str]:
        expanded = [query]
        query_lower = query.lower()
        
        # Intent-based query expansion với variants thực tế hơn
        context_expansions = []
        
        # 1. Câu hỏi về thời gian thành lập, lịch sử
        if any(word in query_lower for word in ["thành lập", "khi nào", "năm nào", "lịch sử", "thành lập", "bắt đầu"]):
            context_expansions = [
                query.replace("thành lập", "ra đời").replace("khi nào", "năm nào"),
                query + " và phương châm hoạt động"
            ]
        
        # 2. Câu hỏi về quy trình tham gia
        elif any(word in query_lower for word in ["tham gia", "vào clb", "đăng ký", "gia nhập"]):
            context_expansions = [
                query.replace("tham gia", "gia nhập"),
                query.replace("CLB", "câu lạc bộ lập trình ProPTIT")
            ]
        
        # 3. Câu hỏi về hoạt động, sự kiện  
        elif any(word in query_lower for word in ["hoạt động", "sự kiện", "event", "tổ chức"]):
            context_expansions = [
                query.replace("hoạt động", "sự kiện"),
                query + " hàng năm của CLB"
            ]
        
        # 4. Câu hỏi về team, cơ cấu
        elif any(word in query_lower for word in ["team", "nhóm", "phân chia", "cơ cấu"]):
            context_expansions = [
                query.replace("team", "nhóm dự án"),
                query.replace("CLB ProPTIT", "câu lạc bộ")
            ]
        
        # 5. Câu hỏi về học tập, lộ trình
        elif any(word in query_lower for word in ["học", "lộ trình", "training", "đào tạo", "chương trình"]):
            context_expansions = [
                query.replace("lộ trình", "chương trình đào tạo"),
                query.replace("học", "training")
            ]
        
        # 6. Câu hỏi chung hoặc không match pattern cụ thể
        else:
            # Generic expansions
            if "CLB" in query:
                context_expansions.append(query.replace("CLB", "Câu lạc bộ"))
            if "ProPTIT" in query:
                context_expansions.append(query.replace("ProPTIT", "Lập trình PTIT"))
        
        # Lọc và chỉ lấy expansions khác với câu gốc
        valid_expansions = [exp for exp in context_expansions if exp != query and exp.strip()]
        expanded.extend(valid_expansions[:2])  # Chỉ lấy 2 expansions tốt nhất
        
        return expanded
    
    def expand_query(self, query: str, techniques: List[str] = None, max_expansions: int = 5) -> List[str]:
        if techniques is None:
            techniques = ["synonym", "context", "combined_llm"]
        
        all_expanded = [query]  # Câu gốc luôn đứng đầu
        
        try:
            # 1. Rule-based synonym expansion (nhanh, không cần LLM)
            if "synonym" in techniques:
                synonyms = self.synonym_expansion(query)
                all_expanded.extend(synonyms[1:2])  # Chỉ lấy 1 synonym tốt nhất
            
            # 2. Context-aware expansion (template-based, không cần LLM)
            if "context" in techniques:
                context_aware = self.context_aware_expansion(query)
                all_expanded.extend(context_aware[1:3])  # Lấy 2 context variants
            
            # 3. LLM expansion (chỉ khi cần thiết và chưa đủ)
            if "combined_llm" in techniques and len(all_expanded) < max_expansions:
                remaining_slots = max_expansions - len(all_expanded)
                if remaining_slots > 0:
                    combined = self.combined_llm_expansion(query)
                    all_expanded.extend(combined[1:remaining_slots + 1])
        
        except Exception as e:
            print(f"Error in query expansion: {e}")
            # Fallback: chỉ trả về câu gốc nếu có lỗi
            return [query]
        
        unique_expanded = []
        seen = set()
        for q in all_expanded:
            q_norm = q.lower().strip()
            if q_norm not in seen and q.strip():
                unique_expanded.append(q)
                seen.add(q_norm)
        
        # Đảm bảo câu gốc luôn đầu tiên
        if unique_expanded[0] != query:
            if query in unique_expanded:
                unique_expanded.remove(query)
            unique_expanded.insert(0, query)
        
        return unique_expanded[:max_expansions]