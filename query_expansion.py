import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import centralized LLM configuration  
from llm_config import get_config_info

load_dotenv()

class QueryExpansion:
    """
    Lớp QueryExpansion thực hiện các kỹ thuật mở rộng câu truy vấn để cải thiện hiệu suất retrieval trong RAG.
    
    Các kỹ thuật được áp dụng:
    1. Synonym Expansion - Mở rộng với từ đồng nghĩa sử dụng domain-specific keywords (synonym_expansion)
    2. Context-Aware Expansion - Mở rộng dựa trên ngữ cảnh và intent của câu hỏi (context_aware_expansion)
    """
    
    def __init__(self):
        # Use centralized LLM configuration
        config_info = get_config_info()
        # print(f"🤖 QueryExpansion using {config_info['provider'].upper()} - {config_info['model']}")
        self.llm_available = config_info['client_initialized']
        # Disable LLM expansion by default to save time - chỉ dùng rule-based
        self.use_llm = False
        self._cache: Dict[Any, List[str]] = {}
        
        # Format: {original: replacement} cho O(1) lookup
        self.synonym_map = {
            'clb': 'câu lạc bộ',
            'câu lạc bộ': 'clb',
            'thành viên': 'sinh viên',
            'sinh viên': 'thành viên',
            'hoạt động': 'sự kiện',
            'sự kiện': 'hoạt động',
            'team': 'nhóm',
            'nhóm': 'team',
            'đào tạo': 'học tập',
            'học tập': 'đào tạo',
        }

    def synonym_expansion(self, query: str, limit: int = 1) -> List[str]:
        query_lower = query.lower()
        
        # Single pass: tìm từ khóa và thay thế ngay
        for original, replacement in self.synonym_map.items():
            if original in query_lower:
                # Case-insensitive replacement với regex cực nhanh
                pattern = re.compile(r'\b' + re.escape(original) + r'\b', flags=re.IGNORECASE)
                new_query = pattern.sub(replacement, query, count=1)
                if new_query != query:
                    return [new_query]  # Return ngay khi tìm được variant đầu tiên
        
        return []

    def context_aware_expansion(self, query: str, limit: int = 1) -> List[str]:
        """Ultra-fast context expansion - chỉ 1 pattern matching quan trọng nhất"""
        query_lower = query.lower()

        # Thành lập/ra đời
        if "thành lập" in query:
            return [query.replace("thành lập", "ra đời")]
        
        # Tham gia/gia nhập
        if "tham gia" in query:
            return [query.replace("tham gia", "gia nhập")]
        
        # Hoạt động/sự kiện
        if "hoạt động" in query:
            return [query.replace("hoạt động", "sự kiện")]
        
        # Team/nhóm
        if "team" in query_lower and "team" in query:
            return [query.replace("team", "nhóm")]
        
        # CLB/Câu lạc bộ (case-sensitive)
        if "CLB" in query:
            return [query.replace("CLB", "Câu lạc bộ")]

        return []

    def expand_query(self, query: str, techniques: List[str] = None, max_expansions: int = 3) -> List[str]:
        """
        Ultra-optimized query expansion - giảm thiểu overhead và tối ưu tốc độ
        Default max_expansions=3 để cân bằng giữa tốc độ và chất lượng
        """
        query = query.strip()
        if not query:
            return []

        default_order = ["synonym", "context"]
        ordered_techniques = default_order if techniques is None else [tech for tech in default_order if tech in techniques]

        # Minimal cache key
        cache_key = (query.lower(), tuple(ordered_techniques))
        if cache_key in self._cache:
            return self._cache[cache_key][:max_expansions]

        expanded = [query]

        try:
            # Single pass - chỉ lấy 1 variant từ mỗi technique
            for technique in ordered_techniques:
                if len(expanded) >= max_expansions:
                    break

                if technique == "synonym":
                    variant = self.synonym_expansion(query, limit=1)
                elif technique == "context":
                    variant = self.context_aware_expansion(query, limit=1)
                else:
                    continue

                if variant and len(expanded) < max_expansions:
                    expanded.append(variant[0])
        except Exception as e:
            print(f"Error in query expansion: {e}")
            expanded = [query]

        # Cache với kích thước cố định
        if len(self._cache) > 100: 
            self._cache.clear()
        self._cache[cache_key] = expanded[:]
        return expanded[:max_expansions]