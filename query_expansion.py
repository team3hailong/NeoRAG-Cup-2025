import re
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from groq import Groq

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
    
    def __init__(self, model_name="meta-llama/llama-4-maverick-17b-128e-instruct"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name
        
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
                'bạn', 'em', 'anh', 'chị', 'các em', 'mọi người', 'học sinh'
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
        """Helper function để gọi LLM với retry logic"""
        backoff = 1
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,  # Tăng temperature để tạo đa dạng
                    max_completion_tokens=512,
                    top_p=0.9,
                    stream=False,
                    stop=None
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling LLM (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
        return ""
    
    def query_rewriting(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Technique 1: Query Rewriting
        Tạo ra các cách diễn đạt khác nhau cho cùng một câu hỏi
        """
        system_prompt = """Bạn là một chuyên gia ngôn ngữ tiếng Việt, chuyên viết lại câu hỏi với nhiều cách diễn đạt khác nhau.
        Nhiệm vụ: Viết lại câu hỏi ban đầu thành nhiều phiên bản khác nhau nhưng giữ nguyên ý nghĩa.
        
        Yêu cầu:
        - Giữ nguyên ý nghĩa của câu hỏi gốc
        - Sử dụng từ ngữ đa dạng, phong cách khác nhau (trang trọng, thân thiện, trực tiếp)
        - Phù hợp với ngữ cảnh CLB Lập trình ProPTIT
        - Trả về dưới dạng danh sách Python, mỗi phần tử là một cách viết lại
        
        Ví dụ:
        Input: "CLB ProPTIT có những hoạt động gì?"
        Output: ["Câu lạc bộ ProPTIT tổ chức những sự kiện nào?", "ProPTIT có những chương trình hoạt động gì?", "Các hoạt động của CLB Lập trình ProPTIT là gì?"]
        """
        
        user_prompt = f"Hãy viết lại câu hỏi sau thành {num_variants} phiên bản khác nhau:\n\nCâu hỏi: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._get_llm_response(messages)
        
        try:
            # Tìm và parse danh sách Python từ response
            import ast
            # Tìm pattern list trong response
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if list_match:
                variants = ast.literal_eval(list_match.group())
                return [query] + variants[:num_variants]  # Bao gồm query gốc
            else:
                # Fallback: tách theo dòng
                lines = [line.strip().strip('"\'') for line in response.split('\n') if line.strip()]
                return [query] + lines[:num_variants]
        except:
            # Fallback đơn giản
            return [query, query.replace("CLB", "Câu lạc bộ"), query.replace("ProPTIT", "Lập trình PTIT")]
    
    def query_decomposition(self, query: str) -> List[str]:
        """
        Technique 2: Query Decomposition
        Phân tách câu hỏi phức tạp thành các câu hỏi con đơn giản
        """
        system_prompt = """Bạn là một chuyên gia phân tích câu hỏi, chuyên phân tách câu hỏi phức tạp thành các câu hỏi con đơn giản.
        
        Nhiệm vụ: Phân tích câu hỏi và chia thành các câu hỏi con độc lập, đơn giản hơn.
        
        Nguyên tắc:
        - Mỗi câu hỏi con chỉ tập trung vào một khía cạnh
        - Câu hỏi con phải đơn giản, dễ hiểu
        - Tập hợp câu hỏi con phải bao phủ đầy đủ câu hỏi gốc
        - Phù hợp với ngữ cảnh CLB Lập trình ProPTIT
        - Trả về dưới dạng danh sách Python
        
        Ví dụ:
        Input: "CLB ProPTIT được thành lập khi nào và có những hoạt động chính nào?"
        Output: ["CLB ProPTIT được thành lập khi nào?", "CLB ProPTIT có những hoạt động chính nào?"]
        
        Input: "Tôi muốn tham gia CLB ProPTIT, cần điều kiện gì và quy trình như thế nào?"
        Output: ["Điều kiện để tham gia CLB ProPTIT là gì?", "Quy trình tham gia CLB ProPTIT như thế nào?"]
        """
        
        user_prompt = f"Hãy phân tách câu hỏi sau thành các câu hỏi con đơn giản:\n\nCâu hỏi: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._get_llm_response(messages)
        
        try:
            import ast
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if list_match:
                sub_queries = ast.literal_eval(list_match.group())
                return [query] + sub_queries
            else:
                # Fallback
                lines = [line.strip().strip('"-') for line in response.split('\n') if line.strip() and '?' in line]
                return [query] + lines[:3]
        except:
            return [query]
    
    def synonym_expansion(self, query: str) -> List[str]:
        """
        Technique 3: Synonym/Paraphrase Expansion
        Mở rộng với từ đồng nghĩa và các cách diễn đạt khác
        """
        expanded_queries = [query]
        
        # Thay thế từ đồng nghĩa dựa trên domain keywords, chỉ thay thế khi khớp từ nguyên
        for category, synonyms in self.proptit_keywords.items():
            for synonym in synonyms:
                # Chỉ khớp khi là từ nguyên (word boundary)
                pattern = re.compile(r'\b' + re.escape(synonym) + r'\b', flags=re.IGNORECASE)
                if pattern.search(query):
                    for alternative in synonyms:
                        if alternative.lower() != synonym.lower():
                            new_query = pattern.sub(alternative, query)
                            if new_query != query and new_query not in expanded_queries:
                                expanded_queries.append(new_query)
        
        # Sử dụng LLM để tạo thêm paraphrases
        system_prompt = """Bạn là chuyên gia ngôn ngữ, tạo các cách diễn đạt khác nhau với từ đồng nghĩa.
        
        Nhiệm vụ: Tạo 2-3 cách diễn đạt khác bằng cách thay thế từ/cụm từ bằng từ đồng nghĩa.
        
        Yêu cầu:
        - Giữ nguyên ý nghĩa
        - Sử dụng từ đồng nghĩa tự nhiên trong tiếng Việt
        - Phù hợp với ngữ cảnh CLB Lập trình
        - Trả về dưới dạng danh sách Python
        
        Ví dụ:
        Input: "CLB có những thành viên nào?"
        Output: ["Câu lạc bộ có những người tham gia nào?", "CLB bao gồm những học viên nào?"]
        """
        
        user_prompt = f"Tạo 2-3 cách diễn đạt khác bằng từ đồng nghĩa:\n\nCâu hỏi: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._get_llm_response(messages)
        
        try:
            import ast
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if list_match:
                synonyms = ast.literal_eval(list_match.group())
                expanded_queries.extend(synonyms)
        except:
            pass
        
        return list(set(expanded_queries))  # Remove duplicates
    
    def context_aware_expansion(self, query: str) -> List[str]:
        """
        Technique 4: Context-Aware Expansion
        Mở rộng câu hỏi dựa trên ngữ cảnh cụ thể của CLB ProPTIT
        """
        system_prompt = """Bạn là chuyên gia về CLB Lập trình ProPTIT, hiểu rõ về hoạt động, thành viên, và đặc trưng của CLB.
        
        Nhiệm vụ: Mở rộng câu hỏi bằng cách thêm ngữ cảnh cụ thể về CLB ProPTIT.
        
        Kiến thức ngữ cảnh chi tiết về CLB ProPTIT:
        - ProPTIT được thành lập ngày 9/10/2011 bởi anh Chế Đình Sơn
        - Phương châm: "Chia sẻ để cùng nhau phát triển"
        - Có 6 team dự án: Team AI, Team Mobile, Team Data, Team Game, Team Web, Team Backend
        - Quy trình tuyển thành viên: 3 vòng (CV, Phỏng vấn, Training)
        - Chỉ tuyển sinh viên năm nhất, khoảng 25 người/năm, không biết lập trình cũng có thể tham gia
        - Lợi ích khi tham gia: kỹ năng lập trình, kỹ năng mềm, 0.1 điểm xét học bổng
        - Lộ trình học: C (training) → C++ → Cấu trúc dữ liệu & Giải thuật → OOP → Java
        - Sự kiện nổi bật: BigGame, SPOJ Tournament, PROCWAR, Code Battle, Game C++, ProGApp
        - Thành viên đạt nhiều giải thưởng: ICPC, AI competitions, Olympic Tin học
        - Thuộc cộng đồng S2B cùng với CLB Multimedia và CLB Nhà sáng tạo game
        - Tiêu chuẩn: dựa trên học tập, hoạt động và cách tương tác giữa các thành viên với nhau.
        - ProPTIT và IT PTIT đều là 2 CLB học thuật. Tuy có những hướng đi riêng nhưng cùng chung một mục đích hỗ trợ sinh viên trên con đường học tập. Mỗi hướng đi mà CLB chọn đều tạo nên những màu sắc đặc trưng riêng.

        Yêu cầu:
        - Thêm ngữ cảnh cụ thể từ thông tin trên
        - Tạo các biến thể tập trung vào khía cạnh khác nhau
        - Sử dụng thuật ngữ chính xác của CLB
        - Trả về dưới dạng danh sách Python
        
        Ví dụ:
        Input: "Làm thế nào để tham gia CLB?"
        Output: ["Quy trình 3 vòng tuyển thành viên ProPTIT như thế nào?", "Sinh viên năm nhất làm sao để vào CLB Lập trình PTIT?", "Điều kiện tham gia training ProPTIT?", "Cách đăng ký vòng CV cho CLB ProPTIT?"]
        """
        
        user_prompt = f"Mở rộng câu hỏi với ngữ cảnh CLB ProPTIT:\n\nCâu hỏi: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._get_llm_response(messages)
        
        try:
            import ast
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if list_match:
                context_queries = ast.literal_eval(list_match.group())
                return [query] + context_queries
        except:
            pass
        
        return [query]
    
    def multi_perspective_expansion(self, query: str) -> List[str]:
        """
        Technique 5: Multi-Perspective Query
        Tạo các góc nhìn khác nhau cho cùng một câu hỏi
        """
        system_prompt = """Bạn là chuyên gia phân tích đa góc độ, tạo các cách tiếp cận khác nhau cho cùng một câu hỏi.
        
        Nhiệm vụ: Tạo các góc nhìn khác nhau cho câu hỏi về CLB ProPTIT.
        
        Các góc nhìn có thể áp dụng:
        - Góc nhìn của sinh viên mới
        - Góc nhìn của thành viên hiện tại
        - Góc nhìn về lợi ích/giá trị
        - Góc nhìn về quá trình/thủ tục
        - Góc nhìn về yêu cầu/điều kiện
        - Góc nhìn về kết quả/thành tựu
        
        Yêu cầu:
        - Mỗi góc nhìn tạo ra câu hỏi khác biệt
        - Phù hợp với ngữ cảnh CLB Lập trình
        - Trả về dưới dạng danh sách Python
        
        Ví dụ:
        Input: "CLB ProPTIT có gì hay?"
        Output: ["CLB ProPTIT mang lại lợi ích gì cho sinh viên?", "Những hoạt động nổi bật của CLB ProPTIT?", "Tại sao nên tham gia CLB ProPTIT?", "CLB ProPTIT khác biệt như thế nào so với các CLB khác?"]
        """
        
        user_prompt = f"Tạo các góc nhìn khác nhau cho câu hỏi:\n\nCâu hỏi: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._get_llm_response(messages)
        
        try:
            import ast
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if list_match:
                perspective_queries = ast.literal_eval(list_match.group())
                return [query] + perspective_queries
        except:
            pass
        
        return [query]
    
    def document_structure_expansion(self, query: str) -> List[str]:
        """
        Technique 6: Document Structure-Aware Expansion
        Mở rộng dựa trên cấu trúc và nội dung cụ thể của tài liệu CLB ProPTIT
        """
        system_prompt = """Bạn là chuyên gia phân tích cấu trúc tài liệu CLB ProPTIT và hiểu rõ mapping giữa câu hỏi và document.
        
        Dựa trên cấu trúc tài liệu CLB ProPTIT, tạo các câu hỏi mở rộng tập trung vào từ khóa chính xác:
        
        Cấu trúc tài liệu CLB ProPTIT:
        - Document 1: Giới thiệu CLB (thành lập, phương châm, slogan)
        - Document 2-4: Hoạt động, giải thưởng, team dự án  
        - Document 5-56: Quy trình tuyển thành viên, lợi ích, điều kiện
        - Document 57-67: Lộ trình học tập (training, C++, CTDL&GT, OOP, Java)
        - Document 68-87: Sự kiện (BigGame, SPOJ, PROCWAR, Code Battle, etc.)
        - Document 88-93: Phòng truyền thống, thành viên tiêu biểu
        - Document 94-99: Quyền lợi, nghĩa vụ, tác phong, văn hóa, khen thưởng
        
        Nhiệm vụ: Tạo các câu hỏi mở rộng sử dụng từ khóa chính xác từ tài liệu.
        
        Yêu cầu:
        - Sử dụng từ khóa chính xác từ tài liệu gốc
        - Tạo các biến thể dựa trên cấu trúc document
        - Trả về dưới dạng danh sách Python
        
        Ví dụ:
        Input: "Khi tham gia CLB, tác phong và văn hóa ứng xử được quy định thế nào?"
        Output: ["Quy định về tác phong thành viên CLB", "Văn hóa ứng xử trong CLB ProPTIT", "Nội quy về ăn mặc và giao tiếp", "Quy tắc sử dụng trụ sở CLB", "Nghĩa vụ và quyền lợi thành viên"]
        """
        
        user_prompt = f"Tạo câu hỏi mở rộng dựa trên cấu trúc tài liệu:\n\nCâu hỏi: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._get_llm_response(messages)
        
        try:
            import ast
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if list_match:
                structure_queries = ast.literal_eval(list_match.group())
                return [query] + structure_queries
        except:
            pass
        
        return [query]
    
    def expand_query(self, query: str, techniques: List[str] = None, max_expansions: int = 10) -> List[str]:
        """
        Hàm chính để mở rộng câu truy vấn sử dụng tất cả các kỹ thuật
        
        Args:
            query: Câu hỏi gốc
            techniques: Danh sách kỹ thuật muốn sử dụng. Nếu None, sử dụng tất cả
            max_expansions: Số lượng tối đa câu hỏi mở rộng
        
        Returns:
            List[str]: Danh sách câu hỏi đã mở rộng (bao gồm câu gốc)
        """
        if techniques is None:
            techniques = ["rewriting", "decomposition", "synonym", "context", "multi_perspective", "document_structure"]
        
        all_expanded = [query]  # Bắt đầu với câu hỏi gốc
        
        try:
            # Apply each technique
            if "rewriting" in techniques:
                rewrites = self.query_rewriting(query, num_variants=2)
                all_expanded.extend(rewrites[1:])  # Skip original query
            
            if "decomposition" in techniques:
                decomposed = self.query_decomposition(query)
                all_expanded.extend(decomposed[1:])  # Skip original query
            
            if "synonym" in techniques:
                synonyms = self.synonym_expansion(query)
                all_expanded.extend(synonyms[1:])  # Skip original query
            
            if "context" in techniques:
                context_aware = self.context_aware_expansion(query)
                all_expanded.extend(context_aware[1:])  # Skip original query
            
            if "multi_perspective" in techniques:
                perspectives = self.multi_perspective_expansion(query)
                all_expanded.extend(perspectives[1:])  # Skip original query
            
            if "document_structure" in techniques:
                structure_aware = self.document_structure_expansion(query)
                all_expanded.extend(structure_aware[1:])  # Skip original query
        
        except Exception as e:
            print(f"Error in query expansion: {e}")
        
        # Remove duplicates while preserving order
        unique_expanded = []
        seen = set()
        for q in all_expanded:
            if q.lower() not in seen:
                unique_expanded.append(q)
                seen.add(q.lower())
        
        # Limit to max_expansions
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


# Utility function để test query expansion
def test_query_expansion():
    """Function để test các kỹ thuật query expansion"""
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
        
        # Test individual techniques
        print("\n1. Query Rewriting:")
        rewrites = expander.query_rewriting(query)
        for i, rw in enumerate(rewrites):
            print(f"   {i+1}. {rw}")
        
        print("\n2. Query Decomposition:")
        decomposed = expander.query_decomposition(query)
        for i, dc in enumerate(decomposed):
            print(f"   {i+1}. {dc}")
        
        print("\n3. Synonym Expansion:")
        synonyms = expander.synonym_expansion(query)
        for i, syn in enumerate(synonyms):
            print(f"   {i+1}. {syn}")
        
        print("\n4. Context-Aware Expansion:")
        context = expander.context_aware_expansion(query)
        for i, ctx in enumerate(context):
            print(f"   {i+1}. {ctx}")
        
        print("\n5. Multi-Perspective Expansion:")
        perspectives = expander.multi_perspective_expansion(query)
        for i, persp in enumerate(perspectives):
            print(f"   {i+1}. {persp}")
        
        print("\n6. Document Structure Expansion:")
        structure = expander.document_structure_expansion(query)
        for i, struct in enumerate(structure):
            print(f"   {i+1}. {struct}")
        
        print("\n7. All Techniques Combined:")
        all_expanded = expander.expand_query(query, max_expansions=10)
        for i, exp in enumerate(all_expanded):
            print(f"   {i+1}. {exp}")


if __name__ == "__main__":
    test_query_expansion()
