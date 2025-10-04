import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

try:
    import onnxruntime as ort
    from onnxruntime import SessionOptions
    from onnxruntime.quantization import QuantType, quantize_dynamic
except Exception:  # pragma: no cover - optional dependency guards
    ort = None
    SessionOptions = None
    quantize_dynamic = None
    QuantType = None

from FlagEmbedding import BGEM3FlagModel

load_dotenv()

# Các em có thể tự thêm embedding model mới hoặc dùng các model có sẵn
class Embeddings:
    def __init__(self, model_name: str, type: str):
        self.model_name = model_name
        self.type = type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fp16 = self.device == 'cuda'
        if self.device == 'cuda':
            print(f"[Embeddings] Using device: {self.device}")

        self.max_seq_length = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", 384))
        self.quantization_mode = os.getenv("EMBEDDING_QUANT_MODE", "int8").lower()
        self.enable_ort = os.getenv("EMBEDDING_USE_ORT", "1") not in {"0", "false", "False"}
        self.cpu_threads = int(os.getenv("EMBEDDING_CPU_THREADS", max(1, (os.cpu_count() or 1))))
        self.quant_backend: Optional[str] = None
        self.ort_session = None  # type: Optional[Any]
        self.tokenizer = None
        self._cache_dir = Path(os.getenv("EMBEDDING_CACHE_DIR", "./.cache/embeddings")).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Toggle BGE instruction prefixes for dense models
        self.use_bge_instruction = True

        if type == "sentence_transformers":
            try:
                self.client = SentenceTransformer(
                    model_name,
                    device=self.device,
                    trust_remote_code=True
                )
                self.use_colbert = False
                self.tokenizer = getattr(self.client, "tokenizer", None) or AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                self._prepare_tokenizer()
                if self.device == 'cpu':
                    self._setup_cpu_acceleration()
            except Exception as e:
                print(f"[Error] Failed to load fine-tuned model '{model_name}': {e}")
                print(f"[Warning] Falling back to base BGE-M3 model")
                # Fallback to base model
                self.client = BGEM3FlagModel("BAAI/bge-m3", use_fp16=self.fp16)
                self.use_colbert = True
                self.tokenizer = None
        else:
            self.client = None
            self.use_colbert = False

    def _maybe_prefix(self, text: str, is_query: bool) -> str:
        if not isinstance(text, str):
            text = str(text)
        if not self.use_bge_instruction:
            return text
        return ("query: " + text) if is_query else ("passage: " + text)

    def _prepare_tokenizer(self):
        if not self.tokenizer:
            return
        try:
            # Respect custom max length to avoid Hugging Face warnings & runtime errors
            self.tokenizer.model_max_length = self.max_seq_length
            if hasattr(self.tokenizer, "init_kwargs"):
                self.tokenizer.init_kwargs["model_max_length"] = self.max_seq_length
        except Exception:
            pass
        if hasattr(self.client, "max_seq_length"):
            self.client.max_seq_length = self.max_seq_length

    def _setup_cpu_acceleration(self):
        # Configure threading for PyTorch and BLAS backends
        try:
            torch.set_num_threads(self.cpu_threads)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", str(self.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.cpu_threads))

        if self.enable_ort and ort and quantize_dynamic:
            try:
                self._init_ort_session()
                if self.ort_session is not None:
                    self.quant_backend = "onnxruntime-int8"
                    print(f"[Embeddings] ONNX Runtime INT8 session ready (threads={self.cpu_threads})")
                    return
            except Exception as err:
                print(f"[Warning] Failed to initialise ONNX Runtime session: {err}")

        # Fallback to PyTorch dynamic quantization
        self._apply_torch_dynamic_quantization()

    def _onnx_export_paths(self) -> Dict[str, Path]:
        safe_model_id = self.model_name.replace('/', '__')
        base = self._cache_dir / safe_model_id
        onnx_path = base.with_suffix('.onnx')
        quant_suffix = 'int8' if self.quantization_mode not in {"int4"} else 'int4'
        quant_path = base.with_name(base.name + f"_{quant_suffix}").with_suffix('.onnx')
        return {"onnx": onnx_path, "quant": quant_path}

    def _init_ort_session(self):
        paths = self._onnx_export_paths()
        onnx_path = paths["onnx"]
        quant_path = paths["quant"]

        if not onnx_path.exists():
            print(f"[Embeddings] Exporting '{self.model_name}' to ONNX at {onnx_path}")
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.save_to_onnx(
                output_path=str(onnx_path),
                device='cpu',
                batch_size=1,
                seq_length=self.max_seq_length
            )

        if not quant_path.exists():
            mode = self.quantization_mode
            if mode == 'int4':
                print("[Warning] INT4 not natively supported by ONNX Runtime dynamic quantization. Falling back to INT8.")
                mode = 'int8'
            quant_type = QuantType.QInt8 if mode == 'int8' else QuantType.QInt8
            print(f"[Embeddings] Quantizing ONNX model to {mode.upper()} at {quant_path}")
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(quant_path),
                weight_type=quant_type,
                optimize_model=True
            )

        if SessionOptions is None:
            raise RuntimeError("ONNX Runtime is unavailable")

        session_options = SessionOptions()
        session_options.intra_op_num_threads = self.cpu_threads
        session_options.inter_op_num_threads = max(1, self.cpu_threads // 2)
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_session = ort.InferenceSession(
            str(quant_path),
            providers=["CPUExecutionProvider"],
            sess_options=session_options
        )

    def _apply_torch_dynamic_quantization(self):
        base_module = None
        try:
            if hasattr(self.client, "_first_module"):
                base_module = self.client._first_module()
            if base_module and hasattr(base_module, "auto_model"):
                print("[Embeddings] Applying PyTorch dynamic INT8 quantization on linear layers")
                quantized_model = torch.quantization.quantize_dynamic(
                    base_module.auto_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                base_module.auto_model = quantized_model
                self.quant_backend = "torch-dynamic-int8"
        except Exception as err:
            print(f"[Warning] Dynamic quantization failed: {err}")

    def _l2_normalize(self, vec):
        if isinstance(vec, np.ndarray):
            v = vec
        else:
            v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return (v / norm).astype(np.float32)

    def _trim_text(self, text: str) -> str:
        if not isinstance(text, str) or not self.tokenizer or self.max_seq_length <= 0:
            return text
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        input_ids = encoded.get("input_ids")
        if not input_ids:
            return text
        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if len(input_ids) <= self.max_seq_length:
            # Tokenizer may already trim; re-decoding ensures normalized spacing
            return self.tokenizer.decode(input_ids, skip_special_tokens=True)
        trimmed_ids = input_ids[: self.max_seq_length]
        return self.tokenizer.decode(trimmed_ids, skip_special_tokens=True)

    def _prepare_ort_inputs(self, text: Union[str, Iterable[str]]) -> Dict[str, Any]:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is required for ONNX Runtime inference")
        return self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="np"
        )

    def _encode_with_onnxruntime(self, text: Union[str, Iterable[str]]):
        if not self.ort_session:
            raise RuntimeError("ONNX Runtime session not initialised")
        ort_inputs = self._prepare_ort_inputs(text)
        outputs = self.ort_session.run(None, ort_inputs)
        if not outputs:
            raise RuntimeError("ONNX Runtime returned no outputs")
        embedding = outputs[0]
        # SentenceTransformer ONNX export returns [batch, dim]
        if isinstance(text, str):
            embedding = embedding[0]
        return embedding

    def encode(self, doc, is_query: Optional[bool] = None):
        if isinstance(doc, (list, tuple)):
            return [self.encode(d, is_query=is_query) for d in doc]

        if self.type in ["openai", "ollama"]:
            return self.client.embeddings.create(
                input=doc,
                model=self.model_name
            ).data[0].embedding
        elif self.type == "sentence_transformers":
            if self.model_name == "BAAI/bge-m3" and self.use_colbert:
                output = self.client.encode(
                    doc,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True
                )
                return output
            else:
                # sentence-transformers encode returns ndarray/list, ensure list for JSON safety
                # Apply BGE instruction prefixes for better alignment between query/passages
                # Auto-detect query when not specified: short text or ends with '?'
                if is_query is None:
                    if isinstance(doc, str):
                        is_query = doc.strip().endswith('?') or len(doc) < 100
                    else:
                        is_query = False
                text = self._maybe_prefix(doc, is_query=is_query)
                text = self._trim_text(text)

                if self.ort_session is not None:
                    embedding = self._encode_with_onnxruntime(text)
                else:
                    embedding = self.client.encode(
                        text,
                        normalize_embeddings=True,
                        batch_size=1,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )

                if isinstance(embedding, dict):
                    embedding = embedding.get('dense_vecs', embedding)

                if isinstance(embedding, np.ndarray):
                    embedding = embedding.astype(np.float32)
                else:
                    embedding = np.array(embedding, dtype=np.float32)

                embedding = self._l2_normalize(embedding).tolist()
                return embedding
        elif self.type == "gemini":
            return self.client.models.embed_content(
                model=self.model_name,
                contents=doc
            ).embeddings[0].values
    
    def compute_colbert_similarity(self, query_vecs, doc_vecs):
        if self.model_name == "BAAI/bge-m3" and self.use_colbert:
            return self.client.colbert_score(query_vecs, doc_vecs)
        else:
            if isinstance(query_vecs, np.ndarray):
                query_vecs = torch.from_numpy(query_vecs)
            if isinstance(doc_vecs, np.ndarray):
                doc_vecs = torch.from_numpy(doc_vecs)
            
            query_vecs = torch.nn.functional.normalize(query_vecs, p=2, dim=-1)
            doc_vecs = torch.nn.functional.normalize(doc_vecs, p=2, dim=-1)
            
            similarity_matrix = torch.matmul(query_vecs, doc_vecs.transpose(0, 1))
            max_sim_per_query_token = torch.max(similarity_matrix, dim=1)[0]
            final_score = torch.mean(max_sim_per_query_token).item()
            
            return final_score