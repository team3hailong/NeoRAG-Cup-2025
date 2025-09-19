from typing import Callable, Optional
from docx import Document
import numpy as np


def build_collection_from_docx(
    doc_path: str,
    embedding_model,
    vector_db,
    collection_name: str = "information",
    rebuild: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Build or update a vector collection from a DOCX file using BGE-M3 when available.

    - Stores ColBERT multi-vectors if present, with cached dense/sparse in metadata,
      so hybrid/fast retrievers don’t need to re-encode at query time.

    Returns the number of inserted documents.
    """
    if rebuild:
        try:
            vector_db.drop_collection(collection_name)
        except Exception:
            pass

    # If already populated and not rebuilding, skip
    try:
        existing = vector_db.count_documents(collection_name)
    except Exception:
        existing = 0

    if existing > 0 and not rebuild:
        return 0

    doc = Document(doc_path)
    paragraphs = [p for p in doc.paragraphs if p.text.strip()]
    total = len(paragraphs)
    inserted = 0

    for idx, para in enumerate(paragraphs, start=1):
        text = para.text
        emb = embedding_model.encode(text)

        # Prefer ColBERT if available
        if isinstance(emb, dict) and 'colbert_vecs' in emb and getattr(embedding_model, 'use_colbert', False):
            embedding_for_db = np.array(emb['colbert_vecs'])
            record = {
                "title": f"Document {inserted + 1}",
                "information": text,
                "embedding": embedding_for_db,
                "m3_dense": emb.get('dense_vecs'),
                "m3_sparse": emb.get('lexical_weights') or emb.get('sparse_weights', {}),
            }
        else:
            # Dense-only fallback
            if isinstance(emb, dict):
                dense = emb.get('dense_vecs') or next(iter(emb.values()))
            else:
                dense = emb
            record = {
                "title": f"Document {inserted + 1}",
                "information": text,
                "embedding": dense,
            }

        vector_db.insert_document(collection_name=collection_name, document=record)
        inserted += 1
        if progress_callback:
            progress_callback(idx, total)

    return inserted
