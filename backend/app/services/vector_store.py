"""
Vector store service using pgvector.

TODO: Implement this service to:
1. Generate embeddings for text chunks
2. Store embeddings in PostgreSQL with pgvector
3. Perform similarity search
4. Link related images and tables
"""
from app.core.config import settings
from app.models.document import DocumentChunk
from sqlalchemy import select, bindparam, text
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import numpy as np
import openai

class VectorStore:
    """
    Vector store for document embeddings and similarity search.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.embeddings_model = None  # TODO: Initialize embedding model
        self._ensure_extension()
    
    def _ensure_extension(self):
        """
        Ensure pgvector extension is enabled.
        
        This is implemented as an example.
        """
        try:
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.db.commit()
        except Exception as e:
            print(f"pgvector extension already exists or error: {e}")
            self.db.rollback()
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        TODO: Implement embedding generation
        - Use OpenAI embeddings API or
        - Use HuggingFace sentence-transformers
        - Return numpy array of embeddings
        
        Example with OpenAI:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return np.array(response.data[0].embedding)
        
        Example with HuggingFace:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text)
        """
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)  

        response = client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text
        )

        embedding = response.data[0].embedding

        return embedding
    
    async def store_chunk(
        self, 
        content: str, 
        document_id: int,
        page_number: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """
        Store a text chunk with its embedding.
        
        TODO: Implement chunk storage
        1. Generate embedding for content
        2. Create DocumentChunk record
        3. Store in database with embedding
        4. Include metadata (related images, tables, etc.)
        
        Args:
            content: Text content
            document_id: Document ID
            page_number: Page number
            chunk_index: Index of chunk in document
            metadata: Additional metadata (related_images, related_tables, etc.)
            
        Returns:
            Created DocumentChunk
        """
    
    async def similarity_search(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        TODO: Implement similarity search
        1. Generate embedding for query
        2. Use pgvector's cosine similarity (<=> operator)
        3. Filter by document_id if provided
        4. Return top k results with scores
        5. Include related images and tables in results
        
        Example SQL query:
        SELECT 
            id,
            content,
            page_number,
            metadata,
            1 - (embedding <=> :query_embedding) as similarity
        FROM document_chunks
        WHERE document_id = :document_id  -- optional filter
        ORDER BY embedding <=> :query_embedding
        LIMIT :k
        
        Args:
            query: Search query text
            document_id: Optional document ID to filter
            k: Number of results to return
            
        Returns:
            [
                {
                    "content": "...",
                    "score": 0.95,
                    "page_number": 3,
                    "metadata": {...},
                    "related_images": [...],
                    "related_tables": [...]
                }
            ]
        """
         # 1 - Generate embedding
        embedding = await self.generate_embedding(query)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        embedding = [float(x) for x in embedding]

        # 2 - Convert to pgvector literal
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        # 3 - Build raw SQL
        sql = f"""
        SELECT
            id,
            content,
            page_number,
            extra_metadata,
            1 - (embedding <=> :query_embedding) AS score
        FROM document_chunks
        {"WHERE document_id = :document_id" if document_id else ""}
        ORDER BY embedding <=> :query_embedding
        LIMIT :k
        """

        params = {"query_embedding": embedding_str, "k": k}
        if document_id:
            params["document_id"] = document_id

        result = self.db.execute(text(sql), params)
        rows = result.all()

        # 4 - Build output
        output = []
        for row in rows:
            chunk_id, content, page_number, extra_metadata, score = row
            related_images = extra_metadata.get("related_images", []) if extra_metadata else []
            related_tables = extra_metadata.get("related_tables", []) if extra_metadata else []
            output.append({
                "content": content,
                "score": float(score),
                "page_number": page_number,
                "metadata": extra_metadata or {},
                "related_images": related_images,
                "related_tables": related_tables
            })

        return output
    
    async def get_related_content(
        self,
        chunk_ids: List[int]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get related images and tables for given chunks.
        
        TODO: Implement related content retrieval
        - Query DocumentImage and DocumentTable based on metadata
        - Return organized by type (images, tables)
        
        Returns:
            {
                "images": [...],
                "tables": [...]
            }
        """
        raise NotImplementedError("Related content retrieval not implemented yet")
