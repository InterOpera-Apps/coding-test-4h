"""
Chat engine service for multimodal RAG.

TODO: Implement this service to:
1. Process user messages
2. Search for relevant context using vector store
3. Find related images and tables
4. Generate responses using LLM
5. Support multi-turn conversations
"""
from app.core.config import settings
from app.models.conversation import Conversation, Message
from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import openai
import time


class ChatEngine:
    """
    Multimodal chat engine with RAG.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        self.llm = None  # TODO: Initialize LLM (OpenAI, Ollama, etc.)
    
    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate multimodal response.
        
        Implementation steps:
        1. Load conversation history (for multi-turn support)
        2. Search vector store for relevant context
        3. Find related images and tables
        4. Build prompt with context and history
        5. Generate response using LLM
        6. Format response with sources (text, images, tables)
        
        Args:
            conversation_id: Conversation ID
            message: User message
            document_id: Optional document ID to scope search
            
        Returns:
            {
                "answer": "...",
                "sources": [
                    {
                        "type": "text",
                        "content": "...",
                        "page": 3,
                        "score": 0.95
                    },
                    {
                        "type": "image",
                        "url": "/uploads/images/xxx.png",
                        "caption": "Figure 1: ...",
                        "page": 3
                    },
                    {
                        "type": "table",
                        "url": "/uploads/tables/yyy.png",
                        "caption": "Table 1: ...",
                        "page": 5,
                        "data": {...}  # structured table data
                    }
                ],
                "processing_time": 2.5
            }
        """
        # TODO: Implement message processing
        # 
        # Example LLM usage with OpenAI:
        # from openai import OpenAI
        # client = OpenAI(api_key=settings.OPENAI_API_KEY)
        # 
        # response = client.chat.completions.create(
        #     model=settings.OPENAI_MODEL,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ]
        # )
        # 
        # Example with LangChain:
        # from langchain_openai import ChatOpenAI
        # from langchain.prompts import ChatPromptTemplate
        # 
        # llm = ChatOpenAI(model=settings.OPENAI_MODEL)
        # prompt = ChatPromptTemplate.from_messages([...])
        # chain = prompt | llm
        # response = chain.invoke({...})
        start_time = time.time()

        # 1 - Load conversation history
        history = await self._load_conversation_history(conversation_id)

        # 2 - Search vector store for relevant context
        context = await self._search_context(message, document_id=document_id)

        # 3 - Find related images and tables
        media = await self._find_related_media(context)

        # 4 - Generate response using LLM
        answer = await self._generate_response(message, context, history, media)

        # 5- Format sources
        sources = self._format_sources(context, media)

        processing_time = time.time() - start_time

        return {
            "answer": answer,
            "sources": sources,
            "processing_time": processing_time
        }
    
    async def _load_conversation_history(
        self,
        conversation_id: int,
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Load recent conversation history.
        
        TODO: Implement conversation history loading
        - Load last N messages from conversation
        - Format for LLM context
        - Include both user and assistant messages
        
        Returns:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.desc()).limit(limit)
        
        result = self.db.execute(stmt)
        messages = result.scalars().all()

        return [{"role": m.role, "content": m.content} for m in reversed(messages)]
    
    async def _search_context(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context using vector store.
        
        TODO: Implement context search
        - Use vector store similarity search
        - Filter by document if specified
        - Return relevant chunks with metadata
        """
        return await self.vector_store.similarity_search(query=query, document_id=document_id, k=k)
    
    async def _find_related_media(
        self,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find related images and tables from context chunks.
        
        TODO: Implement related media finding
        - Extract image/table references from chunk metadata
        - Query database for actual image/table records
        - Return with URLs for frontend display
        
        Returns:
            {
                "images": [
                    {
                        "url": "/uploads/images/xxx.png",
                        "caption": "...",
                        "page": 3
                    }
                ],
                "tables": [
                    {
                        "url": "/uploads/tables/yyy.png",
                        "caption": "...",
                        "page": 5,
                        "data": {...}
                    }
                ]
            }
        """
        images, tables = [], []

        # Gather image ids and table ids
        image_ids = []
        table_ids = []
        for chunk in context_chunks:
            image_ids.extend(chunk.get("related_images", []))
            table_ids.extend(chunk.get("related_tables", []))

        if image_ids:
            stmt = select(DocumentImage).where(DocumentImage.id.in_(image_ids))
            result = self.db.execute(stmt)
            images = [
                {
                    "url": img.file_path,
                    "caption": img.caption,
                    "page": img.page_number
                } for img in result.scalars().all()
            ]

        if table_ids:
            stmt = select(DocumentTable).where(DocumentTable.id.in_(table_ids))
            result = self.db.execute(stmt)
            tables = [
                {
                    "url": tbl.image_path,
                    "caption": tbl.caption,
                    "page": tbl.page_number,
                    "data": tbl.data
                } for tbl in result.scalars().all()
            ]

        return {"images": images, "tables": tables}
    
    async def _generate_response(
        self,
        message: str,
        context: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Generate response using LLM.
        
        TODO: Implement LLM response generation
        - Build comprehensive prompt with:
          - System instructions
          - Conversation history
          - Retrieved context
          - Available images/tables
        - Call LLM API
        - Return generated answer
        
        Prompt engineering tips:
        - Instruct LLM to reference images/tables when relevant
        - Include context from previous messages
        - Ask LLM to cite sources
        - Format for good UX (bullet points, etc.)
        """
        # Build system prompt
        system_prompt = "You are a helpful assistant. Use the context and media to answer questions."

        # Build context prompt
        context_text = "\n".join(
            [f"[Page {c['page_number']}]: {c['content']}" for c in context]
        )

        # Add image/table references
        media_text = ""
        for img in media.get("images", []):
            media_text += f"[Image Page {img['page']}] {img['caption']}\n"
        for tbl in media.get("tables", []):
            media_text += f"[Table Page {tbl['page']}] {tbl['caption']}\n"

        # Add history
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history])

        # Combine into user prompt
        user_prompt = f"{context_text}\n{media_text}\nConversation History:\n{history_text}\nUser: {message}"

        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # temperature=0.2 # not support with current model gpt-5-mini
        )

        return response.choices[0].message.content
    
    def _format_sources(
        self,
        context: List[Dict[str, Any]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Format sources for response.
        
        This is implemented as an example.
        """
        sources = []
        
        # Add text sources
        for chunk in context[:3]:  # Top 3 text chunks
            sources.append({
                "type": "text",
                "content": chunk["content"],
                "page": chunk.get("page_number"),
                "score": chunk.get("score", 0.0)
            })
        
        # Add image sources
        for image in media.get("images", []):
            sources.append({
                "type": "image",
                "url": image["url"],
                "caption": image.get("caption"),
                "page": image.get("page")
            })
        
        # Add table sources
        for table in media.get("tables", []):
            sources.append({
                "type": "table",
                "url": table["url"],
                "caption": table.get("caption"),
                "page": table.get("page"),
                "data": table.get("data")
            })
        
        return sources
