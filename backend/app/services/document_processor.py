"""
Document processing service using Docling

TODO: Implement this service to:
1. Parse PDF documents using Docling
2. Extract text, images, and tables
3. Store extracted content in database
4. Generate embeddings for text chunks
"""
from app.db.session import get_db
from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore
from docling_core.types.doc import TableItem, PictureItem
from docling.document_converter import DocumentConverter
from fastapi import FastAPI, HTTPException, status
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import fitz
import json
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import pandas as pd
import uuid

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Process PDF documents and extract multimodal content.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
    
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Process a PDF document using Docling.
        
        Implementation steps:
        1. Update document status to 'processing'
        2. Use Docling to parse the PDF
        3. Extract and save text chunks
        4. Extract and save images
        5. Extract and save tables
        6. Generate embeddings for text chunks
        7. Update document status to 'completed'
        8. Handle errors appropriately
        
        Args:
            file_path: Path to the uploaded PDF file
            document_id: Database ID of the document
            
        Returns:
            {
                "status": "success" or "error",
                "text_chunks": <count>,
                "images": <count>,
                "tables": <count>,
                "processing_time": <seconds>
            }
        """
        # TODO: Implement document processing
        # 
        # Example Docling usage:
        # from docling.document_converter import DocumentConverter
        # 
        # converter = DocumentConverter()
        # result = converter.convert(file_path)
        # 
        # # Extract text
        # for page in result.pages:
        #     text_content = page.text
        #     # Chunk and store...
        # 
        # # Extract images
        # for image in result.images:
        #     # Save image file and create DocumentImage record
        # 
        # # Extract tables
        # for table in result.tables:
        #     # Render as image and create DocumentTable record
        # 1 - Update document status to "processing"
        document = self._update_document_status(document_id, "processing")

        # 2 - Use Docling to parse the PDF
        result_doc = self._parse_pdf(file_path, document)
        
        # 3 - Extract and save text chunks
        # 6 - Generate embeddings for text chunks
        text_chunks = self._chunk_text(result_doc, document_id)
        total_text_chunks = await self._save_text_chunks(text_chunks)

        # 4 - Extract and save images
        images = self._extract_images(result_doc, file_path, document_id)
        total_images = await self._save_image(images)
        
        # 5 - Extract and save tables
        tables = self._extract_tables(result_doc)
        tables_rows = self._store_tables_as_images(tables, document_id)
        total_tables = await self._save_table(tables_rows)

        document.text_chunks_count = len(total_text_chunks)
        document.images_count = len(total_images)
        document.tables_count = len(total_tables)
        self.db.commit()
        self.db.refresh(document)

        # 7 - Update document status to 'completed'
        document = self._update_document_status(document_id, "completed")

        logger.info("Document successfully finalised")

        return document
    
    def _chunk_text(self, doc, document_id: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage.
        - Merge headers with body text
        - Each header + its body becomes ONE chunk
        """
        texts: List[Dict[str, Any]] = []
        chunk_index = 0
        buffer = []
        buffer_page = None
        header_label = None

        items = list(doc.iterate_items())
        n = len(items)

        for i, (item, level) in enumerate(items):
            if not hasattr(item, "text") or not item.text.strip():
                continue

            content = item.text.strip()
            item_page = item.prov[0].page_no if item.prov else None
            if not buffer_page:
                buffer_page = item_page

            is_header = False
            if item.label and item.label.name.lower() in ["section_header", "header", "title"]:
                if i + 1 < n:
                    next_item, _ = items[i + 1]
                    if hasattr(next_item, "text") and next_item.text.strip():
                        is_header = True

            # Merge headers with body text
            # Skip non-text items unless they are meaningful
            if is_header:
                # Flush previous section
                if buffer:
                    texts.append({
                        "document_id": document_id,
                        "content": "\n".join(buffer),
                        "embedding": None,
                        "page_number": buffer_page,
                        "chunk_index": chunk_index,
                        "extra_metadata": {"header": header_label}
                    })
                    chunk_index += 1
                    buffer = []

                header_label = content
                buffer.append(content)
                buffer_page = item_page
            else:
                buffer.append(content)
                buffer_page = item_page

        # Flush remaining buffer
        if buffer:
            texts.append({
                "document_id": document_id,
                "content": "\n".join(buffer),
                "embedding": None,
                "page_number": buffer_page,
                "chunk_index": chunk_index,
                "extra_metadata": {"header": header_label}
            })

        return texts

    async def _save_text_chunks(self, chunks: List[Dict[str, Any]]) -> DocumentChunk:
        """
        Save text chunks to database with embeddings.
        
        TODO: Implement chunk storage
        - Generate embeddings
        - Store in database
        - Link related images/tables in metadata
        """
        rows = []

        # Generate embeddings
        for chunk in chunks:
            embedding = await self.vector_store.generate_embedding(
                chunk["content"],
            )
            chunk["embedding"] = embedding
    
        # Store in database
        # Link related images/tables in metadata
        for t in chunks:
            chunk = DocumentChunk(
                document_id=t["document_id"],
                content=t["content"],
                embedding=t["embedding"],
                page_number=t.get("page_number"),
                chunk_index=t.get("chunk_index"),
                extra_metadata=t.get("extra_metadata")
            )

            rows.append(chunk)

        self.db.add_all(rows)
        self.db.commit()

        for r in rows:
            self.db.refresh(r)

        return rows
    
    async def _save_image(
        self, 
        images: List[Dict[str, Any]],
    ) -> DocumentImage:
        """
        Save an extracted image.
        
        TODO: Implement image saving
        - Save image file to disk
        - Create DocumentImage record
        - Extract caption if available
        """
        rows = []

        # Save image file to disk
        # Create DocumentImage record
        for t in images:
            image = DocumentImage(
                document_id=t["document_id"],
                file_path=t["file_path"],
                page_number=t.get("page_number"),
                caption=t["caption"],
                width=t["width"],
                height=t["height"],
                extra_metadata=t.get("extra_metadata")
            )

            rows.append(image)
            
        self.db.add_all(rows)
        self.db.commit()

        for r in rows:
            self.db.refresh(r)

        return rows
    
    async def _save_table(
        self,
        tables: List[Dict[str, Any]],
    ) -> DocumentTable:
        """
        Save an extracted table.
        
        TODO: Implement table saving
        - Render table as image
        - Store structured data as JSON
        - Create DocumentTable record
        - Extract caption if available
        """
        # Render table as image will be done in separate method
        rows = []

        # Create DocumentTable record
        for t in tables:
            tables = DocumentTable(
                document_id=t["document_id"],
                image_path=t["image_path"],
                data=t.get("data"),
                page_number=t["page_number"],
                caption=t["caption"],
                rows=t["rows"],
                columns=t["columns"],
                extra_metadata=t.get("extra_metadata")
            )

            rows.append(tables)
            
        self.db.add_all(rows)
        self.db.commit()

        for r in rows:
            self.db.refresh(r)

        return rows
    
    def _update_document_status(
        self, 
        document_id: int, 
        document_status: str, 
        error_message: str = None
    ) -> Document:
        """
        Update document processing status.
        
        This is implemented as an example.
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document can't be found"
            )
        
        if document.processing_status in ("completed", "failed"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document already finalised"
            )

        document.processing_status = document_status

        if error_message:
            document.error_message = error_message

        self.db.commit()
        self.db.refresh(document)

        return document

    def _parse_pdf(self, file_path: str, document: Document) -> Any:
        converter = DocumentConverter()
        result = converter.convert(file_path)

        # Update the document's total_page
        document.total_pages = len(result.pages)
        self.db.commit()
        self.db.refresh(document)

        # Get the structured document
        doc = result.document

        return doc

    """
    Image Extraction Methods
    """
    def _extract_images(self, doc, file_path, document_id) -> List[Dict[str, Any]]:
        pictures = []

        for item, level in doc.iterate_items():
            if isinstance(item, PictureItem):
                # Page number
                page_no = item.prov[0].page_no if item.prov else None

                # Bounding box
                bbox = item.prov[0].bbox if item.prov else None
                width = bbox.r - bbox.l if bbox else None
                height = bbox.t - bbox.b if bbox else None

                # Captions (resolve RefItem to actual text)
                captions = []
                for ref in item.captions:
                    # ref.cref is something like '#/texts/404'
                    # extract the text index
                    idx = int(ref.cref.split('/')[-1])
                    if 0 <= idx < len(doc.texts):
                        captions.append(doc.texts[idx].text)
                caption_text = " ".join(captions) if captions else None

                image_path = self._extract_picture_pixels(
                    pdf_path=file_path,
                    bbox=item.prov[0].bbox,
                    page_number=page_no,
                    output_dir="/app/uploads/images",
                )

                pictures.append({
                    "document_id": document_id,
                    "file_path": image_path,  # no image saved
                    "page_number": page_no,
                    "caption": caption_text,
                    "width": width,
                    "height": height,
                    "extra_metadata": {
                        "level": level,
                        "bbox": self._bbox_to_json(item.prov[0].bbox) if item.prov else None,
                        "prov": self._provenance_to_json(item.prov),
                        "annotations": item.annotations,
                        "children": [c.cref for c in item.children],
                        "parent": item.parent.cref if item.parent else None,
                        "self_ref": item.self_ref,
                    }
                })

        return pictures
    
    def _extract_picture_pixels(self, pdf_path: str, bbox, page_number: int, output_dir: str, scale: float = 2.0) -> str:
        """
        Extract image pixels from a PDF page using Docling bbox coordinates.
        
        Args:
            pdf_path: path to PDF
            bbox: Docling BoundingBox object (l, t, r, b)
            page_number: 0-based page index
            output_dir: directory to save image
            scale: scaling factor for higher-res image
        Returns:
            path to saved image
        """
        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)

        # Convert 1-based Docling page_no to 0-based PyMuPDF page index
        page_index = max(0, min(page_number - 1, len(doc) - 1))
        page = doc[page_index]

        # Page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        # Docling bbox (l, t, r, b) and convert to PyMuPDF coordinates
        x0 = max(0, bbox.l)
        x1 = min(page_width, bbox.r)
        # Flip Y-axis (Docling origin is BOTTOMLEFT)
        y0 = page_height - bbox.t
        y1 = page_height - bbox.b

        # Clamp
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        rect = fitz.Rect(x0, y0, x1, y1)

        # Apply scaling
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)

        # Save
        filename = f"picture_page{page_number + 1}_{int(x0)}_{int(y0)}.png"
        path = os.path.join(output_dir, filename)
        pix.save(path)
        doc.close()

        return path
    
    def _bbox_to_rect(self, bbox, page_height, scale=1.0) -> Any:
        """
        Convert Docling BoundingBox to PyMuPDF Rect with optional scaling.
        Returns fitz.Rect
        """
        x0 = bbox.l * scale
        x1 = bbox.r * scale
        y0 = page_height - bbox.t  # invert top
        y1 = page_height - bbox.b  # invert bottom

        # Ensure x0<x1 and y0<y1
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        return fitz.Rect(x0, y0, x1, y1)

    """
    Table Extraction Methods
    """
    def _extract_tables(self, doc) -> List[Dict[str, Any]]:
        tables = []

        for item, level in doc.iterate_items():
            if isinstance(item, TableItem):
                # Convert to DataFrame
                df: pd.DataFrame = item.export_to_dataframe(doc=doc)

                # Basic metadata
                page_no = item.prov[0].page_no if item.prov else None
                caption_text = None
                if item.captions:
                    captions = []
                    for ref in item.captions:
                        idx = int(ref.cref.split('/')[-1])
                        if 0 <= idx < len(doc.texts):
                            captions.append(doc.texts[idx].text)
                    caption_text = " ".join(captions)

                tables.append({
                    "table_item": item,
                    "df": df,
                    "page_number": page_no,
                    "caption": caption_text,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                })

        return tables

    def _render_table_pillow(self, df: pd.DataFrame, output_path: str, cell_padding=10, font_size=14):
        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        # Column widths
        col_widths = []
        ascent, descent = font.getmetrics()
        row_height = ascent + descent + 2*cell_padding

        for col in df.columns:
            # Width of header
            max_width = font.getlength(str(col))
            # Width of each cell in the column
            for val in df[col]:
                max_width = max(max_width, font.getlength(str(val)))
            col_widths.append(int(max_width) + 2*cell_padding)

        # Table size
        table_width = sum(col_widths)
        table_height = row_height * (len(df) + 1)  # +1 for header

        # Create image
        img = Image.new("RGB", (table_width, table_height), "white")
        draw = ImageDraw.Draw(img)

        # Draw header
        x = 0
        for i, col in enumerate(df.columns):
            draw.rectangle([x, 0, x+col_widths[i], row_height], outline="black", fill="lightgrey")
            draw.text((x+cell_padding, cell_padding), str(col), fill="black", font=font)
            x += col_widths[i]

        # Draw rows
        y = row_height
        for row_idx in range(len(df)):
            x = 0
            for col_idx, col in enumerate(df.columns):
                draw.rectangle([x, y, x+col_widths[col_idx], y+row_height], outline="black")
                draw.text((x+cell_padding, y+cell_padding), str(df.iloc[row_idx, col_idx]), fill="black", font=font)
                x += col_widths[col_idx]
            y += row_height

        # Save image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)

    def _store_tables_as_images(self, tables, document_id) -> List[Dict[str, Any]]:
        output_dir = "/app/uploads/tables"
        os.makedirs(output_dir, exist_ok=True)

        records = []
        
        for t in tables:
            df = t["df"]
            item = t["table_item"]
            
            # Render image
            filename = f"table_{uuid.uuid4().hex}.png"
            image_path = os.path.join(output_dir, filename)
            self._render_table_pillow(df, image_path)

            # Save to DB
            records.append({
                "document_id": document_id,
                "image_path": image_path,
                "data": df.to_dict(orient="records"),
                "page_number": t["page_number"],
                "caption": t["caption"],
                "rows": t["rows"],
                "columns": t["columns"],
                "extra_metadata": {
                    "self_ref": item.self_ref,
                    "children": [c.cref for c in item.children],
                    "parent": item.parent.cref if item.parent else None,
                    "prov": self._provenance_to_json(item.prov),
                }
            })

        return records

    """
    Transforming Docling Result Into JSON
    """
    def _bbox_to_json(self, bbox) -> Dict[str, Any]:
        if bbox is None:
            return None

        return {
            "l": bbox.l,
            "t": bbox.t,
            "r": bbox.r,
            "b": bbox.b,
            "coord_origin": bbox.coord_origin.value  # enum to str
        }

    def _provenance_to_json(self, prov) -> List[Dict[str, Any]]:
        if not prov:
            return None

        return [
            {
                "page_no": p.page_no,
                "bbox": {
                    "l": p.bbox.l,
                    "t": p.bbox.t,
                    "r": p.bbox.r,
                    "b": p.bbox.b,
                    "coord_origin": p.bbox.coord_origin.value,
                } if p.bbox else None,
                "charspan": list(p.charspan) if p.charspan else None,
            }
            for p in prov
        ]
