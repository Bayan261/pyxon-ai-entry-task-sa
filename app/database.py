
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from loguru import logger


Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    file_name     = Column(String(255), nullable=False)
    file_type     = Column(String(10))
    file_size_kb  = Column(Float)
    language      = Column(String(20))
    word_count    = Column(Integer)
    char_count    = Column(Integer)
    num_pages     = Column(Integer, nullable=True)
    strategy_used = Column(String(20))   # fixed أو dynamic
    chunk_count   = Column(Integer)
    created_at    = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("ChunkRecord", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, name='{self.file_name}', lang='{self.language}')>"


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer)       # رقم الـ chunk في الوثيقة
    text        = Column(Text)          # النص نفسه
    strategy    = Column(String(30))    # fixed / dynamic / dynamic-fallback
    word_count  = Column(Integer)
    char_count  = Column(Integer)
    start_char  = Column(Integer)       # موضع البداية في النص الأصلي
    end_char    = Column(Integer)
    section     = Column(String(500), nullable=True)  # اسم القسم/الفصل
    chroma_id   = Column(String(100), nullable=True)  # المعرّف في ChromaDB

    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"


class DatabaseManager:

    def __init__(self, db_path: str = "pyxon_parser.db"):
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,  # True لعرض استعلامات SQL في الكونسول
        )
        Base.metadata.create_all(self.engine)  # إنشاء الجداول
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"قاعدة البيانات جاهزة: {db_path}")

    def save_document(self, metadata: dict, chunks: list, strategy_used: str) -> int:
        session = self.Session()
        try:
            doc = Document(
                file_name     = metadata.get("file_name", "unknown"),
                file_type     = metadata.get("file_type", ""),
                file_size_kb  = metadata.get("file_size_kb", 0),
                language      = metadata.get("language", "unknown"),
                word_count    = metadata.get("word_count", 0),
                char_count    = metadata.get("char_count", 0),
                num_pages     = metadata.get("num_pages"),
                strategy_used = strategy_used,
                chunk_count   = len(chunks),
            )
            session.add(doc)
            session.flush()  # للحصول على doc.id قبل commit

            for chunk in chunks:
                section = chunk.metadata.get("section", "")
                chunk_record = ChunkRecord(
                    document_id = doc.id,
                    chunk_index = chunk.index,
                    text        = chunk.text,
                    strategy    = chunk.strategy,
                    word_count  = chunk.word_count(),
                    char_count  = len(chunk.text),
                    start_char  = chunk.start_char,
                    end_char    = chunk.end_char,
                    section     = section,
                )
                session.add(chunk_record)

            session.commit()
            logger.success(f"تم الحفظ في SQL | Document ID: {doc.id} | Chunks: {len(chunks)}")
            return doc.id

        except Exception as e:
            session.rollback()
            logger.error(f"خطأ في الحفظ: {e}")
            raise
        finally:
            session.close()

    def update_chunk_chroma_id(self, document_id: int, chunk_index: int, chroma_id: str):
        session = self.Session()
        try:
            chunk = session.query(ChunkRecord).filter_by(
                document_id=document_id,
                chunk_index=chunk_index
            ).first()
            if chunk:
                chunk.chroma_id = chroma_id
                session.commit()
        finally:
            session.close()

    def get_all_documents(self) -> list:
        session = self.Session()
        try:
            return session.query(Document).all()
        finally:
            session.close()

    def get_document_chunks(self, document_id: int) -> list:
        session = self.Session()
        try:
            return session.query(ChunkRecord).filter_by(document_id=document_id).all()
        finally:
            session.close()

    def get_stats(self) -> dict:
        session = self.Session()
        try:
            total_docs   = session.query(Document).count()
            total_chunks = session.query(ChunkRecord).count()
            arabic_docs  = session.query(Document).filter_by(language="arabic").count()
            return {
                "total_documents": total_docs,
                "total_chunks":    total_chunks,
                "arabic_documents": arabic_docs,
            }
        finally:
            session.close()
