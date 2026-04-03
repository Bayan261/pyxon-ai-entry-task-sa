
import re
from dataclasses import dataclass, field
from typing import List
from loguru import logger


@dataclass
class Chunk:
    text: str
    index: int
    strategy: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.text)

    def word_count(self):
        return len(self.text.split())


class IntelligentChunker:

    def __init__(
        self,
        fixed_chunk_size: int = 512,    # عدد الكلمات في كل chunk (للـ fixed)
        fixed_overlap: int = 50,         # عدد الكلمات المتداخلة بين chunks
        min_chunk_size: int = 100,       # أقل حجم مقبول لـ chunk
        max_chunk_size: int = 1000,      # أكبر حجم مقبول
    ):
        self.fixed_chunk_size = fixed_chunk_size
        self.fixed_overlap = fixed_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size


    def chunk(self, text: str, metadata: dict) -> List[Chunk]:
        if not text or not text.strip():
            logger.warning("النص فارغ، لا يمكن التقسيم")
            return []

        strategy = self._decide_strategy(text, metadata)
        logger.info(f"استراتيجية التقسيم المختارة: {strategy}")

        if strategy == "fixed":
            chunks = self._fixed_chunking(text)
        else:
            chunks = self._dynamic_chunking(text, metadata)

        chunks = [c for c in chunks if c.word_count() >= 20]

        logger.success(f"تم التقسيم: {len(chunks)} chunk | الاستراتيجية: {strategy}")
        return chunks


    def _decide_strategy(self, text: str, metadata: dict) -> str:
        score_dynamic = 0  # نقاط للـ dynamic chunking
        score_fixed = 0    # نقاط للـ fixed chunking

        heading_patterns = [
            r"^#{1,6}\s+\S+",           # Markdown headings
            r"^[A-ZА-Я\u0600-\u06FF].{0,50}\n[=\-]{3,}",  # underline headings
            r"^\[صفحة \d+\]",           # علامات الصفحات من PDF parser
            r"^الفصل\s+\w+",            # فصل + رقم (عربي)
            r"^Chapter\s+\d+",          # Chapter + number
            r"^(\d+\.){1,3}\s+\w+",     # 1.2.3 Structure
        ]
        heading_count = sum(
            len(re.findall(p, text, re.MULTILINE))
            for p in heading_patterns
        )

        if heading_count >= 3:
            score_dynamic += 3
        elif heading_count >= 1:
            score_dynamic += 1

        headings = metadata.get("headings", [])
        if len(headings) >= 3:
            score_dynamic += 3
        elif len(headings) >= 1:
            score_dynamic += 1

        lines = [l for l in text.split("\n") if l.strip()]
        avg_line_length = sum(len(l) for l in lines) / len(lines) if lines else 0
        if avg_line_length > 200:
            score_fixed += 2  # فقرات طويلة = وثيقة متصلة = fixed

        table_like = len(re.findall(r"\t|\|", text))
        list_like = len(re.findall(r"^[\-\*\•]\s+", text, re.MULTILINE))
        if table_like > 10 or list_like > 10:
            score_fixed += 2

        word_count = len(text.split())
        if word_count > 5000:
            score_dynamic += 1  # نصوص طويلة تستفيد من التقسيم الذكي

        logger.debug(f"نقاط dynamic={score_dynamic}, fixed={score_fixed}")
        return "dynamic" if score_dynamic > score_fixed else "fixed"


    def _fixed_chunking(self, text: str) -> List[Chunk]:
        words = text.split()
        chunks = []
        index = 0
        pos = 0  # موضع الكلمة الحالية

        while pos < len(words):
            end_pos = min(pos + self.fixed_chunk_size, len(words))
            chunk_words = words[pos:end_pos]
            chunk_text = " ".join(chunk_words)

            start_char = text.find(chunk_words[0]) if chunk_words else 0

            chunks.append(Chunk(
                text=chunk_text,
                index=index,
                strategy="fixed",
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata={"word_count": len(chunk_words)},
            ))

            index += 1
            pos += self.fixed_chunk_size - self.fixed_overlap

        return chunks


    def _dynamic_chunking(self, text: str, metadata: dict) -> List[Chunk]:
        sections = self._split_by_headings(text)

        chunks = []
        index = 0
        current_pos = 0

        for section_title, section_text in sections:
            paragraphs = self._split_into_paragraphs(section_text)

            current_chunk_parts = []
            if section_title:
                current_chunk_parts.append(f"## {section_title}")
            current_word_count = 0

            for para in paragraphs:
                para_words = len(para.split())

                if para_words > self.max_chunk_size:
                    if current_chunk_parts:
                        chunk_text = "\n\n".join(current_chunk_parts)
                        chunks.append(Chunk(
                            text=chunk_text,
                            index=index,
                            strategy="dynamic",
                            start_char=current_pos,
                            end_char=current_pos + len(chunk_text),
                            metadata={"section": section_title},
                        ))
                        index += 1
                        current_pos += len(chunk_text)
                        current_chunk_parts = []
                        current_word_count = 0

                    sub_chunks = self._fixed_chunking(para)
                    for sc in sub_chunks:
                        sc.index = index
                        sc.strategy = "dynamic-fallback"
                        sc.metadata["section"] = section_title
                        chunks.append(sc)
                        index += 1

                elif current_word_count + para_words > self.max_chunk_size and current_chunk_parts:
                    chunk_text = "\n\n".join(current_chunk_parts)
                    chunks.append(Chunk(
                        text=chunk_text,
                        index=index,
                        strategy="dynamic",
                        start_char=current_pos,
                        end_char=current_pos + len(chunk_text),
                        metadata={"section": section_title},
                    ))
                    index += 1
                    current_pos += len(chunk_text)

                    current_chunk_parts = [para]
                    current_word_count = para_words

                else:
                    current_chunk_parts.append(para)
                    current_word_count += para_words

            if current_chunk_parts:
                chunk_text = "\n\n".join(current_chunk_parts)
                chunks.append(Chunk(
                    text=chunk_text,
                    index=index,
                    strategy="dynamic",
                    start_char=current_pos,
                    end_char=current_pos + len(chunk_text),
                    metadata={"section": section_title},
                ))
                index += 1
                current_pos += len(chunk_text)

        return chunks

    def _split_by_headings(self, text: str) -> List[tuple]:
        heading_pattern = re.compile(
            r"^(#{1,6}\s+.+|"          # ## Heading
            r"\[صفحة \d+\]|"            # [صفحة 1]
            r"الفصل\s+\w+|"             # الفصل الأول
            r"Chapter\s+\d+)",          # Chapter 1
            re.MULTILINE
        )

        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [("", text)]

        sections = []
        for i, match in enumerate(matches):
            title = match.group().strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((title, section_text))

        if matches[0].start() > 0:
            intro = text[:matches[0].start()].strip()
            if intro:
                sections.insert(0, ("مقدمة", intro))

        return sections

    def _split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n{2,}", text)
        return [p.strip() for p in paragraphs if p.strip()]
