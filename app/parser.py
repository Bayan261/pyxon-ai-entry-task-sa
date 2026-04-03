
import os
import re
import PyPDF2
import docx
from pathlib import Path
from loguru import logger
from typing import Optional


class DocumentParser:

    SUPPORTED_FORMATS = [".pdf", ".docx", ".doc", ".txt"]

    def parse(self, file_path: str) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"الملف غير موجود: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"صيغة غير مدعومة: {ext}. الصيغ المدعومة: {self.SUPPORTED_FORMATS}")

        logger.info(f"جاري قراءة الملف: {path.name}")

        if ext == ".pdf":
            text, metadata = self._parse_pdf(path)
        elif ext in [".docx", ".doc"]:
            text, metadata = self._parse_docx(path)
        elif ext == ".txt":
            text, metadata = self._parse_txt(path)

        text = self._clean_text(text)

        language = self._detect_language(text)

        metadata.update({
            "file_name": path.name,
            "file_path": str(path.absolute()),
            "file_type": ext,
            "file_size_kb": round(path.stat().st_size / 1024, 2),
            "language": language,
            "char_count": len(text),
            "word_count": len(text.split()),
        })

        logger.success(f"تمت القراءة بنجاح: {path.name} | اللغة: {language} | الكلمات: {metadata['word_count']}")

        return {
            "text": text,
            "metadata": metadata,
            "language": language,
        }


    def _parse_pdf(self, path: Path) -> tuple[str, dict]:
        text_parts = []
        num_pages = 0

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[صفحة {page_num + 1}]\n{page_text}")

        return "\n\n".join(text_parts), {"num_pages": num_pages}

    def _parse_docx(self, path: Path) -> tuple[str, dict]:
        doc = docx.Document(str(path))
        text_parts = []
        num_paragraphs = 0
        headings = []

        for para in doc.paragraphs:
            if para.text.strip():
                if para.style.name.startswith("Heading"):
                    headings.append(para.text)
                    text_parts.append(f"\n## {para.text}\n")
                else:
                    text_parts.append(para.text)
                num_paragraphs += 1

        return "\n".join(text_parts), {
            "num_paragraphs": num_paragraphs,
            "headings": headings,
        }

    def _parse_txt(self, path: Path) -> tuple[str, dict]:
        encodings = ["utf-8", "utf-8-sig", "cp1256", "iso-8859-6"]

        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    text = f.read()
                return text, {"encoding_used": encoding}
            except UnicodeDecodeError:
                continue

        raise ValueError(f"لا يمكن قراءة الملف بأي encoding معروف: {path.name}")


    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _detect_language(self, text: str) -> str:
        if not text:
            return "unknown"

        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
        english_chars = len(re.findall(r"[a-zA-Z]", text))

        total = arabic_chars + english_chars
        if total == 0:
            return "unknown"

        arabic_ratio = arabic_chars / total

        if arabic_ratio > 0.7:
            return "arabic"
        elif arabic_ratio < 0.3:
            return "english"
        else:
            return "mixed"
