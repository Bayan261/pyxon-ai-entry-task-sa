
from typing import List, Optional
from openai import OpenAI
from loguru import logger

from app.vector_store import VectorStore


class RAGSystem:

    SYSTEM_PROMPT = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة بناءً على المحتوى المقدم لك.

قواعد مهمة:
1. أجب فقط بناءً على السياق المقدم (Context)
2. إذا لم تجد الإجابة في السياق، قل ذلك بوضوح
3. اذكر من أي جزء من الوثيقة استقيت إجابتك
4. إذا كان السؤال بالعربية، أجب بالعربية
5. إذا كان السؤال بالإنجليزية، أجب بالإنجليزية
6. كن دقيقاً وموجزاً"""

    def __init__(
        self,
        vector_store: VectorStore,
        api_key: str,
        model: str = "gpt-4o-mini",  # نموذج اقتصادي وكافٍ
        n_results: int = 5,
        base_url: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.n_results = n_results
        self.model = model

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        logger.info(f"RAG System جاهز | النموذج: {model}")

    def answer(
        self,
        question: str,
        document_id: Optional[int] = None,
        n_results: Optional[int] = None,
    ) -> dict:
        n = n_results or self.n_results

        logger.info(f"جاري البحث عن: '{question[:50]}...' " if len(question) > 50 else f"جاري البحث: '{question}'")
        retrieved_chunks = self.vector_store.search(
            query=question,
            n_results=n,
            document_id=document_id,
        )

        if not retrieved_chunks:
            return {
                "question": question,
                "answer":   "لم أجد معلومات كافية في الوثائق للإجابة على هذا السؤال.",
                "sources":  [],
            }

        context = self._build_context(retrieved_chunks)

        user_message = f"""السياق من الوثائق:
{context}

السؤال: {question}

الإجابة:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,    # منخفض للحصول على إجابات أكثر دقة
                max_tokens=1000,
            )
            answer_text = response.choices[0].message.content

        except Exception as e:
            logger.error(f"خطأ في الاتصال بالـ LLM: {e}")
            answer_text = f"خطأ في الاتصال بالنموذج: {str(e)}"

        logger.success(f"تمت الإجابة | عدد المصادر: {len(retrieved_chunks)}")

        return {
            "question": question,
            "answer":   answer_text,
            "sources":  retrieved_chunks,
        }

    def _build_context(self, chunks: List[dict]) -> str:
        parts = []
        for chunk in chunks:
            source_info = f"[المصدر: {chunk.get('file_name', 'غير معروف')} | القسم: {chunk.get('section', '-')} | التشابه: {chunk.get('similarity', 0):.2%}]"
            parts.append(f"{source_info}\n{chunk['text']}")

        return "\n\n---\n\n".join(parts)
