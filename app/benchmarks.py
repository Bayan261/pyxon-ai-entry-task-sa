
import time
import re
import statistics
from dataclasses import dataclass, field
from typing import List, Dict
from loguru import logger

from app.chunker import IntelligentChunker, Chunk
from app.vector_store import VectorStore


@dataclass
class BenchmarkResult:
    test_name:   str
    score:       float        # 0.0 إلى 1.0
    details:     Dict = field(default_factory=dict)
    passed:      bool = True
    notes:       str = ""

    def __str__(self):
        status = "✅ نجح" if self.passed else "❌ فشل"
        return f"{status} | {self.test_name}: {self.score:.2%} | {self.notes}"


class BenchmarkSuite:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.chunker = IntelligentChunker()

    def run_all(self) -> Dict:
        logger.info("🚀 بدء تشغيل الاختبارات...")
        results = []

        results.extend(self._test_chunking_quality())

        results.extend(self._test_arabic_support())

        results.extend(self._test_performance())

        if self.vector_store.collection.count() > 0:
            results.extend(self._test_retrieval_accuracy())

        scores = [r.score for r in results]
        overall = statistics.mean(scores) if scores else 0

        passed = sum(1 for r in results if r.passed)
        total  = len(results)

        report = {
            "overall_score": overall,
            "passed":        passed,
            "failed":        total - passed,
            "total_tests":   total,
            "results":       results,
        }

        logger.success(f"انتهت الاختبارات | النتيجة الكلية: {overall:.2%} | نجح: {passed}/{total}")
        return report


    def _test_chunking_quality(self) -> List[BenchmarkResult]:
        results = []

        test_text = "هذا نص تجريبي. " * 100
        chunks = self.chunker.chunk(test_text, {})
        non_empty = all(len(c.text.strip()) > 0 for c in chunks)
        results.append(BenchmarkResult(
            test_name="Non-empty chunks",
            score=1.0 if non_empty else 0.0,
            passed=non_empty,
            details={"chunk_count": len(chunks)},
        ))

        if chunks:
            sizes = [c.word_count() for c in chunks]
            within_bounds = all(
                self.chunker.min_chunk_size <= s <= self.chunker.max_chunk_size + 100
                for s in sizes
            )
            score = sum(
                1 for s in sizes
                if self.chunker.min_chunk_size <= s <= self.chunker.max_chunk_size + 100
            ) / len(sizes)
            results.append(BenchmarkResult(
                test_name="Chunk size within bounds",
                score=score,
                passed=score >= 0.8,
                details={"min_size": min(sizes), "max_size": max(sizes), "avg_size": statistics.mean(sizes)},
                notes=f"Avg size: {statistics.mean(sizes):.0f} words"
            ))

        flat_text = "هذه جملة عادية. " * 200
        flat_chunks = self.chunker.chunk(flat_text, {})
        correct_strategy = all(c.strategy in ["fixed", "dynamic"] for c in flat_chunks)

        structured_text = "## الفصل الأول\n" + "محتوى الفصل الأول. " * 50
        structured_text += "\n\n## الفصل الثاني\n" + "محتوى الفصل الثاني. " * 50
        struct_chunks = self.chunker.chunk(structured_text, {"headings": ["الفصل الأول", "الفصل الثاني"]})
        has_dynamic = any(c.strategy == "dynamic" for c in struct_chunks)

        results.append(BenchmarkResult(
            test_name="Correct strategy selection",
            score=1.0 if has_dynamic else 0.5,
            passed=has_dynamic,
            notes="Dynamic selected for structured text" if has_dynamic else "Dynamic not selected for structured text"
        ))

        return results


    def _test_arabic_support(self) -> List[BenchmarkResult]:
        results = []

        arabic_plain    = "الذكاء الاصطناعي يغير العالم بشكل سريع ومتزايد."
        arabic_diacritics = "الذَّكَاءُ الِاصْطِنَاعِيُّ يُغَيِّرُ الْعَالَمَ بِشَكْلٍ سَرِيعٍ وَمُتَزَايِدٍ."

        long_diacritics = arabic_diacritics * 30
        chunks = self.chunker.chunk(long_diacritics, {})

        diacritic_pattern = re.compile(r"[\u064B-\u065F]")  # Unicode range للحركات
        chunks_with_diacritics = sum(
            1 for c in chunks if diacritic_pattern.search(c.text)
        )
        preservation_rate = chunks_with_diacritics / len(chunks) if chunks else 0

        results.append(BenchmarkResult(
            test_name="Diacritics preservation (harakat)",
            score=preservation_rate,
            passed=preservation_rate >= 0.9,
            details={"chunks_with_diacritics": chunks_with_diacritics, "total_chunks": len(chunks)},
            notes=f"Preservation rate: {preservation_rate:.2%}"
        ))

        arabic_paragraphs = """بسم الله الرحمن الرحيم

هذا الفصل الأول من الكتاب الذي يتناول موضوع الذكاء الاصطناعي.
الفقرة الثانية تتحدث عن تطبيقات الذكاء الاصطناعي في الحياة اليومية.


يستعرض هذا الفصل تحديات الذكاء الاصطناعي ومستقبله.
كما يتناول الجوانب الأخلاقية لهذه التقنية الحديثة."""

        arabic_chunks = self.chunker.chunk(arabic_paragraphs, {"headings": ["الفصل الثاني"]})
        arabic_preserved = all(
            re.search(r"[\u0600-\u06FF]", c.text)
            for c in arabic_chunks
        )

        results.append(BenchmarkResult(
            test_name="Arabic text chunking",
            score=1.0 if arabic_preserved and arabic_chunks else 0.0,
            passed=bool(arabic_preserved and arabic_chunks),
            details={"chunks_count": len(arabic_chunks)},
            notes=f"Chunks produced: {len(arabic_chunks)}"
        ))

        try:
            arabic_query = "ما هو الذكاء الاصطناعي؟"
            embedding = self.vector_store.model.encode([arabic_query])
            embedding_works = len(embedding[0]) > 0
            results.append(BenchmarkResult(
                test_name="Arabic embedding",
                score=1.0 if embedding_works else 0.0,
                passed=embedding_works,
                notes=f"Vector size: {len(embedding[0])}"
            ))
        except Exception as e:
            results.append(BenchmarkResult(
                test_name="Arabic embedding",
                score=0.0,
                passed=False,
                notes=f"خطأ: {e}"
            ))

        return results


    def _test_performance(self) -> List[BenchmarkResult]:
        results = []

        long_text = ("هذه جملة اختبار للأداء تحتوي على كلمات عديدة. " * 200 +
                     "This is an English test sentence for performance benchmarking. " * 100)

        start = time.time()
        chunks = self.chunker.chunk(long_text, {})
        elapsed = time.time() - start

        speed_ok = elapsed < 5.0
        speed_score = max(0, 1 - (elapsed / 5.0))

        results.append(BenchmarkResult(
            test_name="Chunking speed",
            score=speed_score,
            passed=speed_ok,
            details={"elapsed_seconds": round(elapsed, 3), "chunks_produced": len(chunks)},
            notes=f"{elapsed:.2f}s for {len(long_text.split())} words"
        ))

        total_chunk_words = sum(c.word_count() for c in chunks)
        original_words = len(long_text.split())
        retention = min(1.0, total_chunk_words / original_words) if original_words > 0 else 0

        results.append(BenchmarkResult(
            test_name="Retention efficiency",
            score=retention,
            passed=retention >= 0.95,
            details={"original_words": original_words, "chunk_words": total_chunk_words},
            notes=f"Retention rate: {retention:.2%}"
        ))

        return results


    def _test_retrieval_accuracy(self) -> List[BenchmarkResult]:
        results = []

        test_queries = [
            "document content",
            "text analysis",
            "information",
        ]

        hits = 0
        for query in test_queries:
            search_results = self.vector_store.search(query, n_results=3)
            if search_results and search_results[0]["similarity"] > 0.2:
                hits += 1

        test_pairs = test_queries  # for len()
        accuracy = hits / len(test_queries) if test_queries else 0

        results.append(BenchmarkResult(
            test_name="Retrieval accuracy",
            score=accuracy,
            passed=accuracy >= 0.6,
            details={"hits": hits, "total_queries": len(test_pairs)},
            notes=f"{hits}/{len(test_pairs)} correct queries"
        ))

        return results


    def print_report(self, report: Dict):
        print("\n" + "="*60)
        print("📊 تقرير الاختبارات - Pyxon AI Document Parser")
        print("="*60)
        print(f"\n🎯 النتيجة الكلية: {report['overall_score']:.2%}")
        print(f"✅ نجح: {report['passed']} | ❌ فشل: {report['failed']} | المجموع: {report['total_tests']}")
        print("\n" + "-"*60)

        for result in report["results"]:
            print(result)
            if result.details:
                for k, v in result.details.items():
                    print(f"   └ {k}: {v}")

        print("="*60 + "\n")
