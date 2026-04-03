import os
import tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from app.parser import DocumentParser
from app.chunker import IntelligentChunker
from app.database import DatabaseManager
from app.vector_store import VectorStore
from app.rag import RAGSystem
from app.benchmarks import BenchmarkSuite

st.set_page_config(
    page_title="Pyxon AI - Document Parser",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
    [data-testid="stMetric"] {
        border: 1px solid #444;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

@st.cache_resource
def load_components():
    parser       = DocumentParser()
    chunker      = IntelligentChunker()
    db           = DatabaseManager("pyxon_demo.db")
    vector_store = VectorStore(persist_dir="./chroma_demo")
    return parser, chunker, db, vector_store

parser, chunker, db, vector_store = load_components()

with st.sidebar:
    st.markdown("### Pyxon AI")
    st.markdown("Document Parser")
    st.divider()
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""), placeholder="sk-...")
    st.divider()
    stats_db = db.get_stats()
    stats_vs = vector_store.get_stats()
    st.markdown("**Overview**")
    col1, col2 = st.columns(2)
    col1.metric("Documents", stats_db["total_documents"])
    col2.metric("Chunks", stats_db["total_chunks"])
    col1.metric("Vectors", stats_vs["total_vectors"])
    col2.metric("Arabic", stats_db["arabic_documents"])

tab1, tab2, tab3 = st.tabs(["Upload", "Search", "Benchmark"])

with tab1:
    st.markdown("## Document Processing")
    st.caption("Supports PDF, DOCX, and TXT with full Arabic and diacritics support.")
    st.divider()
    uploaded_file = st.file_uploader("Select a file", type=["pdf", "docx", "txt"], label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Processing..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                result    = parser.parse(tmp_path)
                result["metadata"]["file_name"] = uploaded_file.name
                chunks    = chunker.chunk(result["text"], result["metadata"])
                strategy  = chunks[0].strategy if chunks else "unknown"
                doc_id    = db.save_document(result["metadata"], chunks, strategy)
                chunk_ids = vector_store.add_chunks(chunks, doc_id, uploaded_file.name)

                for chunk, cid in zip(chunks, chunk_ids):
                    db.update_chunk_chroma_id(doc_id, chunk.index, cid)

                os.unlink(tmp_path)
                st.success("Document processed successfully.")
                st.divider()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Words",    result["metadata"]["word_count"])
                c2.metric("Chunks",   len(chunks))
                c3.metric("Language", result["metadata"]["language"])
                c4.metric("Strategy", strategy)

                st.divider()
                st.markdown("**Chunk Preview**")
                for i, chunk in enumerate(chunks[:5]):
                    with st.expander(f"Chunk {i+1}  —  {chunk.word_count()} words  |  {chunk.strategy}"):
                        st.text_area("", value=chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""), height=130, disabled=True, key=f"chunk_{i}")

                if len(chunks) > 5:
                    st.caption(f"Showing 5 of {len(chunks)} chunks.")

            except Exception as e:
                st.error(f"Error: {e}")
                if 'tmp_path' in locals():
                    try: os.unlink(tmp_path)
                    except: pass

with tab2:
    st.markdown("## Search & Question Answering")
    st.caption("Search across processed documents using semantic similarity.")
    st.divider()

    if db.get_stats()["total_documents"] == 0:
        st.info("No documents available. Upload a document first.")
    else:
        question = st.text_input("Query", placeholder="Enter your question...", label_visibility="collapsed")
        c1, c2 = st.columns([1, 1])
        search_btn = c1.button("Search", use_container_width=True)
        rag_btn    = c2.button("Ask (RAG)", type="primary", use_container_width=True)

        if search_btn and question:
            with st.spinner("Searching..."):
                results = vector_store.search(question, n_results=5)
            st.divider()
            st.markdown("**Results**")
            for r in results:
                with st.expander(f"Result {r['rank']}  —  {r['similarity']:.2%}  |  {r.get('file_name', '')}"):
                    st.text(r["text"][:400])
                    st.caption(f"Section: {r.get('section','—')}  |  Strategy: {r.get('strategy','—')}")

        if rag_btn and question:
            if not api_key:
                st.warning("Please enter your OpenAI API Key in the sidebar.")
            else:
                with st.spinner("Generating answer..."):
                    rag    = RAGSystem(vector_store=vector_store, api_key=api_key)
                    result = rag.answer(question)
                st.divider()
                st.markdown("**Answer**")
                st.markdown(result["answer"])
                st.markdown("**Sources**")
                for src in result["sources"]:
                    with st.expander(f"Source {src['rank']}  —  {src['similarity']:.2%}"):
                        st.text(src["text"][:300])

with tab3:
    st.markdown("## Benchmark")
    st.caption("Evaluates chunking quality, Arabic support, retrieval accuracy, and performance.")
    st.divider()

    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Running tests..."):
            suite  = BenchmarkSuite(vector_store)
            report = suite.run_all()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Score", f"{report['overall_score']:.2%}")
        c2.metric("Passed",        report["passed"])
        c3.metric("Failed",        report["failed"])
        c4.metric("Total Tests",   report["total_tests"])

        st.divider()
        st.markdown("**Test Details**")
        data = [{"Test": r.test_name, "Score": f"{r.score:.2%}", "Status": "Passed" if r.passed else "Failed", "Notes": r.notes} for r in report["results"]]
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
