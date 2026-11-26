"""ChemIntel demo (Streamlit).  Run:  streamlit run app.py
Retrieves ChemPile passages (BM25) and, if a generator is configured, answers with them as context.
Without a generator it shows the retrieved passages only; it never invents an answer."""
import os, sys
import streamlit as st
sys.path.insert(0, os.path.dirname(__file__))
from chemintel.bm25 import BM25Index
from chemintel.pdfqa import build_pdf_index, search_many
from chemintel.generators import HFGenerator, OpenAICompatGenerator, build_rag_prompt

st.set_page_config(page_title="ChemIntel", layout="wide")
st.title("ChemIntel: retrieval-grounded chemistry assistant")
st.caption("Retrieves passages from a ChemPile-Education (LibreTexts) subset. Retrieved text is shown as-is: it is source material, not a verified answer.")


@st.cache_resource
def get_index(path):
    return BM25Index.load(path)


@st.cache_resource
def get_generator(kind, model, adapter, base_url, key):
    if kind == "Local Hugging Face model":
        return HFGenerator(model, adapter or None)
    return OpenAICompatGenerator(model, base_url, key or None)


with st.sidebar:
    st.header("Settings")
    index_path = st.text_input("BM25 index file", "data/bm25.pkl")
    k = st.slider("Passages to retrieve", 1, 8, 3)
    kind = st.selectbox("Generator", ["None (retrieval only)", "Local Hugging Face model", "OpenAI-compatible endpoint (e.g. Ollama)"])
    model = base_url = key = adapter = ""
    if kind.startswith("Local"):
        model = st.text_input("Model", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
        adapter = st.text_input("LoRA adapter folder (optional)", "")
    elif kind.startswith("OpenAI"):
        model = st.text_input("Model name", "qwen2.5:1.5b")
        base_url = st.text_input("Base URL", "http://localhost:11434/v1")
        key = st.text_input("API key (kept in this session only)", "", type="password")
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    source = st.radio("Knowledge source", ["ChemPile index", "Uploaded PDF", "Both"])
    up = st.file_uploader("PDF (text-based, e.g. a datasheet)", type=["pdf"])
    if up is not None:
        key_ = (up.name, up.size)
        if st.session_state.get("pdf_key") != key_:          # re-index whenever a different file is uploaded
            idx_, info_ = build_pdf_index(up.getvalue(), up.name)
            st.session_state["pdf_key"], st.session_state["pdf_index"], st.session_state["pdf_info"] = key_, idx_, info_
        st.caption(f"PDF: {st.session_state['pdf_info']}")
        if st.session_state["pdf_info"]["pages_with_text"] == 0:
            st.warning("No extractable text found. This may be a scanned PDF; OCR is not included.")
    else:
        st.session_state.pop("pdf_key", None); st.session_state.pop("pdf_index", None)

indexes = []
if source in ("ChemPile index", "Both"):
    if not os.path.exists(index_path):
        st.error(f"Index not found at {index_path}. Run scripts/01_prepare_corpus.py and scripts/02_build_index.py first.")
        st.stop()
    index = get_index(index_path)
    indexes.append(index)
    st.sidebar.write(f"ChemPile index: {len(index.chunks):,} passages")
if source in ("Uploaded PDF", "Both"):
    if "pdf_index" not in st.session_state:
        st.warning("Upload a PDF in the sidebar to use this source.")
        st.stop()
    indexes.append(st.session_state["pdf_index"])

q = st.text_area("Ask a chemistry question", "Why does adding salt raise the boiling point of water?", height=90)
if st.button("Ask", type="primary") and q.strip():
    passages = search_many(indexes, q, k=k)
    left, right = st.columns([3, 2])
    with right:
        st.subheader("Retrieved passages")
        for i, p in enumerate(passages, 1):
            with st.expander(f"[{i}] score {p['score']:.1f}  {p['source'][:70]}"):
                st.write(p["text"])
                st.caption(p["source"])
    with left:
        st.subheader("Answer")
        if not passages:
            st.warning("No passage in the selected source shares any word with the question, so no answer is generated.")
        elif kind.startswith("None"):
            st.info("No generator is configured, so no answer is generated. The retrieved passages on the right are the output. "
                    "Choose a generator in the sidebar to get a written answer.")
        else:
            try:
                with st.spinner("Generating..."):
                    gen = get_generator(kind, model, adapter, base_url, key)
                    st.write(gen.generate(build_rag_prompt(q, passages), temperature=temp))
            except Exception as e:
                st.error(f"Generator failed: {e}")

st.divider()
st.subheader("Datasheet fields (rule-based, heuristic)")
if "pdf_index" in st.session_state:
    if st.button("Extract fields from uploaded PDF"):
        from chemintel.datasheet import extract_regex
        found = extract_regex(st.session_state["pdf_index"].chunks)
        if found:
            st.dataframe([{k: d[k] for k in ("field", "value", "unit", "source", "method", "evidence")} for d in found], use_container_width=True)
            st.caption("Values are copied from the text by patterns. Not checked by a chemist; fields not listed were not found.")
        else:
            st.info("No known fields found in the text.")
else:
    st.caption("Upload a PDF in the sidebar to enable.")

st.subheader("Reaction precedents (retrieval only, not a prediction)")
rpath = st.text_input("Reaction records file (JSON lines)", "examples/reactions_SYNTHETIC.jsonl")
rq = st.text_input("Reaction query (reactants, products, conditions)", "acetic acid and ethanol esterification")
if st.button("Find precedents") and os.path.exists(rpath):
    from chemintel.reactions import load_records, build_index, find_precedents, DISCLAIMER
    for h in find_precedents(build_index(load_records(rpath)), rq, k=3):
        r = h["record"]
        st.write(f"**{r.get('reactants','')} -> {r.get('products','')}**  | conditions: {r.get('conditions','')}  | source: {r.get('source','')}")
    st.caption(DISCLAIMER)
