import streamlit as st
import requests

def safe_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return s

def short_text(s: str, n: int = 240) -> str:
    s = safe_text(s)
    return s[:n] + ("..." if len(s) > n else "")

PLACEHOLDER = "https://via.placeholder.com/120x180?text=No+Cover"

st.set_page_config(
    page_title="Book Semantic Search",
    page_icon="📚",
    layout="wide",
)

# --- minimal styling ---
st.markdown(
    """
<style>
/* page width */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}

/* cards */
.card {
  border: 1px solid rgba(120,120,120,0.25);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.02);
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  margin-bottom: 12px;
}
.meta {opacity: 0.75; font-size: 0.92rem;}
.title {font-size: 1.05rem; font-weight: 650; margin-bottom: 6px;}
.snip {opacity: 0.9; line-height: 1.35;}
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.82rem;
  border: 1px solid rgba(120,120,120,0.35);
  opacity: 0.9;
}
hr {margin: 10px 0 18px 0; opacity: 0.25;}
</style>
""",
    unsafe_allow_html=True,
)
# --- header ---
left, right = st.columns([0.72, 0.28], gap="large")
with left:
    st.title("📚 Book Semantic Search")
    st.caption("FAISS + Embeddings + FastAPI • Demo UI (Streamlit)")
with right:
    st.markdown("###")
    st.info("Tip: Use short queries like **innovation strategy**.")

st.markdown("---")

# --- sidebar controls ---
with st.sidebar:
    st.header("Controls")
    if "query" not in st.session_state:
        st.session_state["query"] = ""
    query = st.text_input("Search query", key="query")
    k = st.slider("Top K", 3, 15, 5)
    st.caption("Next step: connect this UI to your FastAPI endpoints.")

if "results" not in st.session_state:
    st.session_state["results"] = []
if "similar" not in st.session_state:
    st.session_state["similar"] = None

if st.button("Search", type="primary", use_container_width=True, key="search_btn"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/search",
            json={"query": query, "k": k},
            timeout=20,
        )
        response.raise_for_status()
        st.session_state["results"] = response.json().get("results", [])
    except Exception as e:
        st.error(f"Search failed: {e}")

results = st.session_state["results"]

# --- search area ---
cta_left, cta_right = st.columns([0.85, 0.15])
with cta_left:
    st.subheader("Results")
    st.caption(f'Query: "{query}" • Top {k}')
with cta_right:
    pass

similar_payload = st.session_state["similar"]
if similar_payload:
    top_cols = st.columns([0.85, 0.15])
    with top_cols[0]:
        st.markdown("### Similar recommendations")
    with top_cols[1]:
        if st.button("Clear", key="clear_similar"):
            st.session_state["similar"] = None
            st.experimental_rerun()

    with st.expander("Show / hide similar results", expanded=True):
        seed = similar_payload.get("seed", {})
        st.markdown(
            f"""
<div class="card">
  <div class="title">Seed: {seed.get("title","")}</div>
  <div style="height:10px"></div>
  <div class="snip">{seed.get("snippet","")}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.caption("Recommendations:")
        for rr in similar_payload.get("results", []):
            st.markdown(
                f"""
<div class="card">
  <div class="title">{rr.get("title","")}</div>
  <div style="height:10px"></div>
  <div class="snip">{rr.get("snippet","")}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

# --- render cards ---
for r in results:
    title = safe_text(r.get("title")) or "(Untitled)"
    book_id = safe_text(r.get("book_id"))
    img = r.get("cover_url") or PLACEHOLDER

    full = safe_text(r.get("full_description"))  
    short = short_text(full, n=240)

    st.markdown(
        f"""
<div class="card">
  <div style="display:flex; gap:14px; align-items:flex-start;">
    <img src="{img}" style="width:92px; border-radius:12px; border:1px solid rgba(120,120,120,0.25);" />
    <div style="flex:1;">
      <div class="title">{title}</div>
      <div style="height:8px"></div>
      <div class="snip">{short}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if len(full) > 240:
        with st.expander("Show more"):
            st.write(full)


    btn_cols = st.columns([0.16, 0.20, 0.64])
    with btn_cols[0]:
        if st.button("Similar", key=f"sim_{book_id}"):
            try:
                resp = requests.get(
                    f"http://127.0.0.1:8000/similar/{book_id}",
                    params={"k": k},
                    timeout=20,
                )
                resp.raise_for_status()
                st.session_state["similar"] = resp.json()
                st.experimental_rerun()  # ✅ 兼容旧 streamlit
            except Exception as e:
                st.error(f"Similar failed: {e}")

    with btn_cols[1]:
        if book_id:
            st.markdown(f"[OpenLibrary](https://openlibrary.org{book_id})")

    with btn_cols[2]:
        pass
            
            

st.caption("✅ Layout done. Next step: replace fake_results with real results from FastAPI `/search` and add a “Similar” button per card.")