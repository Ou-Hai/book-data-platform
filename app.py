import streamlit as st
import requests

def inline_show_more(full: str, book_id: str, n: int = 240):
    """Render short text with inline Show more/less toggle."""
    if not full:
        return ""

    state_key = f"desc_open_{book_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    opened = st.session_state[state_key]
    if len(full) <= n:
        return full

    if opened:
        return full
    else:
        return full[:n].rstrip() + "..."

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

.cardbox {
  border: 1px solid rgba(120,120,120,0.25);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.02);
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  margin-bottom: 12px;
}
/* Make SECONDARY buttons look like links (Read more/less only) */
button[kind="secondary"] {
  background: none !important;
  border: none !important;
  padding: 0 !important;
  color: #1a73e8 !important;
  text-decoration: underline !important;
  font-size: 0.9rem !important;
  box-shadow: none !important;
  width: auto !important;
  min-width: 0 !important;
  white-space: nowrap !important;
}

/* Card style using Streamlit container */
div[data-testid="stVerticalBlock"] > div:has(> div.stImage) {
  border: 1px solid rgba(120,120,120,0.25);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.02);
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  margin-bottom: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)
# --- header ---
left, right = st.columns([0.72, 0.28], gap="large")
with left:
    st.title("📚 Book Semantic Search")
with right:
    st.markdown("###")

st.markdown("---")

# --- sidebar controls ---
with st.sidebar:
    st.header("Search")

    st.markdown(
        "<style>div[role='radiogroup']{margin-top:-10px;}</style>",
        unsafe_allow_html=True
    )

    query = st.text_input("What are you in the mood for?", key="query", placeholder="e.g., romance, space opera, innovation…")

    st.markdown("**Results**")
    k = st.radio(
        "How many books?",
        options=[3, 4, 5],
        index=2,
        horizontal=True,
        key="top_k",
        label_visibility="collapsed",
    )

if "results" not in st.session_state:
    st.session_state["results"] = []

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
    if results:
        st.subheader("Results")
        st.caption(f'Top {k} • Query: "{query}"')
with cta_right:
    pass

# --- render cards ---
for idx, r in enumerate(results):
    title = safe_text(r.get("title")) or "(Untitled)"
    book_id = safe_text(r.get("book_id"))
    img = r.get("cover_url") or PLACEHOLDER

    full = safe_text(r.get("full_description")) or ""
    is_long = len(full) > 240
    preview = full[:240].rstrip() + "..." if is_long else full

    opened_key = f"desc_open_{book_id}_{idx}"
    if opened_key not in st.session_state:
        st.session_state[opened_key] = False
    opened = st.session_state[opened_key]

    with st.container():
        col_img, col_main = st.columns([0.18, 0.82])
        
        with col_img:
            st.image(img, width=92)
            
        with col_main:
            st.subheader(title)

            # description + inline Read more on right
            full = safe_text(r.get("full_description")) or ""
            is_long = len(full) > 240
            preview = full[:240].rstrip() + "..." if is_long else full

            opened_key = f"desc_open_{book_id}_{idx}"
            if opened_key not in st.session_state:
                st.session_state[opened_key] = False
            opened = st.session_state[opened_key]

            if not full:
                st.caption("No description available.")
            else:
                text = full if opened else preview
                tcol, lcol = st.columns([0.90, 0.10])
                with tcol:
                    st.write(text)
                with lcol:
                    if is_long:
                        label = "Read less" if opened else "Read more"
                        st.markdown("<div style='text-align:left'>", unsafe_allow_html=True)
                        if st.button(label, key=f"read_{book_id}_{idx}", type="secondary"):
                            st.session_state[opened_key] = not opened
                            st.experimental_rerun()
                        st.markdown("</div>", unsafe_allow_html=True)

        # only keep OpenLibrary
            if book_id:
                st.markdown(f"[OpenLibrary](https://openlibrary.org{book_id})")   