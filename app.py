import re
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="BookWise AI | Smart Book Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #f7f8fc 0%, #eef2ff 100%); }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1400px; }
.hero { padding: 2rem 2.2rem; border-radius: 24px; color: white;
        background: linear-gradient(120deg, #111827 0%, #312e81 55%, #6d28d9 100%);
        box-shadow: 0 16px 45px rgba(49,46,129,.18); margin-bottom: 1.4rem; }
.hero h1 { font-family: 'Manrope', sans-serif; font-size: 2.45rem; font-weight: 800; margin: 0 0 .45rem 0; }
.hero p { color: #e0e7ff; font-size: 1.05rem; margin: 0; }
.metric-card { background: white; border: 1px solid #e5e7eb; border-radius: 18px;
               padding: 1rem 1.2rem; box-shadow: 0 6px 20px rgba(31,41,55,.04); }
.book-card { background: white; border: 1px solid #e5e7eb; border-radius: 18px;
             padding: 1.15rem; margin-bottom: .8rem; box-shadow: 0 6px 18px rgba(31,41,55,.035); }
.small-muted { color: #64748b; font-size: .9rem; }
div.stButton > button { border-radius: 12px; font-weight: 700; }
[data-testid="stSidebar"] { background: #111827; }
[data-testid="stSidebar"] * { color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

CATALOGUE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
BASE_BOOK_URL = "https://books.toscrape.com/catalogue/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BookWiseAI/1.0)"}
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
DATA_FILE = Path("complete_books_database.csv")


@st.cache_data(show_spinner=False, ttl=3600)
def scrape_books(total_pages: int) -> pd.DataFrame:
    books = []
    book_id = 1
    session = requests.Session()
    for page in range(1, total_pages + 1):
        response = session.get(CATALOGUE_URL.format(page), headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for article in soup.find_all("article", class_="product_pod"):
            title = article.h3.a.get("title", "Unknown title")
            href = article.h3.a.get("href", "")
            detail_url = requests.compat.urljoin(CATALOGUE_URL.format(page), href)
            rating_classes = article.find("p", class_="star-rating").get("class", [])
            rating_word = rating_classes[1] if len(rating_classes) > 1 else "Three"
            rating = RATING_MAP.get(rating_word, 3)

            detail_resp = session.get(detail_url, headers=HEADERS, timeout=20)
            detail_resp.raise_for_status()
            detail_soup = BeautifulSoup(detail_resp.content, "html.parser")
            crumbs = detail_soup.select("ul.breadcrumb li")
            category = crumbs[2].get_text(" ", strip=True) if len(crumbs) > 2 else "General"

            desc_header = detail_soup.find("div", id="product_description")
            desc_p = desc_header.find_next_sibling("p") if desc_header else None
            description = desc_p.get_text(" ", strip=True) if desc_p else "No description available."

            table_data = {}
            for row in detail_soup.select("table.table-striped tr"):
                th, td = row.find("th"), row.find("td")
                if th and td:
                    table_data[th.get_text(strip=True)] = td.get_text(strip=True)

            availability = table_data.get("Availability", "0")
            stock_match = re.search(r"\d+", availability)
            stock_qty = int(stock_match.group()) if stock_match else 0
            price_raw = table_data.get("Price (incl. tax)", "0")
            price_match = re.search(r"\d+(?:\.\d+)?", price_raw.replace(",", ""))
            price = float(price_match.group()) if price_match else 0.0
            market_score = round((price * .9) + (rating * 1.5) + (stock_qty * .05), 2)

            books.append({
                "Book_ID": book_id, "Title": title, "Category": category,
                "UPC": table_data.get("UPC", "N/A"), "StarRating": rating,
                "StockQuantity": stock_qty, "ActualPrice": price,
                "MarketScore": market_score, "Description": description, "URL": detail_url
            })
            book_id += 1
    return pd.DataFrame(books)


@st.cache_resource(show_spinner=False)
def build_models(df: pd.DataFrame):
    if len(df) < 3:
        return None, None, None, None
    X = df[["StarRating", "StockQuantity", "MarketScore"]]
    y = df["ActualPrice"]
    if y.nunique() < 2:
        model = LinearRegression().fit(X, y)
        score = 0.0
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression().fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test)) if len(X_test) >= 2 else 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform((df["Title"].fillna("") + " " + df["Category"].fillna("")))
    return model, score, vectorizer, matrix


def search_books(query: str, df: pd.DataFrame, vectorizer, matrix, limit: int = 12):
    query = query.strip()
    if not query:
        return pd.DataFrame()
    if query.isdigit():
        return df[df["Book_ID"] == int(query)]
    direct = df[df["Title"].str.contains(re.escape(query), case=False, na=False)]
    if not direct.empty:
        return direct.head(limit)
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    result = df.copy()
    result["Similarity"] = scores
    return result[result["Similarity"] > 0.12].sort_values("Similarity", ascending=False).head(limit)


st.markdown("""
<div class="hero">
  <h1>📚 BookWise AI</h1>
  <p>Your smart book explorer — discover titles, compare prices, explore categories, and get AI-assisted price estimates.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.markdown("Choose how to load the book catalogue.")
    source = st.radio("Data source", ["Use saved CSV", "Scrape website"])
    pages = st.slider("Catalogue pages to scrape", min_value=1, max_value=5, value=2,
                      help="Each page contains 20 books. Scraping details takes a little time.")
    st.markdown("---")
    st.markdown("### About")
    st.caption("Demo data source: Books to Scrape. This app is for learning and demonstration.")

df = None
if source == "Use saved CSV" and DATA_FILE.exists():
    try:
        df = pd.read_csv(DATA_FILE)
        st.sidebar.success(f"Loaded {len(df)} books from saved CSV.")
    except Exception:
        df = None

if source == "Use saved CSV" and df is None:
    st.info("No saved catalogue found yet. Click **Fetch books** to create one, or choose **Scrape website** in the sidebar.")

if source == "Scrape website" or st.button("Fetch books", type="primary"):
    try:
        with st.spinner("Fetching book listings and details..."):
            df = scrape_books(pages)
            df.to_csv(DATA_FILE, index=False)
        st.success(f"Fetched {len(df)} books and saved them to `{DATA_FILE}`.")
    except requests.RequestException as exc:
        st.error(f"Could not fetch the catalogue. Check your internet connection and try again. Details: {exc}")
    except Exception as exc:
        st.error(f"Something went wrong while processing the catalogue: {exc}")

if df is None or df.empty:
    st.markdown("### Getting started")
    st.write("Use the sidebar to load a saved catalogue or scrape the demo book website.")
    st.markdown("""
    - **Search:** find a book by title, category, or book ID.
    - **Explore:** filter the catalogue by category, rating, and price.
    - **AI estimate:** view a simple regression estimate alongside the listed price.
    - **Export:** download your catalogue as a CSV file.
    """)
    st.stop()

required = {"Book_ID", "Title", "Category", "StarRating", "StockQuantity", "ActualPrice", "MarketScore", "Description", "URL"}
missing = required - set(df.columns)
if missing:
    st.error(f"The CSV is missing required columns: {', '.join(sorted(missing))}")
    st.stop()

df["Title"] = df["Title"].fillna("Untitled")
df["Category"] = df["Category"].fillna("General")
df["Description"] = df["Description"].fillna("No description available.")
model, model_score, vectorizer, tfidf_matrix = build_models(df)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Books in catalogue", f"{len(df):,}")
with m2:
    st.metric("Categories", f"{df['Category'].nunique():,}")
with m3:
    st.metric("Average price", f"£{df['ActualPrice'].mean():.2f}")
with m4:
    st.metric("Average rating", f"{df['StarRating'].mean():.1f}/5")

tab_search, tab_explore, tab_data, tab_about = st.tabs(["🔎 Smart Search", "🧭 Explore Catalogue", "📊 Data & Export", "ℹ️ About AI"])

with tab_search:
    st.subheader("Find your next read")
    search_col, button_col = st.columns([5, 1])
    with search_col:
        query = st.text_input("Search by title, category, or book ID", placeholder="Try: mystery, history, or 12")
    with button_col:
        st.write("")
        st.write("")
        search_clicked = st.button("Search", type="primary", use_container_width=True)
    if query:
        results = search_books(query, df, vectorizer, tfidf_matrix)
        if results.empty:
            st.warning("No close match found. Try a shorter title or a category such as fiction, mystery, or history.")
        else:
            st.caption(f"{len(results)} result(s) shown.")
            for _, row in results.iterrows():
                predicted = float(model.predict(pd.DataFrame([{
                    "StarRating": row["StarRating"],
                    "StockQuantity": row["StockQuantity"],
                    "MarketScore": row["MarketScore"]
                }]))[0]) if model is not None else float(row["ActualPrice"])
                similarity_text = f" · {row['Similarity']:.0%} match" if "Similarity" in row.index else ""
                st.markdown(f"""
                <div class="book-card">
                  <h3 style="margin:0 0 .35rem 0;">{row['Title']}</h3>
                  <div class="small-muted">{row['Category']}{similarity_text}</div>
                </div>
                """, unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Listed price", f"£{float(row['ActualPrice']):.2f}")
                c2.metric("Rating", f"{int(row['StarRating'])}/5")
                c3.metric("Stock", f"{int(row['StockQuantity'])}")
                c4.metric("AI price estimate", f"£{predicted:.2f}")
                with st.expander("Read description and details"):
                    st.write(row["Description"])
                    st.write(f"**UPC:** {row.get('UPC', 'N/A')}")
                    st.write(f"**Book ID:** {int(row['Book_ID'])}")
                    st.link_button("Open book page ↗", row["URL"])
    else:
        st.info("Enter a book title, category, or ID to start searching.")
        popular = df.sort_values(["StarRating", "ActualPrice"], ascending=[False, True]).head(3)
        st.markdown("#### Recommended from this catalogue")
        cols = st.columns(3)
        for col, (_, row) in zip(cols, popular.iterrows()):
            with col:
                st.markdown(f"""<div class="book-card"><h4>{row['Title']}</h4>
                <p class="small-muted">{row['Category']}</p>
                <b>£{float(row['ActualPrice']):.2f}</b><br>⭐ {int(row['StarRating'])}/5</div>""", unsafe_allow_html=True)

with tab_explore:
    st.subheader("Explore all books")
    categories = sorted(df["Category"].dropna().unique().tolist())
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        selected_categories = st.multiselect("Categories", categories, default=[])
    with f2:
        min_rating = st.selectbox("Minimum rating", [1, 2, 3, 4, 5], index=0)
    with f3:
        max_price = st.number_input("Maximum price (£)", min_value=0.0,
                                    max_value=float(max(1, df["ActualPrice"].max())),
                                    value=float(df["ActualPrice"].max()), step=1.0)
    filtered = df.copy()
    if selected_categories:
        filtered = filtered[filtered["Category"].isin(selected_categories)]
    filtered = filtered[(filtered["StarRating"] >= min_rating) & (filtered["ActualPrice"] <= max_price)]
    st.caption(f"{len(filtered)} book(s) match your filters.")
    st.dataframe(
        filtered[["Book_ID", "Title", "Category", "StarRating", "StockQuantity", "ActualPrice"]]
        .rename(columns={"Book_ID": "ID", "StarRating": "Rating", "StockQuantity": "In stock", "ActualPrice": "Price (£)"}),
        use_container_width=True, hide_index=True
    )

with tab_data:
    st.subheader("Catalogue data")
    st.write("Download the currently loaded catalogue, or inspect the raw data.")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download catalogue CSV", csv_bytes, "complete_books_database.csv",
                       "text/csv", type="primary")
    st.dataframe(df.drop(columns=["Description"], errors="ignore"), use_container_width=True, hide_index=True)
    st.caption("The CSV download can be uploaded to Google Drive manually. Do not put private credentials in this repository.")

with tab_about:
    st.subheader("How the AI features work")
    st.markdown("""
    **Smart search** uses TF-IDF text features and cosine similarity to find titles related to your query.

    **Price estimate** uses a Linear Regression model trained on the scraped catalogue's star rating, stock quantity, and calculated market score.

    **Important:** this is an educational demonstration, not a reliable commercial pricing model. The target price is also used indirectly to calculate `MarketScore`, so the model's score should not be treated as independent proof of predictive accuracy.
    """)
    st.metric("Regression R² on held-out sample", f"{model_score * 100:.1f}%" if model_score is not None else "N/A")
    st.caption("An R² score can be negative and is not the same thing as classification accuracy.")
