# BookWise AI — Smart Book Explorer

A polished Streamlit app for scraping the public demo catalogue at Books to Scrape, searching books with TF-IDF/cosine similarity, exploring categories, and viewing a basic regression-based price estimate.

bookwise-ai/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── packages.txt
## Run locally
Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install and run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in your terminal (usually `http://localhost:8501`).

## Publish to GitHub
1. Sign in to GitHub and create a **public** repository, for example `bookwise-ai`.
2. Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and `packages.txt`.
3. Do not commit `.venv`, passwords, API keys, or private data.
4. The app creates the CSV when you fetch the catalogue. You can choose to commit the CSV if you intentionally want a fixed sample dataset, but it is not required.

If Git is installed, you can also run:
```bash
git init
git add app.py requirements.txt README.md .gitignore packages.txt
git commit -m "Build BookWise AI Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/bookwise-ai.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username.

## Deploy free on Streamlit Community Cloud
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Choose your `bookwise-ai` repository, branch `main`, and main file path `app.py`.
4. Select **Deploy**. Streamlit installs packages from `requirements.txt`.
5. Wait for the build to finish, then copy the public app URL to share it.

## Google Drive
The app includes a **Download catalogue CSV** button. To store the data in Google Drive without setting up Google API credentials, download the CSV and upload it to Drive manually.

For automatic Google Drive syncing, a Google Cloud project, Drive API enablement, OAuth/service-account setup, and securely stored credentials are required. Do not upload credential JSON files or secrets to a public GitHub repository.

## Notes
- The demo website is intended for scraping practice. Keep request volume modest.
- Scraping product detail pages can take a little while.
- The regression score is an educational metric, not a guarantee of real-world price accuracy. `MarketScore` is calculated using the listed price, so the price model has target leakage.
