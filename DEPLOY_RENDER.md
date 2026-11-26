# Deploy to Render

1. Push this folder to a GitHub repo (`.env` and `chroma_db/` are git-ignored; never commit your key).
2. Render dashboard -> **New +** -> **Blueprint** -> pick the repo. Render reads `render.yaml`.
3. When asked, paste your `OPENAI_API_KEY`. Click **Apply**.
4. Wait for the Docker build (first build takes several minutes). Open the `https://<name>.onrender.com` URL.

Manual alternative: New + -> Web Service -> Runtime **Docker** -> add env var `OPENAI_API_KEY`
-> (paid plans) Disks -> Add Disk, mount path `/var/data`, plus env var `CHROMA_DIR=/var/data/chroma_db`.

Free plan: no persistent disk and the app sleeps after ~15 min idle, so uploaded
data is lost and must be re-processed. Use a paid instance + disk to keep the database.
Check Render's pricing page for current plan limits.
