# AutoAssist Docker default-ML fix

Replace **only** `backend/Dockerfile.unified-v12` in your existing working V12 project. This patch does not change Python code, the active LangChain agent, or storage.

- Installs CPU PyTorch, Transformers, `faiss-cpu`, and `sentence-transformers` on every image build, without needing `INSTALL_DISTILBERT` or `INSTALL_FAISS` build args.
- Defaults to `MODEL_VARIANT=multilabel_distilbert_v3` and `RAG_BACKEND=faiss` (Render environment variables, if set, take precedence).
- Avoids a Docker `COPY ml/artifacts` error by copying `ml/` and creating an empty artifacts directory when missing.
- **Does not bundle model weights automatically.** Make sure `ml/artifacts/multilabel_distilbert_v3/` is available in the remote build context, or implement model downloading. If the directory is absent, the Docker build prints a warning; classification remains unavailable. Do not commit large/private artifacts without a suitable storage plan.

From project root:

```bash
# optional safety backup
cp backend/Dockerfile.unified-v12 backend/Dockerfile.unified-v12.bak
unzip -o /path/to/autoassist-docker-default-ml-fix.zip -d .
sudo docker build --platform linux/amd64 -f backend/Dockerfile.unified-v12 -t autoassist-backend:render-test .
sudo docker run --rm --entrypoint python autoassist-backend:render-test -c "import torch, transformers, faiss, sentence_transformers; print('ML imports OK')"
sudo docker run --rm --entrypoint sh autoassist-backend:render-test -c 'find /app/ml/artifacts -maxdepth 2 -type f | head -20'
```

If your Render variables include `RAG_BACKEND=tfidf`, change it to `faiss` to use FAISS. `INSTALL_DISTILBERT` and `INSTALL_FAISS` can be removed from Render since they are not used by this Dockerfile. Build from the **repository root** with the Dockerfile path `backend/Dockerfile.unified-v12`.

Do not treat successful imports or a 200 health check as proof the trained DistilBERT weights or retrieval corpus are present. A full end-to-end complaint/recall test is still needed. Image size and CPU/RAM use will increase substantially.
