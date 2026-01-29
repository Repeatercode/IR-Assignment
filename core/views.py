from pathlib import Path

from django.conf import settings
from django.shortcuts import render

from search_engine.search import search as run_search
from search_engine.storage import load_json
from classifier.predict import predict_label, load_model

INDEX_PATH = settings.BASE_DIR / "data" / "index.json"


def load_index():
    return load_json(str(INDEX_PATH))


def home(request):
    return render(request, "index.html")


def search(request):
    q = (request.GET.get("q") or "").strip()
    use_stemming = request.GET.get("stem") == "1"
    payload = load_index()
    results = []

    if q and payload:
        results = run_search(q, payload, top_k=15, use_stemming=use_stemming)
    elif payload:
        docs = list(payload.get("docs", {}).values())
        def sort_key(d):
            year = d.get("year") or ""
            try:
                y = int(year)
            except ValueError:
                y = 0
            title = (d.get("title") or "").lower()
            return (-y, title)
        docs.sort(key=sort_key)
        results = [{**d, "score": None} for d in docs]

    context = {
        "q": q,
        "results": results,
        "use_stemming": use_stemming,
        "has_index": bool(payload),
        "doc_count": len(results),
    }
    return render(request, "results.html", context)


def classify(request):
    text = ""
    label = None
    model_ready = load_model() is not None

    if request.method == "POST":
        text = (request.POST.get("text") or "").strip()
        if text and model_ready:
            label = predict_label(text)

    context = {
        "text": text,
        "label": label,
        "model_ready": model_ready,
    }
    return render(request, "classification.html", context)
