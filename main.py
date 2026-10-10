# -*- coding: utf-8 -*-
"""
Сервер распознавания картин для Render.
Использует CLIP + Википедию.
"""

import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import requests
import urllib.parse

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# ЗАГРУЗКА МОДЕЛИ CLIP

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

print(f"CLIP загружен на {device}")

# ЗАГРУЗКА ЭМБЕДДИНГОВ

reference_embeddings = np.load("embeddings.npy")

with open("paths.json", "r", encoding="utf-8") as f:
    ref_paths = json.load(f)

print(f"Эмбеддингов: {reference_embeddings.shape}")
print(f"Имён картин: {len(ref_paths)}")

# РАСПОЗНАВАНИЕ

def recognize_painting(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            query_embedding = model.encode_image(image)
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.cpu().numpy().flatten()

        similarities = cosine_similarity([query_embedding], reference_embeddings)[0]
        best_idx = np.argmax(similarities)

        return ref_paths[best_idx], similarities[best_idx]
    except Exception as e:
        print(f"Ошибка: {e}")
        return None, 0.0


# ВИКИПЕДИЯ

def search_wikipedia(query, lang="ru"):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("query", {}).get("search"):
                return data["query"]["search"][0]["title"]
    except:
        pass
    return None

def get_wikipedia_content(title, lang="ru"):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {"action": "query", "prop": "extracts", "exintro": True, "explaintext": True, "titles": title, "format": "json"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id != "-1":
                    return page.get("extract", "")
    except:
        pass
    return None

def get_russian_version(english_title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "titles": english_title, "prop": "langlinks", "lllang": "ru", "format": "json"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if "langlinks" in page and len(page["langlinks"]) > 0:
                    return page["langlinks"][0]["*"]
    except:
        pass
    return None

def format_wiki_result(title, content, source_type, lang="ru"):
    sentences = content.split(".")
    summary = ". ".join(sentences[:5]) + "."
    if len(summary) > 500:
        summary = summary[:500]
        last_dot = summary.rfind(".")
        if last_dot > 0:
            summary = summary[:last_dot + 1]
    encoded_title = urllib.parse.quote(title.replace(" ", "_"))
    return {
        "source": f"Wikipedia {lang} ({source_type})",
        "title": title,
        "summary": summary,
        "url": f"https://{lang}.wikipedia.org/wiki/{encoded_title}"
    }

def get_facts_about_painting(pred_filename):
    clean_name = pred_filename.replace(".jpg", "").replace("_", " ")
    if " - " in clean_name:
        artist_name = clean_name.split(" - ")[0]
        painting_name = clean_name.split(" - ")[1]
    else:
        artist_name = clean_name
        painting_name = clean_name

    en_queries = [f"{painting_name} {artist_name}", f"{painting_name} painting {artist_name}"]
    for query in en_queries:
        title = search_wikipedia(query, "en")
        if title:
            content = get_wikipedia_content(title, "en")
            if content:
                keywords = ["painting", "oil on", "canvas", "artwork"]
                if any(kw in content.lower() for kw in keywords):
                    ru_title = get_russian_version(title)
                    if ru_title:
                        ru_content = get_wikipedia_content(ru_title, "ru")
                        if ru_content:
                            return format_wiki_result(ru_title, ru_content, "картина", "ru")
                    return format_wiki_result(title, content, "картина", "en")

    ru_queries = [f"{artist_name} художник", artist_name]
    for query in ru_queries:
        title = search_wikipedia(query, "ru")
        if title:
            content = get_wikipedia_content(title, "ru")
            if content:
                keywords = ["художник", "живопис", "живописец"]
                if any(kw in content.lower() for kw in keywords):
                    return format_wiki_result(title, content, "художник", "ru")

    return {"source": None, "message": "Информация не найдена"}

# СОЗДАНИЕ API

app = FastAPI(title="Art Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "API работает"}

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        temp_path = "/tmp/photo.jpg"
        with open(temp_path, "wb") as f:
            f.write(contents)

        pred_path, confidence = recognize_painting(temp_path)
        if pred_path is None:
            return JSONResponse(status_code=404, content={"recognized": False, "message": "Не распознано"})

        pred_name = os.path.basename(pred_path)
        facts = get_facts_about_painting(pred_name)

        return {
            "recognized": True,
            "filename": pred_name,
            "confidence": float(confidence),
            "wikipedia": facts
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"recognized": False, "error": str(e)})