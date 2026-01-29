import re
import json
from typing import Dict, List, Iterable, Any
from bs4 import BeautifulSoup
from urllib.parse import urljoin

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def absolute_url(base: str, href: str) -> str:
    return urljoin(base, href)

def is_person_profile(url: str) -> bool:
    if "/en/persons/" not in url:
        return False
    return not url.rstrip("/").endswith("/en/persons")

def extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: List[str] = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        urls.append(absolute_url(base_url, href))
    return urls

def _txt(el) -> str:
    return el.get_text(" ", strip=True) if el else ""

def _meta_content(soup: BeautifulSoup, key: str) -> str:
    tag = soup.find("meta", attrs={"name": key}) or soup.find("meta", attrs={"property": key})
    if not tag:
        return ""
    return (tag.get("content") or "").strip()

def _add_author(name: str, authors: List[str]) -> None:
    if not name:
        return
    cleaned = re.sub(r"\s+", " ", name).strip()
    if cleaned and cleaned not in authors:
        authors.append(cleaned)

def _split_author_names(value: str) -> Iterable[str]:
    if not value:
        return []
    parts = re.split(r"\s*;\s*|\s+and\s+", value)
    return [p.strip() for p in parts if p.strip()]

def _extract_author_names_from_jsonld(obj: Any, names: List[str]) -> None:
    if isinstance(obj, dict):
        for key in ("author", "creator", "contributor", "contributors"):
            if key in obj:
                _extract_author_names_from_jsonld(obj[key], names)
        name = obj.get("name")
        if isinstance(name, str):
            names.append(name)
        return
    if isinstance(obj, list):
        for item in obj:
            _extract_author_names_from_jsonld(item, names)
        return
    if isinstance(obj, str):
        names.append(obj)

def parse_publication_page(url: str, html: str) -> Dict:
    soup = BeautifulSoup(html, "lxml")

    title = _txt(soup.find("h1")) or _meta_content(soup, "citation_title") or _meta_content(soup, "og:title") or _txt(soup.find("title"))

    page_text = soup.get_text(" ", strip=True)
    year = ""
    m = YEAR_RE.search(page_text)
    if m:
        year = m.group(0)
    else:
        meta_date = _meta_content(soup, "citation_publication_date") or _meta_content(soup, "citation_date")
        m2 = YEAR_RE.search(meta_date)
        if m2:
            year = m2.group(0)

    authors: List[str] = []
    author_urls = []
    author_profiles = []
    for a in soup.select('a[href*="/en/persons/"]'):
        name = _txt(a)
        href = (a.get("href") or "").strip()
        if href:
            au = absolute_url(url, href)
            if is_person_profile(au):
                _add_author(name, authors)
                if au not in author_urls:
                    author_urls.append(au)
                author_profiles.append({"name": name or "Profile", "url": au})

    for tag in soup.find_all("meta", attrs={"name": "citation_author"}):
        for name in _split_author_names((tag.get("content") or "").strip()):
            _add_author(name, authors)

    for tag in soup.find_all("meta", attrs={"name": "dc.creator"}):
        for name in _split_author_names((tag.get("content") or "").strip()):
            _add_author(name, authors)

    for tag in soup.find_all("meta", attrs={"name": "author"}):
        for name in _split_author_names((tag.get("content") or "").strip()):
            _add_author(name, authors)

    for el in soup.select('[itemprop="author"], [itemprop="creator"]'):
        if el.has_attr("content"):
            for name in _split_author_names((el.get("content") or "").strip()):
                _add_author(name, authors)
            continue
        name_el = el.find(attrs={"itemprop": "name"})
        if name_el:
            for name in _split_author_names(_txt(name_el)):
                _add_author(name, authors)
            continue
        for name in _split_author_names(_txt(el)):
            _add_author(name, authors)

    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.get_text(strip=True))
        except Exception:
            continue
        jsonld_names: List[str] = []
        _extract_author_names_from_jsonld(data, jsonld_names)
        for name in jsonld_names:
            for part in _split_author_names(name):
                _add_author(part, authors)

    abstract = ""
    h = soup.find(lambda tag: tag.name in ("h2","h3","strong") and "abstract" in tag.get_text(" ", strip=True).lower())
    if h:
        nxt = h.find_next(["p","div"])
        abstract = _txt(nxt)
    if not abstract:
        abstract = _meta_content(soup, "citation_abstract") or _meta_content(soup, "description")

    return {
        "publication_url": url,
        "title": title,
        "year": year,
        "authors": authors,
        "author_urls": author_urls,
        "author_profiles": author_profiles,
        "abstract": abstract,
    }

def parse_list_page_for_publications(base_url: str, html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    pubs = []
    seen = set()

    for a in soup.select('a[href*="/en/publications/"]'):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        absu = absolute_url(base_url, href)
        if "/en/publications/" not in absu or absu in seen:
            continue

        title = _txt(a)
        if not title:
            title = (a.get("title") or "").strip()
        if not title:
            title = (a.get("aria-label") or "").strip()

        if len(title) < 4:
            continue

        pubs.append({"title": title, "publication_url": absu})
        seen.add(absu)

    return pubs
