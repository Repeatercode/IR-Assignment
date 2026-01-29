import argparse
import time
import re
from collections import deque
from urllib.parse import urlparse, urlunparse
import urllib.robotparser as robotparser

import requests

from .config import CrawlConfig, PUBLICATIONS_JSONL, INDEX_JSON
from .storage import append_jsonl, load_jsonl
from .parser import extract_links, parse_publication_page, parse_list_page_for_publications
from .indexer import build_documents, build_inverted_index, save_index

PUB_RE = re.compile(r"/en/publications/")
ORG_SLUG = "/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo"

def same_domain(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    cleaned = parsed._replace(query="", fragment="")
    return urlunparse(cleaned).rstrip("/")

def is_org_url(url: str) -> bool:
    return ORG_SLUG in url

class PoliteCrawler:
    def __init__(self, seed_url: str, cfg: CrawlConfig):
        self.seed_url = seed_url
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent})
        self.delay_seconds = cfg.delay_seconds
        self.ics_person_urls = set()
        self.org_publication_urls = set()

        self.robots = robotparser.RobotFileParser()
        robots_url = f"{urlparse(seed_url).scheme}://{urlparse(seed_url).netloc}/robots.txt"
        self.robots.set_url(robots_url)
        try:
            resp = self.session.get(robots_url, timeout=15)
            if resp.status_code == 200:
                self.robots.parse(resp.text.splitlines())
                self.robots_ok = True
                crawl_delay = self.robots.crawl_delay(cfg.user_agent) or self.robots.crawl_delay("*")
                if crawl_delay:
                    self.delay_seconds = max(self.delay_seconds, float(crawl_delay))
            else:
                self.robots_ok = False
        except Exception:
            self.robots_ok = False

    def allowed(self, url: str) -> bool:
        try:
            return self.robots.can_fetch(self.cfg.user_agent, url) if self.robots_ok else True
        except Exception:
            return False

    def fetch(self, url: str) -> str:
        time.sleep(self.delay_seconds)
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        return r.text

    def crawl_bfs(self):
        queue = deque([self.seed_url])
        visited = set()
        publications = []

        while queue and (self.cfg.max_pages == 0 or len(visited) < self.cfg.max_pages):
            url = queue.popleft()
            norm_url = normalize_url(url)
            if norm_url in visited:
                continue
            visited.add(norm_url)

            if self.cfg.same_domain_only and not same_domain(self.seed_url, url):
                continue

            if not self.allowed(url):
                continue

            try:
                html = self.fetch(url)
            except Exception:
                continue

            is_org = is_org_url(url)
            links = extract_links(url, html)

            if is_org:
                for link in links:
                    nlink = normalize_url(link)
                    if "/en/persons/" in nlink and not nlink.endswith("/en/persons"):
                        self.ics_person_urls.add(nlink)

            # Extract publication links from list pages
            for lp in parse_list_page_for_publications(url, html):
                pu = lp.get("publication_url")
                if pu:
                    npu = normalize_url(pu)
                    if is_org:
                        self.org_publication_urls.add(npu)
                    if npu not in visited:
                        queue.append(pu)

            # Extract publication data if it is a publication page
            if PUB_RE.search(norm_url):
                pub = parse_publication_page(url, html)
                pub["source_url"] = url
                publications.append(pub)

            # Add more internal links for BFS
            for link in links:
                if self.cfg.same_domain_only and not same_domain(self.seed_url, link):
                    continue
                if ("/en/organisations/" in link) or ("/en/publications/" in link) or ("/en/persons/" in link):
                    if normalize_url(link) not in visited:
                        queue.append(link)

        return publications

def filter_publications_by_membership(publications, ics_person_urls, org_publication_urls):
    if not publications:
        return []
    if not ics_person_urls and not org_publication_urls:
        return publications
    filtered = []
    for pub in publications:
        pub_url = normalize_url(pub.get("publication_url") or "")
        if pub_url and pub_url in org_publication_urls:
            filtered.append(pub)
            continue
        author_urls = [normalize_url(u) for u in pub.get("author_urls", [])]
        if ics_person_urls and any(u in ics_person_urls for u in author_urls):
            filtered.append(pub)
    return filtered

def merge_by_url(old, new):
    by_url = {p.get("publication_url"): p for p in old if p.get("publication_url")}
    for p in new:
        u = p.get("publication_url")
        if u:
            by_url[u] = {**by_url.get(u, {}), **p}
    return list(by_url.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--max-pages", type=int, default=CrawlConfig.max_pages, help="0 = no limit")
    ap.add_argument("--delay", type=float, default=CrawlConfig.delay_seconds)
    ap.add_argument("--user-agent", default=CrawlConfig.user_agent)
    args = ap.parse_args()

    cfg = CrawlConfig(user_agent=args.user_agent, delay_seconds=args.delay, max_pages=args.max_pages)
    crawler = PoliteCrawler(args.seed, cfg)
    new_pubs = crawler.crawl_bfs()
    new_pubs = filter_publications_by_membership(
        new_pubs, crawler.ics_person_urls, crawler.org_publication_urls
    )

    old = load_jsonl(PUBLICATIONS_JSONL)
    merged = merge_by_url(old, new_pubs)
    merged = filter_publications_by_membership(
        merged, crawler.ics_person_urls, crawler.org_publication_urls
    )

    append_jsonl(PUBLICATIONS_JSONL, merged)

    docs = build_documents(merged)
    index, doc_lengths = build_inverted_index(docs)
    save_index(INDEX_JSON, docs, index, doc_lengths)

    print("Crawl finished.")
    print(f"Publications stored: {len(merged)}")
    print(f"Saved: {PUBLICATIONS_JSONL}")
    print(f"Saved: {INDEX_JSON}")

if __name__ == "__main__":
    main()
