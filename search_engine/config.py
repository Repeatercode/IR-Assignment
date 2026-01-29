from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class CrawlConfig:
    user_agent: str = "ST7071CEM-StudentCrawler/1.0 (+contact: your_email@example.com)"
    delay_seconds: float = 1.2
    max_pages: int = 300
    same_domain_only: bool = True

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PUBLICATIONS_JSONL = str(DATA_DIR / "publications.jsonl")
INDEX_JSON = str(DATA_DIR / "index.json")
