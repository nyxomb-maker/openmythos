"""
Web Scraper for Training Data
==============================
Fetches and extracts clean text from websites for model training.

Features:
- URL list or seed-based crawling
- HTML to clean text extraction
- Rate limiting (respectful crawling)
- Deduplication
- Robots.txt compliance
- Saves extracted text to data directory
- Configurable depth and max pages

Dependencies:
    pip install requests beautifulsoup4
"""

import os
import re
import time
import json
import hashlib
import logging
from typing import List, Set, Optional, Dict, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup, Comment
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScraperConfig:
    """Configuration for web scraping."""

    # --- URLs ---
    seed_urls: List[str] = field(default_factory=list)
    url_file: Optional[str] = None          # File with one URL per line

    # --- Crawling ---
    max_pages: int = 100                    # Maximum pages to scrape
    max_depth: int = 2                      # Maximum link-following depth
    stay_on_domain: bool = True             # Only follow links on same domain
    follow_links: bool = True               # Enable crawling (vs. just seed URLs)

    # --- Rate Limiting ---
    delay_seconds: float = 1.0              # Delay between requests
    timeout_seconds: int = 15               # Request timeout
    max_retries: int = 2                    # Retries per URL

    # --- Content Filtering ---
    min_text_length: int = 200              # Skip pages with less text
    max_text_length: int = 500000           # Skip extremely large pages
    allowed_languages: List[str] = field(
        default_factory=lambda: ['en', 'pt', 'es']
    )

    # --- Output ---
    output_dir: str = "../data/scraped"     # Where to save scraped text
    save_metadata: bool = True              # Save URL/timestamp metadata

    # --- User Agent ---
    user_agent: str = (
        "Mozilla/5.0 (compatible; LLM-MoE-Trainer/1.0; "
        "+https://github.com/llm-moe)"
    )

    # --- Filtering ---
    exclude_patterns: List[str] = field(default_factory=lambda: [
        r'/login', r'/signup', r'/register', r'/cart', r'/checkout',
        r'/privacy', r'/terms', r'/cookie', r'\.(pdf|zip|tar|gz|exe|dmg|mp4|mp3|jpg|png|gif|svg)$',
        r'/wp-admin', r'/feed', r'/rss',
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Text Extraction
# ═══════════════════════════════════════════════════════════════════════════

# HTML elements to remove entirely (not just their text)
REMOVE_TAGS = {
    'script', 'style', 'nav', 'header', 'footer', 'aside',
    'form', 'button', 'iframe', 'noscript', 'svg', 'canvas',
    'video', 'audio', 'figure', 'figcaption',
}

# Elements whose text should be preceded by a newline
BLOCK_TAGS = {
    'p', 'div', 'section', 'article', 'main',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'li', 'tr', 'td', 'th', 'blockquote',
    'pre', 'code', 'br', 'hr',
}


def extract_text_from_html(html: str, url: str = "") -> str:
    """
    Extract clean, readable text from HTML.
    Removes navigation, scripts, styles, and boilerplate.
    Preserves paragraph structure and code blocks.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Remove unwanted elements
    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove elements with common boilerplate classes/IDs
    boilerplate_patterns = [
        'cookie', 'popup', 'modal', 'newsletter', 'subscribe',
        'social', 'share', 'sidebar', 'widget', 'ad-', 'advert',
        'banner', 'promo', 'related-posts', 'comment',
    ]

    for element in soup.find_all(True):
        classes = ' '.join(element.get('class', []))
        element_id = element.get('id', '')
        combined = f"{classes} {element_id}".lower()

        for pattern in boilerplate_patterns:
            if pattern in combined:
                element.decompose()
                break

    # Extract text with structure
    lines = []
    _extract_recursive(soup, lines)

    # Clean up
    text = '\n'.join(lines)

    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def _extract_recursive(element, lines: list):
    """Recursively extract text preserving structure."""
    if isinstance(element, str):
        text = element.strip()
        if text:
            lines.append(text)
        return

    if not hasattr(element, 'children'):
        return

    tag_name = getattr(element, 'name', None)

    # Add newline before block elements
    if tag_name in BLOCK_TAGS:
        lines.append('')

    # Special handling for code blocks
    if tag_name == 'pre' or tag_name == 'code':
        code_text = element.get_text()
        if code_text.strip():
            lines.append(f'\n```\n{code_text.strip()}\n```\n')
        return

    # Special handling for headings
    if tag_name and tag_name.startswith('h') and len(tag_name) == 2:
        level = int(tag_name[1])
        heading_text = element.get_text().strip()
        if heading_text:
            lines.append(f'\n{"#" * level} {heading_text}\n')
        return

    # Special handling for list items
    if tag_name == 'li':
        item_text = element.get_text().strip()
        if item_text:
            lines.append(f'- {item_text}')
        return

    # Recurse into children
    for child in element.children:
        _extract_recursive(child, lines)


def extract_title(html: str) -> str:
    """Extract the page title from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text().strip()
    h1_tag = soup.find('h1')
    if h1_tag:
        return h1_tag.get_text().strip()
    return "Untitled"


def extract_links(html: str, base_url: str) -> List[str]:
    """Extract all links from HTML, resolved relative to base_url."""
    soup = BeautifulSoup(html, 'html.parser')
    links = []

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']

        # Skip anchors, javascript, mailto
        if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
            continue

        # Resolve relative URLs
        full_url = urljoin(base_url, href)

        # Remove fragments
        full_url = full_url.split('#')[0]

        if full_url.startswith(('http://', 'https://')):
            links.append(full_url)

    return list(set(links))


# ═══════════════════════════════════════════════════════════════════════════
# Robots.txt Parser (Simple)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleRobotsParser:
    """Minimal robots.txt parser."""

    def __init__(self):
        self._cache: Dict[str, List[str]] = {}

    def can_fetch(self, url: str, session: 'requests.Session') -> bool:
        """Check if URL is allowed by robots.txt."""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        if domain not in self._cache:
            self._load_robots(domain, session)

        disallowed = self._cache.get(domain, [])
        path = parsed.path

        for pattern in disallowed:
            if path.startswith(pattern):
                return False

        return True

    def _load_robots(self, domain: str, session: 'requests.Session'):
        """Load and parse robots.txt for a domain."""
        try:
            resp = session.get(
                f"{domain}/robots.txt",
                timeout=5,
                allow_redirects=True,
            )
            if resp.status_code == 200:
                self._cache[domain] = self._parse(resp.text)
            else:
                self._cache[domain] = []  # No restrictions
        except Exception:
            self._cache[domain] = []  # Fail open

    def _parse(self, content: str) -> List[str]:
        """Parse robots.txt and return disallowed paths for * user-agent."""
        disallowed = []
        applies_to_us = False

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            if ':' not in line:
                continue

            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'user-agent':
                applies_to_us = value == '*'
            elif key == 'disallow' and applies_to_us and value:
                disallowed.append(value)

        return disallowed


# ═══════════════════════════════════════════════════════════════════════════
# Web Scraper
# ═══════════════════════════════════════════════════════════════════════════

class WebScraper:
    """
    Web scraper that fetches pages, extracts clean text, and saves
    the results for model training.
    """

    def __init__(self, config: ScraperConfig):
        if not HAS_DEPS:
            raise ImportError(
                "Web scraping requires 'requests' and 'beautifulsoup4'.\n"
                "Install them with: pip install requests beautifulsoup4"
            )

        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8',
        })

        self.robots = SimpleRobotsParser()
        self.visited: Set[str] = set()
        self.content_hashes: Set[str] = set()  # Deduplication
        self.results: List[Dict] = []

        # Compile exclude patterns
        self.exclude_re = [
            re.compile(p, re.IGNORECASE)
            for p in config.exclude_patterns
        ]

    def _should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped based on filters."""
        if url in self.visited:
            return True

        for pattern in self.exclude_re:
            if pattern.search(url):
                return True

        return False

    def _is_same_domain(self, url: str, seed_url: str) -> bool:
        """Check if URL is on the same domain as the seed."""
        return urlparse(url).netloc == urlparse(seed_url).netloc

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a single page with retries."""
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.session.get(
                    url,
                    timeout=self.config.timeout_seconds,
                    allow_redirects=True,
                )

                # Check content type
                content_type = resp.headers.get('Content-Type', '')
                if 'text/html' not in content_type and 'text/plain' not in content_type:
                    logger.debug(f"Skipping non-HTML: {url} ({content_type})")
                    return None

                resp.raise_for_status()
                return resp.text

            except requests.RequestException as e:
                if attempt < self.config.max_retries:
                    wait = (attempt + 1) * 2
                    logger.warning(f"Retry {attempt+1} for {url}: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Failed to fetch {url}: {e}")
                    return None

        return None

    def process_page(self, url: str, html: str) -> Optional[Dict]:
        """Extract text from HTML and return metadata."""
        text = extract_text_from_html(html, url)

        # Length check
        if len(text) < self.config.min_text_length:
            logger.debug(f"Skipping short page ({len(text)} chars): {url}")
            return None

        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
            logger.info(f"Truncated long page: {url}")

        # Deduplication
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if content_hash in self.content_hashes:
            logger.debug(f"Duplicate content: {url}")
            return None
        self.content_hashes.add(content_hash)

        title = extract_title(html)

        return {
            'url': url,
            'title': title,
            'text': text,
            'length': len(text),
            'hash': content_hash,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        }

    def scrape(self) -> List[Dict]:
        """
        Main scraping loop. Fetches pages, extracts text, follows links.
        Returns list of page data dictionaries.
        """
        # Collect seed URLs
        urls_to_visit: List[Tuple[str, int, str]] = []  # (url, depth, seed_domain)

        # From config
        for url in self.config.seed_urls:
            urls_to_visit.append((url, 0, urlparse(url).netloc))

        # From file
        if self.config.url_file:
            url_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.config.url_file,
            )
            if os.path.exists(url_file):
                with open(url_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            urls_to_visit.append((line, 0, urlparse(line).netloc))
                logger.info(f"Loaded URLs from {url_file}")

        if not urls_to_visit:
            logger.warning("No URLs to scrape!")
            return []

        logger.info(f"Starting scrape: {len(urls_to_visit)} seed URLs, "
                     f"max_pages={self.config.max_pages}, max_depth={self.config.max_depth}")

        page_count = 0

        while urls_to_visit and page_count < self.config.max_pages:
            url, depth, seed_domain = urls_to_visit.pop(0)

            # Normalize URL
            url = url.rstrip('/')

            if self._should_skip_url(url):
                continue

            self.visited.add(url)

            # Robots.txt check
            if not self.robots.can_fetch(url, self.session):
                logger.info(f"Blocked by robots.txt: {url}")
                continue

            # Rate limiting
            time.sleep(self.config.delay_seconds)

            # Fetch
            logger.info(f"[{page_count+1}/{self.config.max_pages}] "
                        f"Fetching (depth={depth}): {url}")
            html = self.fetch_page(url)
            if html is None:
                continue

            # Process
            result = self.process_page(url, html)
            if result is None:
                continue

            self.results.append(result)
            page_count += 1

            logger.info(f"  -> {result['title'][:60]} ({result['length']:,} chars)")

            # Follow links
            if (self.config.follow_links and
                    depth < self.config.max_depth and
                    page_count < self.config.max_pages):

                links = extract_links(html, url)
                for link in links:
                    if link not in self.visited and not self._should_skip_url(link):
                        if self.config.stay_on_domain:
                            if self._is_same_domain(link, f"https://{seed_domain}"):
                                urls_to_visit.append((link, depth + 1, seed_domain))
                        else:
                            urls_to_visit.append((link, depth + 1, seed_domain))

        logger.info(f"\nScraping complete: {page_count} pages, "
                     f"{sum(r['length'] for r in self.results):,} total characters")

        return self.results

    def save(self, output_dir: Optional[str] = None):
        """Save scraped text to the output directory."""
        out_dir = output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.config.output_dir,
        )
        os.makedirs(out_dir, exist_ok=True)

        if not self.results:
            logger.warning("No results to save")
            return

        # Save all text as a single concatenated file
        combined_path = os.path.join(out_dir, "scraped_corpus.txt")
        with open(combined_path, 'w', encoding='utf-8') as f:
            for result in self.results:
                f.write(f"\n\n{'='*60}\n")
                f.write(f"Source: {result['url']}\n")
                f.write(f"Title: {result['title']}\n")
                f.write(f"{'='*60}\n\n")
                f.write(result['text'])
                f.write('\n')

        logger.info(f"Saved combined corpus: {combined_path}")

        # Save individual files
        for i, result in enumerate(self.results):
            # Create filename from URL
            parsed = urlparse(result['url'])
            slug = re.sub(r'[^a-zA-Z0-9]', '_', parsed.path.strip('/'))
            if not slug:
                slug = 'index'
            slug = slug[:80]  # Limit filename length

            fname = f"{parsed.netloc}_{slug}.txt"
            fpath = os.path.join(out_dir, fname)

            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(f"Source: {result['url']}\n")
                f.write(f"Title: {result['title']}\n")
                f.write(f"Date: {result['timestamp']}\n")
                f.write(f"{'─'*40}\n\n")
                f.write(result['text'])

        # Save metadata
        if self.config.save_metadata:
            meta_path = os.path.join(out_dir, "scrape_metadata.json")
            metadata = [{
                'url': r['url'],
                'title': r['title'],
                'length': r['length'],
                'hash': r['hash'],
                'timestamp': r['timestamp'],
            } for r in self.results]

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved metadata: {meta_path}")

        logger.info(f"Saved {len(self.results)} files to {out_dir}")
        return out_dir


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """CLI for standalone scraping."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape websites for LLM training data"
    )
    parser.add_argument(
        'urls', nargs='*',
        help='URLs to scrape'
    )
    parser.add_argument(
        '--url-file', type=str, default=None,
        help='File with URLs (one per line)'
    )
    parser.add_argument(
        '--max-pages', type=int, default=50,
        help='Maximum number of pages to scrape (default: 50)'
    )
    parser.add_argument(
        '--max-depth', type=int, default=2,
        help='Maximum crawl depth (default: 2)'
    )
    parser.add_argument(
        '--no-follow', action='store_true',
        help='Do not follow links (only scrape given URLs)'
    )
    parser.add_argument(
        '--delay', type=float, default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--output', type=str, default='../data/scraped',
        help='Output directory (default: ../data/scraped)'
    )
    parser.add_argument(
        '--any-domain', action='store_true',
        help='Follow links to other domains'
    )

    args = parser.parse_args()

    config = ScraperConfig(
        seed_urls=args.urls,
        url_file=args.url_file,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        follow_links=not args.no_follow,
        delay_seconds=args.delay,
        output_dir=args.output,
        stay_on_domain=not args.any_domain,
    )

    scraper = WebScraper(config)
    scraper.scrape()
    scraper.save()


if __name__ == "__main__":
    main()
