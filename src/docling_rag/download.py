"""Download papers from Google Scholar author pages."""

import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from curl_cffi import requests as cf_requests

# Tor SOCKS5 proxy for Google Scholar
TOR_PROXY = {
    "http": "socks5://127.0.0.1:9050",
    "https": "socks5://127.0.0.1:9050",
}


def _check_tor_available() -> bool:
    """Check if Tor is running on port 9050."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", 9050))
        sock.close()
        return result == 0
    except Exception:
        return False


def _fetch_scholar_page(url: str, timeout: int = 30) -> str | None:
    """
    Fetch a Google Scholar page using curl_cffi with Chrome impersonation.

    Uses Tor if available, otherwise direct connection.
    """
    try:
        proxies = TOR_PROXY if _check_tor_available() else None
        response = cf_requests.get(
            url,
            impersonate="chrome",
            proxies=proxies,
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.text
        return None
    except Exception:
        return None


def extract_author_id(url: str) -> str | None:
    """Extract Google Scholar author ID from URL."""
    parsed = urlparse(url)
    if "scholar.google" in parsed.netloc:
        params = parse_qs(parsed.query)
        if "user" in params:
            return params["user"][0]
    return None


def sanitize_filename(title: str, max_length: int = 80) -> str:
    """Convert paper title to safe filename."""
    safe = re.sub(r'[<>:"/\\|?*]', "", title)
    safe = re.sub(r"\s+", " ", safe).strip()
    if len(safe) > max_length:
        safe = safe[:max_length].rsplit(" ", 1)[0]
    return safe


def _parse_author_page(html: str) -> dict:
    """Parse Google Scholar author page HTML to extract author info and publications."""
    result = {"name": "Unknown", "publications": []}

    # Extract author name
    name_match = re.search(r'<div id="gsc_prf_in"[^>]*>([^<]+)</div>', html)
    if name_match:
        result["name"] = name_match.group(1).strip()

    # Extract publications from the table
    # Each publication is in a row with class "gsc_a_tr"
    # Format: <a href="/citations?..." class="gsc_a_at">Title</a>
    pub_pattern = re.compile(
        r'<tr class="gsc_a_tr"[^>]*>.*?'
        r'<a href="([^"]*)"[^>]*class="gsc_a_at"[^>]*>([^<]+)</a>.*?'
        r'<span class="gsc_a_h gsc_a_hc gs_ibl">(\d*)</span>',
        re.DOTALL,
    )

    for match in pub_pattern.finditer(html):
        citation_url, title, year = match.groups()
        # Unescape HTML entities in URL
        citation_url = citation_url.replace("&amp;", "&")
        pub = {
            "title": title.strip(),
            "year": year.strip() if year else "",
            "citation_url": citation_url,
        }
        result["publications"].append(pub)

    return result


def _parse_publication_page(html: str) -> dict:
    """Parse a Google Scholar publication page to extract details."""
    result = {"doi": None, "eprint_url": None, "pub_url": None}

    # Extract links from the publication page
    # Look for PDF/eprint links (gsc_oci_title_ggi contains PDF links)
    eprint_match = re.search(
        r'gsc_oci_title_ggi">\s*<a[^>]+href="([^"]+)"',
        html,
    )
    if eprint_match:
        url = eprint_match.group(1).replace("&amp;", "&")
        result["eprint_url"] = url

    # Look for publisher URL
    pub_url_match = re.search(r'<a class="gsc_oci_title_link"[^>]+href="([^"]+)"', html)
    if pub_url_match:
        url = pub_url_match.group(1).replace("&amp;", "&")
        result["pub_url"] = url

    # Extract DOI from the page (look for DOI in various formats)
    # Format 1: doi.org URL
    doi_match = re.search(r"doi\.org/(10\.\d{4,}/[^\s<\"'&]+)", html)
    if doi_match:
        result["doi"] = doi_match.group(1).rstrip(".")
    else:
        # Format 2: Plain DOI
        doi_match = re.search(r"(10\.\d{4,}/[^\s<\"'&]+)", html)
        if doi_match:
            result["doi"] = doi_match.group(1).rstrip(".")

    return result


def _get_all_publications(author_id: str, verbose: bool = True) -> list[dict]:
    """Fetch all publications for an author, handling pagination."""
    publications = []
    start = 0
    page_size = 100

    while True:
        params = {
            "user": author_id,
            "hl": "en",
            "cstart": start,
            "pagesize": page_size,
        }
        url = f"https://scholar.google.com/citations?{urlencode(params)}"

        if verbose:
            print(f"  Fetching publications {start + 1}-{start + page_size}...")

        html = _fetch_scholar_page(url)
        if not html:
            break

        parsed = _parse_author_page(html)
        new_pubs = parsed["publications"]

        if not new_pubs:
            break

        publications.extend(new_pubs)

        # Check if there are more pages
        if len(new_pubs) < page_size:
            break

        start += page_size
        time.sleep(1)  # Be nice to Google

    return publications


def try_get_pdf_url(pub: dict) -> str | None:
    """Try to find a PDF URL for a publication."""
    if pub.get("eprint_url"):
        return pub["eprint_url"]

    pub_url = pub.get("pub_url", "")
    if pub_url:
        if pub_url.endswith(".pdf"):
            return pub_url
        if "arxiv.org/abs/" in pub_url:
            return pub_url.replace("/abs/", "/pdf/") + ".pdf"

    return None


def download_pdf(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a PDF from URL to output path."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not url.endswith(".pdf"):
            first_bytes = next(response.iter_content(chunk_size=8), b"")
            if not first_bytes.startswith(b"%PDF"):
                return False
            response = requests.get(url, headers=headers, timeout=timeout)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    except Exception:
        return False


SCIHUB_MIRRORS = [
    "https://sci-hub.st",
    "https://sci-hub.ru",
    "https://sci-hub.ee",
    "https://sci-hub.wf",
]


def download_from_scihub(doi: str | None, output_path: Path, pub_url: str | None = None) -> bool:
    """Try to download a paper from Sci-Hub using DOI or publisher URL."""
    # Try DOI first, then publisher URL
    queries = []
    if doi:
        queries.append(doi)
    if pub_url:
        queries.append(pub_url)

    if not queries:
        return False

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for query in queries:
        for mirror in SCIHUB_MIRRORS:
            try:
                url = f"{mirror}/{query}"
                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code != 200:
                    continue

                html = response.text
                pdf_url = None

                # Pattern 1: iframe src
                match = re.search(r'<iframe[^>]+src\s*=\s*["\']([^"\']+)["\']', html)
                if match:
                    pdf_url = match.group(1)
                    if "#" in pdf_url:
                        pdf_url = pdf_url.split("#")[0]

                # Pattern 2: embed src
                if not pdf_url:
                    match = re.search(r'<embed[^>]+src\s*=\s*["\']([^"\']+\.pdf[^"\']*)["\']', html)
                    if match:
                        pdf_url = match.group(1)

                # Pattern 3: button/link onclick
                if not pdf_url:
                    match = re.search(r"location\.href\s*=\s*['\"]([^'\"]+\.pdf[^'\"]*)['\"]", html)
                    if match:
                        pdf_url = match.group(1)

                # Pattern 4: direct PDF link
                if not pdf_url:
                    match = re.search(r'href\s*=\s*["\']([^"\']+\.pdf[^"\']*)["\']', html)
                    if match:
                        pdf_url = match.group(1)

                if not pdf_url:
                    continue

                if pdf_url.startswith("//"):
                    pdf_url = "https:" + pdf_url
                elif pdf_url.startswith("/"):
                    pdf_url = mirror + pdf_url

                pdf_response = requests.get(pdf_url, headers=headers, timeout=60)
                if (
                    pdf_response.status_code == 200
                    and len(pdf_response.content) > 10000
                    and pdf_response.content[:4] == b"%PDF"
                ):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(pdf_response.content)
                    return True

            except Exception:
                continue

    return False


def download_author_papers(
    scholar_url: str,
    output_dir: Path = Path("data"),
    max_papers: int | None = None,
    use_scihub: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Download papers from a Google Scholar author page.

    Args:
        scholar_url: Google Scholar author profile URL
        output_dir: Directory to save PDFs
        max_papers: Maximum number of papers to download (None = all)
        use_scihub: Try Sci-Hub for papers without open access PDFs
        verbose: Print progress information

    Returns:
        Statistics dict with counts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract author ID
    author_id = extract_author_id(scholar_url)
    if not author_id:
        print(f"Error: Could not extract author ID from URL: {scholar_url}")
        print("Expected format: https://scholar.google.com/citations?user=AUTHOR_ID")
        return {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0}

    if verbose:
        print(f"Fetching author profile: {author_id}")
        if _check_tor_available():
            print("Using Tor proxy for Google Scholar requests")
        else:
            print("Warning: Tor not available, using direct connection (may get blocked)")

    # Fetch author page to get name
    author_url = f"https://scholar.google.com/citations?user={author_id}&hl=en"
    html = _fetch_scholar_page(author_url)

    if not html:
        print("Error: Could not fetch author page from Google Scholar")
        print()
        print("Solutions:")
        print("  1. Start Tor: brew services start tor")
        print("  2. Wait a few minutes and try again")
        print("  3. Use a VPN")
        return {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0}

    if "unusual traffic" in html.lower() or "captcha" in html.lower():
        print("Error: Google Scholar is blocking requests (captcha)")
        print()
        print("Solutions:")
        print("  1. Start Tor: brew services start tor")
        print("  2. Wait a few minutes and try again")
        return {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0}

    author_info = _parse_author_page(html)
    author_name = author_info["name"]

    if verbose:
        print(f"Author: {author_name}")

    # Get all publications with pagination
    publications = _get_all_publications(author_id, verbose=verbose)

    if verbose:
        print(f"Found {len(publications)} publications")
        if use_scihub:
            print("Sci-Hub fallback: enabled")
        print()

    if max_papers:
        publications = publications[:max_papers]

    stats = {"found": len(publications), "downloaded": 0, "skipped": 0, "failed": 0}

    for i, pub in enumerate(publications, 1):
        title = pub.get("title", "Unknown")
        year = pub.get("year", "")

        if verbose:
            display_title = title[:60] + "..." if len(title) > 60 else title
            print(f"[{i}/{len(publications)}] {display_title}")

        # Check if already downloaded
        safe_title = sanitize_filename(title)
        filename = f"{year}_{safe_title}.pdf" if year else f"{safe_title}.pdf"
        output_path = output_dir / filename

        if output_path.exists():
            if verbose:
                print("  → Already exists, skipping")
            stats["skipped"] += 1
            continue

        # Fetch publication details page to get PDF URL and DOI
        citation_url = pub.get("citation_url", "")
        pub_details = {}

        if citation_url:
            full_url = f"https://scholar.google.com{citation_url}"
            pub_html = _fetch_scholar_page(full_url)
            if pub_html:
                pub_details = _parse_publication_page(pub_html)
            time.sleep(0.5)  # Be nice to Google

        # Try to find and download PDF from open access
        pdf_url = try_get_pdf_url(pub_details)
        downloaded = False

        if pdf_url:
            if verbose:
                display_url = pdf_url[:50] + "..." if len(pdf_url) > 50 else pdf_url
                print(f"  → Trying open access: {display_url}")

            if download_pdf(pdf_url, output_path):
                if verbose:
                    print(f"  → Saved: {filename}")
                stats["downloaded"] += 1
                downloaded = True

        # Try Sci-Hub if open access failed
        if not downloaded and use_scihub:
            doi = pub_details.get("doi")

            # Also try to extract DOI from pub_url
            if not doi and pub_details.get("pub_url"):
                pub_url = pub_details["pub_url"]
                doi_match = re.search(r"(10\.\d{4,}/[^\s&?]+)", pub_url)
                if doi_match:
                    doi = doi_match.group(1)

            pub_url = pub_details.get("pub_url")

            if verbose:
                if doi:
                    display_doi = doi[:30] + "..." if len(doi) > 30 else doi
                    print(f"  → Trying Sci-Hub (DOI: {display_doi})")
                elif pub_url:
                    display_url = pub_url[:40] + "..." if len(pub_url) > 40 else pub_url
                    print(f"  → Trying Sci-Hub (URL: {display_url})")
                else:
                    print("  → No DOI or URL found, skipping Sci-Hub")

            if download_from_scihub(doi, output_path, pub_url=pub_url):
                if verbose:
                    print(f"  → Saved: {filename}")
                stats["downloaded"] += 1
                downloaded = True
            elif verbose:
                print("  → Sci-Hub: not found")

        if not downloaded:
            stats["failed"] += 1

        time.sleep(1)  # Be nice to servers

    if verbose:
        print()
        print(
            f"Done! Downloaded: {stats['downloaded']}, "
            f"Skipped: {stats['skipped']}, "
            f"Not available: {stats['failed']}"
        )

    return stats
