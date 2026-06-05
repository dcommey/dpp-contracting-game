"""Search scholarly APIs and download only legal open-access PDFs.

The script combines seed literature with metadata from OpenAlex, Crossref,
Semantic Scholar, optional CORE, and optional Unpaywall DOI lookups. It writes:

- code/data/literature_metadata.csv
- lit/literature_log.csv
- lit/bib/dpp_contracting_game_literature.bib
- paper/references.bib

PDF downloads are attempted only for open-access PDF URLs returned by OpenAlex
or Unpaywall. Publisher landing pages are recorded as metadata but are not used
as PDF sources unless the API marks the PDF as open access.
"""

from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "code" / "data"
LIT_DIR = PROJECT_ROOT / "lit"
PDF_DIR = LIT_DIR / "pdfs"
BIB_DIR = LIT_DIR / "bib"
PAPER_DIR = PROJECT_ROOT / "paper"

HEADERS = {
    "User-Agent": "dpp-contracting-game-literature-review/0.1 (mailto:research@example.com)"
}

SEARCH_STRINGS = [
    '"digital product passport" "supplier data sharing"',
    '"digital product passport" contract',
    '"digital product passport" governance',
    '"digital product passport" "supply chain"',
    '"digital product passport" verification',
    '"digital product passport" audit',
    '"digital product passport" "data quality"',
    '"circular economy" contracts "supply chain"',
    '"circular economy clauses" contracts',
    '"circular supply chain" "transaction cost economics"',
    '"circular supply chain" "supplier collaboration"',
    '"circular supply chain" "game theory"',
    '"supplier data sharing" "game theory"',
    '"supply chain information sharing" "game theory"',
    '"green supply chain" "game theory" contracts',
    '"blockchain" "circular economy" "supply chain governance"',
]


@dataclass
class LiteratureRecord:
    id: str
    title: str
    authors: str = ""
    year: str = ""
    journal: str = ""
    doi: str = ""
    url: str = ""
    source_database: str = ""
    open_access_pdf_url: str = ""
    open_access_pdf_path: str = ""
    screening_status: str = "included"
    reason_for_inclusion: str = ""
    notes: str = ""
    bibtex_type: str = "article"
    extra: Dict[str, str] = field(default_factory=dict)


SEED_RECORDS: List[LiteratureRecord] = [
    LiteratureRecord(
        id="castrolopez2026contracting",
        title="Contracting for circularity: How circular economy clauses shape market and financial performance",
        authors="Castro-Lopez, Adrian and Iglesias, Victor and Santos-Vijande, Maria Leticia",
        year="2026",
        journal="Journal of Purchasing and Supply Management",
        doi="10.1016/j.pursup.2026.101153",
        source_database="seed",
        reason_for_inclusion="Circular-economy contract clauses and firm performance.",
    ),
    LiteratureRecord(
        id="rahimpour2020rationality",
        title="Rationality of using phosphorus primary and secondary sources in circular economy: Game-theory-based analysis",
        authors="Rahimpour Golroudbary, Saeed and El Wali, Mohammad and Kraslawski, Andrzej",
        year="2020",
        journal="Environmental Science & Policy",
        doi="10.1016/j.envsci.2020.02.004",
        source_database="seed",
        reason_for_inclusion="Game-theoretic circular-economy analysis.",
        extra={"volume": "106", "pages": "166--176"},
    ),
    LiteratureRecord(
        id="uruena2026digital",
        title="Digital-enabled dynamic capabilities for circular economy: The role of firm size",
        authors="Uruena, Alberto and Neri, Alessandra and Cagno, Enrico and Susur, Ebru",
        year="2026",
        journal="Journal of Cleaner Production",
        doi="10.1016/j.jclepro.2026.148441",
        source_database="seed",
        reason_for_inclusion="Supplier capability and firm-size framing for digital circularity.",
        extra={"volume": "561", "pages": "148441"},
    ),
    LiteratureRecord(
        id="deangelis2018circular",
        title="Circular economy in supply chains: A review",
        authors="De Angelis, Roberta and Howard, Mickey and Miemczyk, Joe",
        year="2018",
        journal="International Journal of Production Research",
        doi="10.1080/00207543.2018.1449244",
        source_database="seed",
        reason_for_inclusion="Circular supply-chain governance foundation.",
        extra={"volume": "56", "number": "13", "pages": "4411--4427"},
    ),
    LiteratureRecord(
        id="williamson2008outsourcing",
        title="Outsourcing: Transaction cost economics and supply chain management",
        authors="Williamson, Oliver E.",
        year="2008",
        journal="Journal of Supply Chain Management",
        doi="10.1111/j.1745-493X.2008.00051.x",
        source_database="seed",
        reason_for_inclusion="Transaction cost economics foundation.",
        extra={"volume": "44", "number": "2", "pages": "5--16"},
    ),
    LiteratureRecord(
        id="poppo2002do",
        title="Do formal contracts and relational governance function as substitutes or complements?",
        authors="Poppo, Laura and Zenger, Todd",
        year="2002",
        journal="Strategic Management Journal",
        doi="10.1002/smj.249",
        source_database="seed",
        reason_for_inclusion="Formal contracts and relational governance complementarity.",
        extra={"volume": "23", "number": "8", "pages": "707--725"},
    ),
    LiteratureRecord(
        id="malhotra2011trust",
        title="Trust and collaboration in the aftermath of conflict: The effects of contract structure",
        authors="Malhotra, Deepak and Lumineau, Fabrice",
        year="2011",
        journal="Academy of Management Journal",
        doi="10.5465/amj.2009.0683",
        source_database="seed",
        reason_for_inclusion="Contract structure, trust, and collaboration.",
        extra={"volume": "54", "number": "5", "pages": "981--998"},
    ),
    LiteratureRecord(
        id="cachon2003supply",
        title="Supply chain coordination with contracts",
        authors="Cachon, Gerard P.",
        year="2003",
        journal="Handbooks in Operations Research and Management Science",
        doi="10.1016/S0927-0507(03)11006-7",
        source_database="seed",
        reason_for_inclusion="Canonical supply-chain contract theory.",
        extra={"volume": "11", "pages": "227--339"},
    ),
    LiteratureRecord(
        id="cachon2005supply",
        title="Supply chain coordination with revenue-sharing contracts: Strengths and limitations",
        authors="Cachon, Gerard P. and Lariviere, Martin A.",
        year="2005",
        journal="Management Science",
        doi="10.1287/mnsc.1040.0215",
        source_database="seed",
        reason_for_inclusion="Incentive contracts in supply-chain coordination.",
        extra={"volume": "51", "number": "1", "pages": "30--44"},
    ),
    LiteratureRecord(
        id="govindan2015reverse",
        title="Reverse logistics and closed-loop supply chain: A comprehensive review to explore the future",
        authors="Govindan, Kannan and Soleimani, Hamed and Kannan, Devika",
        year="2015",
        journal="European Journal of Operational Research",
        doi="10.1016/j.ejor.2014.07.012",
        source_database="seed",
        reason_for_inclusion="Closed-loop and reverse supply-chain context.",
        extra={"volume": "240", "number": "3", "pages": "603--626"},
    ),
    LiteratureRecord(
        id="jensen2023digital",
        title="Digital product passports for a circular economy: Data needs for product life cycle decision-making",
        authors="Jensen, Steffen Foldager and Kristensen, Jesper Hemdrup and Adamsen, Sofie and Christensen, Andreas and Waehrens, Brian Vejrum",
        year="2023",
        journal="Sustainable Production and Consumption",
        doi="10.1016/j.spc.2023.02.021",
        url="https://doi.org/10.1016/j.spc.2023.02.021",
        source_database="seed_verified",
        reason_for_inclusion="DPP data needs across life-cycle stakeholders.",
        extra={"volume": "37", "pages": "242--255"},
    ),
    LiteratureRecord(
        id="king2023proposed",
        title="A proposed universal definition of a Digital Product Passport Ecosystem (DPPE): Worldviews, discrete capabilities, stakeholder requirements and concerns",
        authors="King, Melanie R. N. and Timms, Paul D. and Mountney, Sara",
        year="2023",
        journal="Journal of Cleaner Production",
        doi="10.1016/j.jclepro.2022.135538",
        url="https://doi.org/10.1016/j.jclepro.2022.135538",
        source_database="seed_verified",
        reason_for_inclusion="DPP ecosystem capabilities and stakeholder requirements.",
        extra={"volume": "384", "pages": "135538"},
    ),
    LiteratureRecord(
        id="ducuing2023data",
        title="Data governance: Digital product passports as a case study",
        authors="Ducuing, Charlotte and Reich, Rene Herbert",
        year="2023",
        journal="European Journal of Risk Regulation",
        doi="10.1177/17835917231152799",
        url="https://doi.org/10.1177/17835917231152799",
        source_database="seed_verified",
        reason_for_inclusion="DPP data governance, access, trust, and data quality.",
        extra={"volume": "14", "number": "4", "pages": "789--808"},
    ),
    LiteratureRecord(
        id="lopes2024digital",
        title="Digital Product Passport: A Review and Research Agenda",
        authors="Lopes, Carla and Barata, Joao",
        year="2024",
        journal="Procedia Computer Science",
        doi="10.1016/j.procs.2024.09.517",
        url="https://doi.org/10.1016/j.procs.2024.09.517",
        source_database="seed_verified",
        reason_for_inclusion="Recent review of DPP structure, technologies, and adoption challenges.",
    ),
    LiteratureRecord(
        id="eu2024espr",
        title="Regulation (EU) 2024/1781 establishing a framework for the setting of ecodesign requirements for sustainable products",
        authors="{European Parliament and Council of the European Union}",
        year="2024",
        journal="Official Journal of the European Union",
        url="https://eur-lex.europa.eu/eli/reg/2024/1781/oj/eng",
        source_database="seed_verified",
        reason_for_inclusion="Legal basis for EU DPP requirements under ESPR.",
        bibtex_type="misc",
        extra={"howpublished": "\\url{https://eur-lex.europa.eu/eli/reg/2024/1781/oj/eng}"},
    ),
]


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def slugify(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return slug[:max_len] or "untitled"


def doi_key(doi: str) -> str:
    return doi.lower().replace("https://doi.org/", "").strip()


def make_record_id(title: str, year: str, doi: str = "") -> str:
    if doi:
        tail = re.sub(r"[^a-zA-Z0-9]+", "", doi_key(doi).split("/")[-1])
        if tail:
            return f"doi{tail[:18]}"
    tokens = re.findall(r"[A-Za-z0-9]+", title.lower())
    return f"{tokens[0] if tokens else 'work'}{year}"


def request_json(url: str, params: Optional[dict] = None, timeout: int = 30) -> dict:
    response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def search_openalex(query: str, per_page: int = 8) -> List[LiteratureRecord]:
    url = "https://api.openalex.org/works"
    params = {"search": query, "per-page": per_page, "mailto": os.getenv("OPENALEX_EMAIL", "research@example.com")}
    payload = request_json(url, params=params)
    records = []
    for item in payload.get("results", []):
        title = clean_text(item.get("title"))
        if not title:
            continue
        authors = " and ".join(
            clean_text(auth.get("author", {}).get("display_name"))
            for auth in item.get("authorships", [])
            if auth.get("author", {}).get("display_name")
        )
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        doi = doi_key(clean_text(item.get("doi")))
        pdf_url = ""
        best = item.get("best_oa_location") or {}
        if best.get("is_oa") and best.get("pdf_url"):
            pdf_url = clean_text(best.get("pdf_url"))
        records.append(
            LiteratureRecord(
                id=make_record_id(title, str(item.get("publication_year") or ""), doi),
                title=title,
                authors=authors,
                year=str(item.get("publication_year") or ""),
                journal=clean_text(source.get("display_name")),
                doi=doi,
                url=clean_text(item.get("doi") or item.get("id")),
                source_database="OpenAlex",
                open_access_pdf_url=pdf_url,
                reason_for_inclusion=f"Matched search string: {query}",
                notes=f"OpenAlex ID: {item.get('id', '')}",
            )
        )
    return records


def search_crossref(query: str, rows: int = 5) -> List[LiteratureRecord]:
    url = "https://api.crossref.org/works"
    payload = request_json(url, params={"query.bibliographic": query, "rows": rows})
    records = []
    for item in payload.get("message", {}).get("items", []):
        title = clean_text((item.get("title") or [""])[0])
        if not title:
            continue
        authors = " and ".join(
            " ".join(part for part in [a.get("given"), a.get("family")] if part)
            for a in item.get("author", [])
        )
        year_parts = item.get("published-print") or item.get("published-online") or item.get("created") or {}
        year = ""
        if year_parts.get("date-parts"):
            year = str(year_parts["date-parts"][0][0])
        journal = clean_text((item.get("container-title") or [""])[0])
        doi = doi_key(clean_text(item.get("DOI")))
        records.append(
            LiteratureRecord(
                id=make_record_id(title, year, doi),
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                url=clean_text(item.get("URL")),
                source_database="Crossref",
                reason_for_inclusion=f"Matched search string: {query}",
            )
        )
    return records


def search_semantic_scholar(query: str, limit: int = 5) -> List[LiteratureRecord]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,authors,year,venue,externalIds,url,openAccessPdf"
    payload = request_json(url, params={"query": query, "limit": limit, "fields": fields})
    records = []
    for item in payload.get("data", []):
        title = clean_text(item.get("title"))
        if not title:
            continue
        doi = doi_key(clean_text((item.get("externalIds") or {}).get("DOI")))
        pdf_info = item.get("openAccessPdf") or {}
        records.append(
            LiteratureRecord(
                id=make_record_id(title, str(item.get("year") or ""), doi),
                title=title,
                authors=" and ".join(clean_text(author.get("name")) for author in item.get("authors", [])),
                year=str(item.get("year") or ""),
                journal=clean_text(item.get("venue")),
                doi=doi,
                url=clean_text(item.get("url")),
                source_database="Semantic Scholar",
                open_access_pdf_url=clean_text(pdf_info.get("url")),
                reason_for_inclusion=f"Matched search string: {query}",
            )
        )
    return records


def search_core(query: str, limit: int = 5) -> List[LiteratureRecord]:
    api_key = os.getenv("CORE_API_KEY")
    if not api_key:
        return []
    url = "https://api.core.ac.uk/v3/search/works"
    headers = {**HEADERS, "Authorization": f"Bearer {api_key}"}
    response = requests.post(url, json={"q": query, "limit": limit}, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    records = []
    for item in payload.get("results", []):
        title = clean_text(item.get("title"))
        if not title:
            continue
        doi = doi_key(clean_text(item.get("doi")))
        pdf_url = clean_text(item.get("downloadUrl"))
        records.append(
            LiteratureRecord(
                id=make_record_id(title, str(item.get("yearPublished") or ""), doi),
                title=title,
                authors=" and ".join(clean_text(a.get("name")) for a in item.get("authors", [])),
                year=str(item.get("yearPublished") or ""),
                journal=clean_text(item.get("publisher")),
                doi=doi,
                url=clean_text(item.get("sourceFulltextUrls", [""])[0] if item.get("sourceFulltextUrls") else ""),
                source_database="CORE",
                open_access_pdf_url=pdf_url,
                reason_for_inclusion=f"Matched search string: {query}",
            )
        )
    return records


def lookup_unpaywall(doi: str) -> str:
    if not doi:
        return ""
    email = os.getenv("UNPAYWALL_EMAIL", "research@example.com")
    url = f"https://api.unpaywall.org/v2/{doi_key(doi)}"
    try:
        payload = request_json(url, params={"email": email})
    except Exception:
        return ""
    best = payload.get("best_oa_location") or {}
    return clean_text(best.get("url_for_pdf"))


def deduplicate(records: Iterable[LiteratureRecord]) -> List[LiteratureRecord]:
    merged: Dict[str, LiteratureRecord] = {}
    for record in records:
        key = doi_key(record.doi) if record.doi else slugify(record.title)
        if key in merged:
            existing = merged[key]
            if not existing.open_access_pdf_url and record.open_access_pdf_url:
                existing.open_access_pdf_url = record.open_access_pdf_url
            if record.source_database not in existing.source_database:
                existing.source_database = f"{existing.source_database};{record.source_database}"
            continue
        merged[key] = record
    return list(merged.values())


def download_pdf(record: LiteratureRecord) -> LiteratureRecord:
    pdf_url = record.open_access_pdf_url or lookup_unpaywall(record.doi)
    record.open_access_pdf_url = pdf_url
    if not pdf_url:
        record.notes = (record.notes + "; " if record.notes else "") + "No legal OA PDF URL found by API."
        return record
    file_name = f"{record.id}_{slugify(record.title, 45)}.pdf"
    target = PDF_DIR / file_name
    try:
        response = requests.get(pdf_url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            record.notes = (record.notes + "; " if record.notes else "") + f"OA URL did not return a PDF: {content_type}"
            return record
        target.write_bytes(response.content)
        record.open_access_pdf_path = str(target.relative_to(PROJECT_ROOT))
    except Exception as exc:
        record.notes = (record.notes + "; " if record.notes else "") + f"PDF download failed: {exc}"
    return record


def bibtex_escape(value: str) -> str:
    return value.replace("&", "\\&")


def to_bibtex(record: LiteratureRecord) -> str:
    entry_type = record.bibtex_type
    fields = {
        "title": record.title,
        "author": record.authors,
        "journal": record.journal,
        "year": record.year,
        "doi": record.doi,
        "url": record.url,
    }
    fields.update(record.extra)
    fields = {key: bibtex_escape(value) for key, value in fields.items() if value}
    body = ",\n".join(f"  {key}={{{value}}}" for key, value in fields.items())
    return f"@{entry_type}{{{record.id},\n{body}\n}}\n"


def write_outputs(records: List[LiteratureRecord]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    BIB_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "title",
        "authors",
        "year",
        "journal",
        "doi",
        "url",
        "source_database",
        "open_access_pdf_path",
        "screening_status",
        "reason_for_inclusion",
        "notes",
    ]
    with (LIT_DIR / "literature_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: getattr(record, key) for key in fieldnames})

    metadata = pd.DataFrame([{key: getattr(record, key) for key in fieldnames} for record in records])
    metadata.to_csv(DATA_DIR / "literature_metadata.csv", index=False)

    bib_records = [record for record in records if record.screening_status == "included" and record.title]
    bibtex = "\n".join(to_bibtex(record) for record in bib_records)
    (BIB_DIR / "dpp_contracting_game_literature.bib").write_text(bibtex, encoding="utf-8")
    (PAPER_DIR / "references.bib").write_text(bibtex, encoding="utf-8")


def main() -> None:
    records: List[LiteratureRecord] = list(SEED_RECORDS)

    for query in tqdm(SEARCH_STRINGS, desc="Searching literature"):
        for searcher in (search_openalex, search_crossref, search_semantic_scholar, search_core):
            try:
                records.extend(searcher(query))
            except Exception as exc:
                records.append(
                    LiteratureRecord(
                        id=f"log_{slugify(searcher.__name__)}_{slugify(query, 20)}",
                        title=f"Search failure for {query}",
                        source_database=searcher.__name__,
                        screening_status="search_error",
                        reason_for_inclusion="API search attempted but failed.",
                        notes=str(exc),
                        bibtex_type="misc",
                    )
                )
            time.sleep(0.2)

    records = deduplicate(records)
    downloaded = [download_pdf(record) for record in tqdm(records, desc="Checking OA PDFs")]
    write_outputs(downloaded)
    print(f"Wrote {len(downloaded)} literature records.")


if __name__ == "__main__":
    main()
