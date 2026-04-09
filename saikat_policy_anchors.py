"""
Map FAQ / embedding ``stub`` strings (Markdown section titles from saikat-policies.md)
to URL fragments on https://www.saikat.us/en/policies (element ``id`` values on the live site).
"""

from __future__ import annotations

POLICIES_PAGE_URL = "https://www.saikat.us/en/policies"

# Keys must match the heading text stored as ``stub`` by embed-lines-onnx (from ### lines).
STUB_TO_FRAGMENT: dict[str, str] = {
    "Build Millions of Homes": "housing",
    "Save Muni and BART": "muni-and-bart",
    "Kick Out PG&E": "lower-utility-bills",
    "Universal Pre-K and Daycare": "universal-daycare",
    "Monthly Child Stipend": "monthly-child-stipend",
    "Medicare for All": "universal-healthcare",
    "Paid Parental Leave": "paid-parental-leave",
    "Tuition-Free Public Universities and Trade Schools": "tuition-free-university",
    "Stop Corporations from Ripping You Off": "stop-corporations",
    "Build the High Speed Rail": "high-speed-rail",
    "Preventing Housing Displacement": "preventing-displacement",
    "Stop Trump's Authoritarian Coup": "stop-trumps-coup",
    "Create Real Public Safety": "real-public-safety",
    "Fully Fund Public Schools": "fund-public-schools",
    "Enshrine Reproductive Rights": "enshrine-reproductive-rights",
    "End the Wars": "end-the-wars",
    "Protect the LGBTQ+ Community": "protect-lgbtq-community",
    "Disability Rights": "disability-rights",
    "Stop Funding Genocide": "stop-funding-genocide",
    "US-China relations": "US-China-Relations",
    "Build the Clean Economy to Create Prosperity for All": "clean-economy",
    "Welcome Immigrants": "welcome-immigrants",
    "Making the Wealthy and Corporations Pay Their Fair Share": "wealth-tax",
    "Empower Workers": "empower-workers",
    "AI": "AI",
    "Ban Congressional Stock Trading": "ban-stock-trading",
    "End Money in Politics": "money-in-politics",
    "I'd love to get your feedback on my platform — both on the positions themselves and how we're explaining them.": "feedback",
}


def policies_url_for_stub(stub: str | None) -> str | None:
    """Return full policies URL with hash, or None if stub is unknown."""
    if not stub or not str(stub).strip():
        return None
    key = str(stub).strip()
    frag = STUB_TO_FRAGMENT.get(key)
    if frag is None:
        return None
    return f"{POLICIES_PAGE_URL}#{frag}"
