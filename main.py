"""Send a preference profile to the running FastAPI recommender."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = BASE_DIR / "sample_preference_profile1.json"
DEFAULT_API_URL = "http://127.0.0.1:8000"


def request_recommendations(api_url: str, profile_path: Path, count: int) -> list[dict]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    request = Request(
        f"{api_url.rstrip('/')}/recommend?top_n={count}",
        data=json.dumps(profile).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    args = parser.parse_args()

    try:
        recommendations = request_recommendations(args.api_url, args.profile, args.count)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"FastAPI rejected the request ({error.code}): {detail}") from error
    except URLError as error:
        raise SystemExit(f"Could not reach FastAPI at {args.api_url}: {error.reason}") from error

    print(json.dumps(recommendations, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
