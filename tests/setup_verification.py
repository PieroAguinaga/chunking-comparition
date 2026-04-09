"""
scripts/verify_setup.py

Run this script before starting the server to verify that all
external connections (Azure OpenAI, Supabase) are working correctly.

Usage:
    python scripts/verify_setup.py
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()


def check(label: str, fn) -> bool:
    """Run a check function and print pass/fail."""
    try:
        fn()
        print(f"  ✓  {label}")
        return True
    except Exception as exc:
        print(f"  ✗  {label}")
        print(f"     └─ {exc}")
        return False


def verify_env():
    """Check that all required environment variables are set."""
    from config.settings import settings
    required = [
        ("AZURE_CHAT_DEPLOYMENT",    settings.azure_chat_deployment),
        ("AZURE_OPENAI_ENDPOINT",    settings.azure_openai_endpoint),
        ("AZURE_OPENAI_API_KEY",     settings.azure_openai_api_key),
        ("AZURE_OPENAI_API_VERSION", settings.azure_openai_api_version),
        ("SUPABASE_URL",             settings.supabase_url),
        ("SUPABASE_SERVICE_ROLE_KEY",settings.supabase_service_role_key),
    ]
    missing = [name for name, val in required if not val or val.startswith("your-")]
    if missing:
        raise ValueError(f"Missing or placeholder values: {', '.join(missing)}")


def verify_azure():
    """Send a minimal prompt to Azure OpenAI and verify a response is returned."""
    from llm import get_llm
    llm = get_llm()
    response = llm.invoke("Reply with the single word: OK")
    assert response.content.strip(), "Empty response from Azure OpenAI"


def verify_supabase_connection():
    """Connect to Supabase and verify the project is reachable."""
    from supabase import create_client
    from config.settings import settings
    client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    # A simple select on a system table that always exists
    client.table("conversation_history").select("id").limit(1).execute()




def main():
    print("\n── Project Setup Verification ──────────────────────────────\n")

    checks = [
        ("Environment variables",   verify_env),
        ("Azure OpenAI connection", verify_azure),
        ("Supabase connection",     verify_supabase_connection),
    ]


    results = [check(label, fn) for label, fn in checks]
    passed  = sum(results)
    total   = len(results)

    print(f"\n── Result: {passed}/{total} checks passed ─────────────────────\n")

    if passed == total:
        print("  Everything looks good.\n")
        sys.exit(0)
    else:
        print("  Fix the issues above, then re-run this script.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
