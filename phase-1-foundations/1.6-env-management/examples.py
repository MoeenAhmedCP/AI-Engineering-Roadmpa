"""
1.6 Environment Management — Examples
Demonstrates: python-dotenv loading, pydantic-settings BaseSettings,
startup validation, environment-aware config, and .env.example generation.
"""

import os
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Load .env with python-dotenv (graceful fallback if not installed)
# ---------------------------------------------------------------------------

def load_env_file(path: str = ".env") -> bool:
    """Load environment variables from a .env file. Returns True on success."""
    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(path, override=False)
        if loaded:
            print(f"[env] Loaded environment from {path}")
        else:
            print(f"[env] No .env file found at {path} — using system env vars")
        return loaded
    except ImportError:
        print("[env] python-dotenv not installed. Install with: pip install python-dotenv")
        print("[env] Falling back to system environment variables only.")
        return False


# ---------------------------------------------------------------------------
# 2. Settings class — pydantic-settings preferred, dataclass fallback
# ---------------------------------------------------------------------------

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import SecretStr, Field

    class Settings(BaseSettings):
        """
        Production-grade settings class using pydantic-settings.
        Reads from environment variables (and optionally .env file).
        SecretStr values are masked in logs and repr.
        """
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            # Don't raise if .env file doesn't exist
            env_file=None,
        )

        # AI API Keys — SecretStr prevents accidental logging
        anthropic_api_key: SecretStr = Field(default=SecretStr(""), description="Anthropic API key")
        openai_api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")

        # Database / Cache
        database_url: str = Field(default="", description="PostgreSQL connection string")
        redis_url: str = Field(default="redis://localhost:6379", description="Redis connection string")

        # App configuration
        debug: bool = Field(default=False, description="Enable debug mode")
        environment: str = Field(default="development", description="development | test | production")

        # AI model settings
        max_tokens: int = Field(default=4096, ge=1, le=200000)
        model_name: str = Field(default="claude-sonnet-4-6")
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)

        def get_anthropic_key(self) -> str:
            """Return the raw Anthropic API key string."""
            return self.anthropic_api_key.get_secret_value()

        def get_openai_key(self) -> str:
            """Return the raw OpenAI API key string."""
            return self.openai_api_key.get_secret_value()

        def is_production(self) -> bool:
            return self.environment == "production"

        def is_test(self) -> bool:
            return self.environment == "test"

    PYDANTIC_AVAILABLE = True
    print("[settings] Using pydantic-settings BaseSettings")

except ImportError:
    # Fallback: simple dataclass that reads from os.environ directly
    print("[settings] pydantic-settings not installed. Using dataclass fallback.")
    print("[settings] Install with: pip install pydantic-settings")

    @dataclass
    class Settings:  # type: ignore[no-redef]
        """Fallback settings using stdlib dataclass + os.environ."""
        anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
        openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
        database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
        redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
        debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
        environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
        max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096")))
        model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "claude-sonnet-4-6"))
        temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))

        def get_anthropic_key(self) -> str:
            return self.anthropic_api_key

        def get_openai_key(self) -> str:
            return self.openai_api_key

        def is_production(self) -> bool:
            return self.environment == "production"

        def is_test(self) -> bool:
            return self.environment == "test"

    PYDANTIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# 3. startup_check() — validate all required env vars at boot time
# ---------------------------------------------------------------------------

REQUIRED_VARS: list[tuple[str, str]] = [
    ("ANTHROPIC_API_KEY", "Get from: https://console.anthropic.com/"),
    ("DATABASE_URL", "Format: postgresql://user:pass@host:5432/dbname"),
]

OPTIONAL_VARS: list[tuple[str, str, str]] = [
    ("OPENAI_API_KEY", "sk-...", "Required only if using OpenAI models"),
    ("REDIS_URL", "redis://localhost:6379", "Required for caching / rate limiting"),
    ("DEBUG", "false", "Set to true for verbose logging"),
    ("ENVIRONMENT", "development", "development | test | production"),
]


def startup_check(raise_on_missing: bool = True) -> bool:
    """
    Validate that all required environment variables are present.
    Called at application startup before any routes are registered.

    Args:
        raise_on_missing: If True, raise SystemExit on missing vars.
                          Set False in tests to just return False.

    Returns:
        True if all required vars are present, False otherwise.
    """
    missing: list[tuple[str, str]] = []

    for var_name, hint in REQUIRED_VARS:
        value = os.getenv(var_name, "").strip()
        if not value:
            missing.append((var_name, hint))

    if missing:
        print("\n" + "=" * 60)
        print("STARTUP ERROR: Missing required environment variables")
        print("=" * 60)
        for var_name, hint in missing:
            print(f"  MISSING: {var_name}")
            print(f"    Hint:  {hint}")
        print()
        print("Set these variables in your .env file or shell environment.")
        print("See .env.example for a template.")
        print("=" * 60 + "\n")

        if raise_on_missing:
            sys.exit(1)
        return False

    print("[startup] All required environment variables are present.")

    # Warn about missing optional vars
    for var_name, default, note in OPTIONAL_VARS:
        if not os.getenv(var_name):
            print(f"[startup] Optional {var_name} not set (default: {default!r}) — {note}")

    return True


# ---------------------------------------------------------------------------
# 4. Environment-aware config factory
# ---------------------------------------------------------------------------

def get_settings() -> Settings:
    """
    Return a Settings instance tuned for the current ENVIRONMENT.
    Dev gets verbose debug mode; prod enforces stricter settings.
    In real FastAPI apps, wrap this with @lru_cache so it runs once.
    """
    env = os.getenv("ENVIRONMENT", "development").lower()

    # Load the appropriate .env file first
    if env == "test":
        load_env_file(".env.test")
    elif env == "production":
        load_env_file(".env.production")
    else:
        load_env_file(".env.development")
        load_env_file(".env")  # fallback to base

    settings = Settings()

    # Enforce production safety rules
    if settings.is_production():
        if settings.debug:
            print("[WARNING] DEBUG=true in production — this leaks internal details!")
        if not settings.get_anthropic_key():
            raise ValueError("ANTHROPIC_API_KEY must be set in production")

    # Dev-only conveniences
    if not settings.is_production() and not settings.is_test():
        print(f"[config] Running in {settings.environment} mode")
        print(f"[config] Model: {settings.model_name} | Max tokens: {settings.max_tokens}")
        print(f"[config] Debug: {settings.debug}")

    return settings


# ---------------------------------------------------------------------------
# 5. Print a .env.example template to stdout
# ---------------------------------------------------------------------------

ENV_EXAMPLE_TEMPLATE = """\
# .env.example — Copy to .env and fill in real values
# DO NOT commit .env — only commit this .env.example file

# ---- AI API Keys ----
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-openai-key-here

# ---- Database ----
DATABASE_URL=postgresql://postgres:password@localhost:5432/docsense
REDIS_URL=redis://localhost:6379

# ---- App Config ----
DEBUG=true
ENVIRONMENT=development

# ---- AI Model Settings ----
MODEL_NAME=claude-sonnet-4-6
MAX_TOKENS=4096
TEMPERATURE=0.7
"""


def print_env_example() -> None:
    """Print the .env.example template to stdout."""
    print("\n--- .env.example template ---")
    print(ENV_EXAMPLE_TEMPLATE)
    print("--- end template ---\n")


def write_env_example(path: str = ".env.example") -> None:
    """Write the .env.example template to a file (only if it doesn't exist)."""
    if os.path.exists(path):
        print(f"[env] {path} already exists — not overwriting.")
        return
    with open(path, "w") as f:
        f.write(ENV_EXAMPLE_TEMPLATE)
    print(f"[env] Created {path}")


# ---------------------------------------------------------------------------
# 6. Demo block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("1.6 Environment Management — Demo")
    print("=" * 60)

    # Demo 1: Show .env.example template
    print("\n[Demo 1] .env.example template:")
    print_env_example()

    # Demo 2: Simulate loading settings (without real env vars)
    print("[Demo 2] Creating Settings instance:")
    # Set some demo values so the Settings object can be created
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("DEBUG", "true")

    settings = Settings()
    print(f"  environment  = {settings.environment}")
    print(f"  debug        = {settings.debug}")
    print(f"  model_name   = {settings.model_name}")
    print(f"  max_tokens   = {settings.max_tokens}")
    print(f"  is_prod      = {settings.is_production()}")
    print(f"  is_test      = {settings.is_test()}")
    # SecretStr masking
    if PYDANTIC_AVAILABLE:
        print(f"  api_key repr = {settings.anthropic_api_key!r}")  # shows ***** not the real key
    else:
        print("  (dataclass fallback — SecretStr masking not available)")

    # Demo 3: startup_check (won't raise since we pass raise_on_missing=False)
    print("\n[Demo 3] startup_check() without required vars:")
    all_present = startup_check(raise_on_missing=False)
    print(f"  All required vars present: {all_present}")

    # Demo 4: Simulate setting required vars and re-checking
    print("\n[Demo 4] startup_check() with required vars set:")
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-demo-key-not-real"
    os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/demo"
    all_present = startup_check(raise_on_missing=False)
    print(f"  All required vars present: {all_present}")

    print("\nDone. In a FastAPI app, call get_settings() and startup_check() in main.py.")
