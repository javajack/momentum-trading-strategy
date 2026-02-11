"""
Zerodha Kite Connect authentication for FORTRESS MOMENTUM.

Enforces invariants:
- P1: Authentication required before any API call
- P2: Token cache expires daily
"""

import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

from kiteconnect import KiteConnect


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


class ZerodhaAuth:
    """
    Handles Zerodha Kite Connect authentication.

    Flow:
    1. User provides API key and secret in config
    2. System opens login URL in browser
    3. User logs in and gets redirected with request_token
    4. User pastes request_token in console
    5. System exchanges for access_token
    6. Access token cached for the day (P2)
    """

    TOKEN_CACHE_FILE = ".kite_token_cache.json"

    # Increased timeout for slow connections
    DEFAULT_TIMEOUT = 30  # seconds

    def __init__(self, api_key: str, api_secret: str, timeout: int = None):
        """
        Initialize auth handler.

        Args:
            api_key: Zerodha API key
            api_secret: Zerodha API secret
            timeout: Request timeout in seconds (default: 30)
        """
        if not api_key or not api_secret:
            raise AuthenticationError(
                "API key and secret required. Set them in config.yaml"
            )

        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.kite = KiteConnect(api_key=api_key, timeout=self.timeout)
        self._access_token: Optional[str] = None

    def get_login_url(self) -> str:
        """
        Get the Kite login URL.

        Returns:
            Login URL string
        """
        return self.kite.login_url()

    def authenticate(self, request_token: str) -> str:
        """
        Exchange request_token for access_token.

        Args:
            request_token: Token from redirect URL after login

        Returns:
            Access token string

        Raises:
            AuthenticationError: If authentication fails
        """
        if not request_token:
            raise AuthenticationError("No request_token provided")

        try:
            data = self.kite.generate_session(
                request_token=request_token,
                api_secret=self.api_secret,
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to generate session: {e}")

        self._access_token = data["access_token"]
        self.kite.set_access_token(self._access_token)

        # Cache token for the day (P2)
        self._save_token_cache()

        return self._access_token

    def _save_token_cache(self) -> None:
        """Save access token to cache file."""
        cache = {
            "access_token": self._access_token,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "api_key": self.api_key,
        }
        Path(self.TOKEN_CACHE_FILE).write_text(json.dumps(cache))

    def _load_token_cache(self) -> Optional[str]:
        """
        Load access token from cache if valid for today.

        P2: Token cache expires daily.

        Returns:
            Cached access token or None
        """
        try:
            cache = json.loads(Path(self.TOKEN_CACHE_FILE).read_text())

            # P2: Check if token is from today
            if cache.get("date") != datetime.now().strftime("%Y-%m-%d"):
                return None

            # Check if same API key
            if cache.get("api_key") != self.api_key:
                return None

            return cache.get("access_token")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def is_authenticated(self) -> bool:
        """
        Check if we have a valid access token.

        Returns:
            True if authenticated
        """
        if self._access_token:
            return True

        cached = self._load_token_cache()
        if cached:
            self.kite.set_access_token(cached)
            try:
                self.kite.profile()
            except Exception:
                return False
            self._access_token = cached
            return True

        return False

    def get_kite(self) -> KiteConnect:
        """
        Get authenticated KiteConnect instance.

        P1: Authentication required before any API call.

        Returns:
            KiteConnect instance

        Raises:
            AuthenticationError: If not authenticated
        """
        if not self.is_authenticated():
            raise AuthenticationError(
                "Not authenticated. Call login_interactive() first."
            )
        return self.kite

    def login_interactive(self) -> KiteConnect:
        """
        Interactive login flow for CLI.

        Returns:
            Authenticated KiteConnect instance
        """
        # Check cache first
        cached_token = self._load_token_cache()
        if cached_token:
            print("Using cached access token from today")
            self.kite.set_access_token(cached_token)
            self._access_token = cached_token

            # Validate cached token with a simple API call
            try:
                self.kite.profile()  # Lightweight validation call
                return self.kite
            except Exception as e:
                print(f"Cached token invalid: {e}")
                print("Starting fresh login...")
                self._access_token = None
                self.logout()  # Clear invalid cache

        # Fresh login required
        login_url = self.get_login_url()
        print("\nOpening Zerodha login in browser...")
        print(f"URL: {login_url}\n")

        try:
            webbrowser.open(login_url)
        except Exception:
            print("Could not open browser. Please open the URL manually.")

        print("After logging in, you will be redirected to your redirect URL.")
        print("Copy the 'request_token' parameter from the URL.\n")

        request_token = input("Paste request_token here: ").strip()

        if not request_token:
            raise AuthenticationError("No request_token provided")

        self.authenticate(request_token)
        print("Authentication successful!\n")

        return self.kite

    def logout(self) -> None:
        """Clear cached token and logout."""
        try:
            Path(self.TOKEN_CACHE_FILE).unlink()
        except FileNotFoundError:
            pass

        self._access_token = None
        print("Logged out successfully")
