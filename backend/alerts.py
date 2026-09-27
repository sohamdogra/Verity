"""Family alerts.

Delivery is chosen automatically, in this order:
  1. Twilio SMS      - a real text message, when TWILIO_* is configured and we have a number
  2. Webhook         - Discord / Slack / any JSON endpoint, when ALERT_WEBHOOK_URL is set
  3. Simulated       - always available; the alert is still recorded in the family's history

The app never depends on delivery succeeding: verification and the trusted callback keep
working, and the UI always offers a "text them yourself" fallback that needs no network.
"""

import logging
import re
from typing import Literal

import httpx

import config

log = logging.getLogger("verity.alerts")

Delivery = Literal["simulated", "sms", "sms_failed", "webhook", "webhook_failed"]

TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


# ---- phone numbers ------------------------------------------------------------------


def to_e164(raw: str | None, default_country: str = "1") -> str | None:
    """Best-effort E.164 ("+14045550199"). Returns None if there aren't enough digits.

    Deliberately simple: no phonenumbers dependency, and it only has to handle what a
    family types into the setup form."""
    if not raw:
        return None
    text = raw.strip()
    plus = text.startswith("+") or text.startswith("00")
    digits = re.sub(r"\D", "", text)
    if text.startswith("00"):
        digits = digits[2:]
    if not digits:
        return None
    if not plus:
        if len(digits) == 10 and default_country == "1":  # US/Canada without country code
            digits = default_country + digits
        elif len(digits) == 11 and digits.startswith("1") and default_country == "1":
            pass  # already has the country code
        elif len(digits) < 10:
            return None
    return "+" + digits if 8 <= len(digits) <= 15 else None


def extract_phone(text: str | None) -> str | None:
    """Pull a phone number out of free text like "Maya (daughter) +1 404 555 0199"."""
    if not text:
        return None
    for candidate in re.findall(r"\+?[\d][\d\s().\-]{6,}\d", text):
        number = to_e164(candidate)
        if number:
            return number
    return None


def display_name(text: str | None) -> str:
    """The human part of an alert contact, with any phone number removed."""
    if not text:
        return "your family"
    name = re.sub(r"\+?[\d][\d\s().\-]{6,}\d", "", text)
    name = re.sub(r"[\s,;|·\-]+$", "", name.strip()).strip(" ,;|·-")
    return name or "your family"


# ---- delivery -----------------------------------------------------------------------


def sms_configured() -> bool:
    return bool(config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN and config.TWILIO_FROM_NUMBER)


def send_sms(to_number: str, message: str) -> tuple[bool, str | None]:
    """Send one SMS through Twilio. Returns (ok, error).

    On a trial account Twilio rejects custom text, so TWILIO_TRIAL_TEMPLATE names one of
    their predefined templates instead and they substitute their own wording."""
    body = config.TWILIO_TRIAL_TEMPLATE or message
    try:
        response = httpx.post(
            TWILIO_API.format(sid=config.TWILIO_ACCOUNT_SID),
            auth=(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN),
            data={"To": to_number, "From": config.TWILIO_FROM_NUMBER, "Body": body},
            timeout=10.0,
        )
        if response.status_code >= 400:
            # Twilio returns a helpful JSON error; surface it so setup mistakes are obvious.
            detail = response.json().get("message", response.text[:200]) if response.content else response.reason_phrase
            log.warning("Twilio SMS to %s failed (%s): %s", to_number, response.status_code, detail)
            return False, str(detail)
        return True, None
    except Exception as exc:
        log.warning("Twilio SMS to %s failed: %s", to_number, exc)
        return False, str(exc)


def _webhook_payload(url: str, message: str, details: dict) -> dict:
    if "discord.com/api/webhooks" in url or "discordapp.com/api/webhooks" in url:
        return {"content": message}
    if "hooks.slack.com" in url:
        return {"text": message}
    return {"message": message, **details}  # custom endpoint


def send_webhook(message: str, details: dict) -> Delivery:
    url = config.ALERT_WEBHOOK_URL
    if not url:
        return "simulated"
    try:
        response = httpx.post(url, json=_webhook_payload(url, message, details), timeout=5.0)
        response.raise_for_status()
        return "webhook"
    except Exception as exc:
        log.warning("Alert webhook failed (alert still logged in-app): %s", exc)
        return "webhook_failed"


def send_alert(message: str, details: dict, to_number: str | None = None) -> tuple[Delivery, str | None]:
    """Deliver an alert the best way available. Returns (delivery, error)."""
    if to_number and sms_configured():
        ok, error = send_sms(to_number, message)
        if ok:
            return "sms", None
        # Fall through to the webhook so a Twilio outage isn't a dead end.
        fallback = send_webhook(message, details)
        return ("webhook" if fallback == "webhook" else "sms_failed"), error
    return send_webhook(message, details), None
