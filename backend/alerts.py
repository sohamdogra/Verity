"""Family alerts: always simulated in-app; optionally forwarded to ALERT_WEBHOOK_URL."""

import logging
from typing import Literal

import httpx

import config

log = logging.getLogger("verity.alerts")

Delivery = Literal["simulated", "webhook", "webhook_failed"]


def _payload(url: str, message: str, details: dict) -> dict:
    if "discord.com/api/webhooks" in url or "discordapp.com/api/webhooks" in url:
        return {"content": message}
    if "hooks.slack.com" in url:
        return {"text": message}
    return {"message": message, **details}  # custom endpoint / Twilio proxy


def send_alert(message: str, details: dict) -> Delivery:
    url = config.ALERT_WEBHOOK_URL
    if not url:
        return "simulated"
    try:
        response = httpx.post(url, json=_payload(url, message, details), timeout=5.0)
        response.raise_for_status()
        return "webhook"
    except Exception as exc:
        log.warning("Alert webhook failed (alert still logged in-app): %s", exc)
        return "webhook_failed"
