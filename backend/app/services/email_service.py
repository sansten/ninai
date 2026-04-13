"""Transactional email via SendGrid HTTP API, with SMTP fallback."""

from __future__ import annotations

import logging
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Send transactional email.

    Tries SendGrid when SENDGRID_API_KEY is set, falls back to SMTP when
    SMTP_HOST is set, otherwise logs a warning and does nothing.
    Failures are always caught and logged — email is best-effort and must
    never crash an API request.
    """

    async def _sendgrid(self, to: str, subject: str, html: str) -> None:
        payload = {
            "personalizations": [{"to": [{"email": to}]}],
            "from": {
                "email": settings.EMAIL_FROM,
                "name": settings.EMAIL_FROM_NAME,
            },
            "subject": subject,
            "content": [{"type": "text/html", "value": html}],
        }
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                json=payload,
                headers={"Authorization": f"Bearer {settings.SENDGRID_API_KEY}"},
            )
            r.raise_for_status()

    def _smtp(self, to: str, subject: str, html: str) -> None:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM}>"
        msg["To"] = to
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as s:
            if settings.SMTP_USE_TLS:
                s.starttls()
            if settings.SMTP_USERNAME:
                s.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            s.sendmail(settings.EMAIL_FROM, to, msg.as_string())

    async def send(self, to: str, subject: str, html: str) -> None:
        try:
            if settings.SENDGRID_API_KEY:
                await self._sendgrid(to, subject, html)
            elif settings.SMTP_HOST:
                self._smtp(to, subject, html)
            else:
                logger.warning("No email backend configured — skipping send to %s", to)
        except Exception as exc:
            logger.error("Email send failed to %s: %s", to, exc)

    async def send_verification(self, to: str, token: str, org_name: str) -> None:
        url = f"{settings.FRONTEND_URL}/signup/verify?token={token}"
        await self.send(
            to=to,
            subject="Verify your Ninai account",
            html=(
                f"<p>Welcome to Ninai, <b>{org_name}</b>!</p>"
                f"<p><a href='{url}'>Click here to verify your email address</a></p>"
                f"<p>This link expires in 24 hours.</p>"
            ),
        )

    async def send_welcome(self, to: str, org_name: str) -> None:
        await self.send(
            to=to,
            subject="Your Ninai workspace is ready",
            html=(
                f"<p>Your workspace <b>{org_name}</b> is active.</p>"
                f"<p><a href='{settings.FRONTEND_URL}'>Sign in now</a></p>"
            ),
        )

    async def send_version_deprecation_notice(
        self, to: str, version: str, deadline: date
    ) -> None:
        await self.send(
            to=to,
            subject=f"Ninai API {version} deprecation notice",
            html=(
                f"<p>Ninai API <b>{version}</b> will be retired on <b>{deadline}</b>.</p>"
                f"<p>Please migrate before this date. Your workspace will be "
                f"auto-migrated after the deadline.</p>"
                f"<p><a href='{settings.FRONTEND_URL}/docs/migration'>Migration guide</a></p>"
            ),
        )


# Module-level singleton — import and use directly
email_service = EmailService()
