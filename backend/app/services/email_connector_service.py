"""Email Connector Service — Phase 88.

SMTP-based email dispatch for the autonomous action engine.
This is a separate transport from ExternalConnectorService (which is HTTP-only).

Supports plain-text and HTML emails, BCC, Reply-To, and custom headers.
All I/O is synchronous (smtplib); callers should run via asyncio.to_thread
when used from async context.

Usage::

    from app.services.email_connector_service import EmailConnectorService, EmailMessage

    svc = EmailConnectorService(
        smtp_host="smtp.example.com",
        smtp_port=587,
        username="ninai@example.com",
        password="secret",
        use_tls=True,
    )
    result = svc.send(EmailMessage(
        to=["recipient@example.com"],
        subject="Ninai alert: anomaly detected",
        body="An anomaly was detected in your memory stream.",
    ))
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15.0


@dataclass
class EmailMessage:
    to: list[str]
    subject: str
    body: str
    from_addr: str | None = None       # falls back to service default_from
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)
    reply_to: str | None = None
    html_body: str | None = None       # when set, sends multipart/alternative
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class EmailDispatchResult:
    status: str                         # "success" | "failed"
    error: str | None = None
    recipients_accepted: list[str] = field(default_factory=list)
    recipients_rejected: dict[str, Any] = field(default_factory=dict)


class EmailConnectorService:
    """SMTP email dispatcher.

    Parameters
    ----------
    smtp_host:    SMTP server hostname.
    smtp_port:    SMTP port (25, 465, 587).
    username:     SMTP auth username. None = no auth.
    password:     SMTP auth password.
    use_tls:      True → STARTTLS on port 587. False → plain SMTP or SMTPS.
    use_ssl:      True → SSL from connection start (port 465). Mutually exclusive with use_tls.
    default_from: Default From address used when EmailMessage.from_addr is None.
    timeout:      Socket timeout in seconds.
    """

    def __init__(
        self,
        *,
        smtp_host: str,
        smtp_port: int = 587,
        username: str | None = None,
        password: str | None = None,
        use_tls: bool = True,
        use_ssl: bool = False,
        default_from: str = "ninai@ninai.ai",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.use_ssl = use_ssl
        self.default_from = default_from
        self.timeout = timeout

    def send(self, message: EmailMessage) -> EmailDispatchResult:
        """Send an email synchronously.

        Run via asyncio.to_thread() from async callers.
        """
        from_addr = str(message.from_addr or self.default_from)
        all_recipients = list(message.to) + list(message.cc) + list(message.bcc)

        if not all_recipients:
            return EmailDispatchResult(status="failed", error="No recipients specified")

        try:
            mime_msg = self._build_mime(message, from_addr)
            refused = self._dispatch_smtp(mime_msg, from_addr, all_recipients)
            accepted = [r for r in all_recipients if r not in refused]
            return EmailDispatchResult(
                status="success" if accepted else "failed",
                recipients_accepted=accepted,
                recipients_rejected=refused,
                error=f"{len(refused)} recipient(s) rejected" if refused else None,
            )
        except smtplib.SMTPAuthenticationError as exc:
            logger.error("SMTP auth failed: %s", exc)
            return EmailDispatchResult(status="failed", error=f"Auth failed: {exc}")
        except smtplib.SMTPConnectError as exc:
            logger.error("SMTP connect failed: %s", exc)
            return EmailDispatchResult(status="failed", error=f"Connect failed: {exc}")
        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return EmailDispatchResult(status="failed", error=str(exc))

    def _build_mime(self, message: EmailMessage, from_addr: str) -> MIMEMultipart:
        if message.html_body:
            mime_msg = MIMEMultipart("alternative")
            mime_msg.attach(MIMEText(str(message.body), "plain", "utf-8"))
            mime_msg.attach(MIMEText(str(message.html_body), "html", "utf-8"))
        else:
            mime_msg = MIMEMultipart()
            mime_msg.attach(MIMEText(str(message.body), "plain", "utf-8"))

        mime_msg["From"] = from_addr
        mime_msg["To"] = ", ".join(str(r) for r in message.to)
        mime_msg["Subject"] = str(message.subject)

        if message.cc:
            mime_msg["Cc"] = ", ".join(str(r) for r in message.cc)
        if message.reply_to:
            mime_msg["Reply-To"] = str(message.reply_to)
        for header_name, header_val in (message.headers or {}).items():
            mime_msg[str(header_name)] = str(header_val)

        return mime_msg

    def _dispatch_smtp(
        self,
        mime_msg: MIMEMultipart,
        from_addr: str,
        recipients: list[str],
    ) -> dict[str, Any]:
        """Open SMTP connection, authenticate if needed, and send.

        Returns a dict of refused recipients (empty on full success).
        """
        smtp_cls = smtplib.SMTP_SSL if self.use_ssl else smtplib.SMTP
        with smtp_cls(self.smtp_host, self.smtp_port, timeout=self.timeout) as server:
            if self.use_tls and not self.use_ssl:
                server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            refused = server.sendmail(from_addr, recipients, mime_msg.as_string())
        return refused or {}
