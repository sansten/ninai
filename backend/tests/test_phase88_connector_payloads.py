"""Phase 88 — Connector Payload Builder + Email Connector Service tests."""
from __future__ import annotations

import json
import smtplib
import unittest.mock as mock

import pytest

from app.services.connector_payload_builder import ConnectorPayloadBuilder
from app.services.email_connector_service import (
    EmailConnectorService,
    EmailMessage,
    EmailDispatchResult,
)


# ---------------------------------------------------------------------------
# Slack payloads
# ---------------------------------------------------------------------------

class TestSlackPayload:
    def test_minimal_text_only(self):
        p = ConnectorPayloadBuilder.slack("Hello from Ninai")
        assert p["text"] == "Hello from Ninai"
        assert p["username"] == "Ninai"

    def test_channel_included_when_provided(self):
        p = ConnectorPayloadBuilder.slack("msg", channel="#alerts")
        assert p["channel"] == "#alerts"

    def test_no_channel_key_when_omitted(self):
        p = ConnectorPayloadBuilder.slack("msg")
        assert "channel" not in p

    def test_no_blocks_key_when_omitted(self):
        p = ConnectorPayloadBuilder.slack("msg")
        assert "blocks" not in p

    def test_blocks_included_when_provided(self):
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "hello"}}]
        p = ConnectorPayloadBuilder.slack("fallback", blocks=blocks)
        assert p["blocks"] == blocks

    def test_custom_username_and_emoji(self):
        p = ConnectorPayloadBuilder.slack("hi", username="Bot", icon_emoji=":robot:")
        assert p["username"] == "Bot"
        assert p["icon_emoji"] == ":robot:"

    def test_block_kit_message_structure(self):
        p = ConnectorPayloadBuilder.slack_blocks(
            header="Alert: Memory Anomaly",
            body="An anomaly was detected in the memory stream.",
            footer="Ninai Cognitive OS",
        )
        assert p["text"] == "Alert: Memory Anomaly"
        assert "blocks" in p
        block_types = [b["type"] for b in p["blocks"]]
        assert "header" in block_types
        assert "section" in block_types
        assert "context" in block_types

    def test_block_kit_no_footer(self):
        p = ConnectorPayloadBuilder.slack_blocks("H", "B")
        block_types = [b["type"] for b in p["blocks"]]
        assert "context" not in block_types

    def test_header_truncated_at_150_chars(self):
        long_header = "X" * 200
        p = ConnectorPayloadBuilder.slack_blocks(long_header, "body")
        header_block = next(b for b in p["blocks"] if b["type"] == "header")
        assert len(header_block["text"]["text"]) <= 150

    def test_payload_is_json_serializable(self):
        p = ConnectorPayloadBuilder.slack("test")
        assert json.dumps(p)  # must not raise


# ---------------------------------------------------------------------------
# Jira payloads
# ---------------------------------------------------------------------------

class TestJiraPayload:
    def test_minimal_structure(self):
        p = ConnectorPayloadBuilder.jira("Fix login bug", project_key="ENG")
        assert p["fields"]["project"]["key"] == "ENG"
        assert p["fields"]["summary"] == "Fix login bug"
        assert p["fields"]["issuetype"]["name"] == "Task"

    def test_custom_issue_type(self):
        p = ConnectorPayloadBuilder.jira("s", project_key="K", issue_type="Bug")
        assert p["fields"]["issuetype"]["name"] == "Bug"

    def test_description_uses_adf_format(self):
        p = ConnectorPayloadBuilder.jira("s", project_key="K", description="Details here")
        desc = p["fields"]["description"]
        assert desc["type"] == "doc"
        assert desc["version"] == 1
        text_node = desc["content"][0]["content"][0]
        assert text_node["type"] == "text"
        assert text_node["text"] == "Details here"

    def test_no_description_key_when_omitted(self):
        p = ConnectorPayloadBuilder.jira("s", project_key="K")
        assert "description" not in p["fields"]

    def test_labels_included(self):
        p = ConnectorPayloadBuilder.jira("s", project_key="K", labels=["bug", "urgent"])
        assert p["fields"]["labels"] == ["bug", "urgent"]

    def test_priority_included(self):
        p = ConnectorPayloadBuilder.jira("s", project_key="K", priority="High")
        assert p["fields"]["priority"]["name"] == "High"

    def test_assignee_included(self):
        p = ConnectorPayloadBuilder.jira("s", project_key="K", assignee_account_id="abc123")
        assert p["fields"]["assignee"]["accountId"] == "abc123"

    def test_summary_truncated_at_255(self):
        p = ConnectorPayloadBuilder.jira("X" * 300, project_key="K")
        assert len(p["fields"]["summary"]) <= 255

    def test_payload_is_json_serializable(self):
        p = ConnectorPayloadBuilder.jira("title", project_key="K", description="desc")
        assert json.dumps(p)


# ---------------------------------------------------------------------------
# GitHub payloads
# ---------------------------------------------------------------------------

class TestGitHubPayload:
    def test_minimal_issue(self):
        p = ConnectorPayloadBuilder.github_issue("Memory leak detected")
        assert p["title"] == "Memory leak detected"
        assert "body" not in p

    def test_full_issue(self):
        p = ConnectorPayloadBuilder.github_issue(
            "Bug report",
            body="Steps to reproduce...",
            labels=["bug", "memory"],
            assignees=["ts"],
            milestone=3,
        )
        assert p["body"] == "Steps to reproduce..."
        assert p["labels"] == ["bug", "memory"]
        assert p["assignees"] == ["ts"]
        assert p["milestone"] == 3

    def test_title_truncated_at_256(self):
        p = ConnectorPayloadBuilder.github_issue("T" * 300)
        assert len(p["title"]) <= 256

    def test_github_comment_payload(self):
        p = ConnectorPayloadBuilder.github_comment("LGTM")
        assert p == {"body": "LGTM"}

    def test_payload_is_json_serializable(self):
        p = ConnectorPayloadBuilder.github_issue("t", body="b", labels=["l"])
        assert json.dumps(p)


# ---------------------------------------------------------------------------
# Notion payloads
# ---------------------------------------------------------------------------

class TestNotionPayload:
    def test_page_with_parent_page_id(self):
        p = ConnectorPayloadBuilder.notion_page("Meeting notes", parent_page_id="page-abc")
        assert p["parent"]["type"] == "page_id"
        assert p["parent"]["page_id"] == "page-abc"

    def test_page_with_database_id(self):
        p = ConnectorPayloadBuilder.notion_page("Task", parent_database_id="db-xyz")
        assert p["parent"]["type"] == "database_id"
        assert p["parent"]["database_id"] == "db-xyz"

    def test_title_in_properties(self):
        p = ConnectorPayloadBuilder.notion_page("My page", parent_page_id="p1")
        title_prop = p["properties"]["title"]["title"][0]
        assert title_prop["type"] == "text"
        assert title_prop["text"]["content"] == "My page"

    def test_content_creates_paragraph_block(self):
        p = ConnectorPayloadBuilder.notion_page("T", parent_page_id="p1", content="Hello world")
        assert "children" in p
        assert p["children"][0]["type"] == "paragraph"

    def test_no_children_when_no_content(self):
        p = ConnectorPayloadBuilder.notion_page("T", parent_page_id="p1")
        assert "children" not in p

    def test_raises_without_parent(self):
        with pytest.raises(ValueError):
            ConnectorPayloadBuilder.notion_page("T")

    def test_payload_is_json_serializable(self):
        p = ConnectorPayloadBuilder.notion_page("T", parent_page_id="p", content="c")
        assert json.dumps(p)


# ---------------------------------------------------------------------------
# Teams payloads
# ---------------------------------------------------------------------------

class TestTeamsPayload:
    def test_messagecard_structure(self):
        p = ConnectorPayloadBuilder.teams("Alert", "Something happened")
        assert p["@type"] == "MessageCard"
        assert p["title"] == "Alert"
        assert p["sections"][0]["text"] == "Something happened"

    def test_theme_color_no_hash(self):
        p = ConnectorPayloadBuilder.teams("T", "B", theme_color="#FF0000")
        assert p["themeColor"] == "FF0000"

    def test_facts_in_section(self):
        facts = [{"name": "severity", "value": "high"}, {"name": "count", "value": "5"}]
        p = ConnectorPayloadBuilder.teams("T", "B", facts=facts)
        assert p["sections"][0]["facts"][0]["name"] == "severity"

    def test_adaptive_card_structure(self):
        p = ConnectorPayloadBuilder.teams_adaptive_card("Title", "Body text")
        assert p["type"] == "message"
        assert p["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"
        card = p["attachments"][0]["content"]
        assert card["type"] == "AdaptiveCard"

    def test_adaptive_card_with_facts(self):
        p = ConnectorPayloadBuilder.teams_adaptive_card(
            "T", "B", facts=[{"name": "k", "value": "v"}]
        )
        card_body = p["attachments"][0]["content"]["body"]
        fact_set = next(b for b in card_body if b["type"] == "FactSet")
        assert fact_set["facts"][0]["title"] == "k"

    def test_payload_is_json_serializable(self):
        p = ConnectorPayloadBuilder.teams("T", "B")
        assert json.dumps(p)


# ---------------------------------------------------------------------------
# EmailConnectorService
# ---------------------------------------------------------------------------

class TestEmailConnectorService:
    def _make_svc(self, **kwargs):
        defaults = {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "username": "user@example.com",
            "password": "secret",
            "use_tls": True,
            "default_from": "ninai@example.com",
        }
        defaults.update(kwargs)
        return EmailConnectorService(**defaults)

    def _make_msg(self, **kwargs):
        defaults = {
            "to": ["recipient@example.com"],
            "subject": "Test subject",
            "body": "Test body",
        }
        defaults.update(kwargs)
        return EmailMessage(**defaults)

    def test_no_recipients_returns_failed(self):
        svc = self._make_svc()
        msg = EmailMessage(to=[], subject="s", body="b")
        result = svc.send(msg)
        assert result.status == "failed"
        assert "recipient" in (result.error or "").lower()

    def test_success_path_with_mock_smtp(self):
        svc = self._make_svc()
        msg = self._make_msg()
        with mock.patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = mock.MagicMock()
            mock_server.sendmail.return_value = {}
            mock_smtp_cls.return_value.__enter__.return_value = mock_server
            result = svc.send(msg)
        assert result.status == "success"
        assert "recipient@example.com" in result.recipients_accepted

    def test_auth_failure_returns_failed(self):
        svc = self._make_svc()
        msg = self._make_msg()
        with mock.patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = mock.MagicMock()
            mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Auth failed")
            mock_smtp_cls.return_value.__enter__.return_value = mock_server
            result = svc.send(msg)
        assert result.status == "failed"
        assert "Auth" in (result.error or "")

    def test_connect_failure_returns_failed(self):
        import smtplib
        svc = self._make_svc()
        msg = self._make_msg()
        with mock.patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp_cls.side_effect = smtplib.SMTPConnectError(421, b"no connection")
            result = svc.send(msg)
        assert result.status == "failed"

    def test_partial_rejection(self):
        svc = self._make_svc()
        msg = self._make_msg(to=["good@ex.com", "bad@ex.com"])
        with mock.patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = mock.MagicMock()
            mock_server.sendmail.return_value = {"bad@ex.com": (550, b"User unknown")}
            mock_smtp_cls.return_value.__enter__.return_value = mock_server
            result = svc.send(msg)
        assert result.status == "success"
        assert "good@ex.com" in result.recipients_accepted
        assert "bad@ex.com" in result.recipients_rejected

    def test_html_body_sends_multipart_alternative(self):
        svc = self._make_svc()
        msg = self._make_msg(html_body="<b>Hello</b>")
        with mock.patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = mock.MagicMock()
            mock_server.sendmail.return_value = {}
            mock_smtp_cls.return_value.__enter__.return_value = mock_server
            result = svc.send(msg)
            # Check that sendmail was called with the MIME content
            call_args = mock_server.sendmail.call_args
            raw_msg = call_args[0][2]
            assert "multipart/alternative" in raw_msg
        assert result.status == "success"

    def test_cc_and_bcc_go_to_sendmail_recipients(self):
        svc = self._make_svc()
        msg = self._make_msg(to=["a@x.com"], cc=["b@x.com"], bcc=["c@x.com"])
        with mock.patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = mock.MagicMock()
            mock_server.sendmail.return_value = {}
            mock_smtp_cls.return_value.__enter__.return_value = mock_server
            svc.send(msg)
            recipients_arg = mock_server.sendmail.call_args[0][1]
            assert "a@x.com" in recipients_arg
            assert "b@x.com" in recipients_arg
            assert "c@x.com" in recipients_arg

    def test_ssl_uses_smtp_ssl_class(self):
        svc = self._make_svc(use_ssl=True, use_tls=False, smtp_port=465)
        msg = self._make_msg()
        with mock.patch("smtplib.SMTP_SSL") as mock_ssl_cls:
            mock_server = mock.MagicMock()
            mock_server.sendmail.return_value = {}
            mock_ssl_cls.return_value.__enter__.return_value = mock_server
            result = svc.send(msg)
        assert mock_ssl_cls.called
        assert result.status == "success"

    def test_starttls_called_when_use_tls(self):
        svc = self._make_svc(use_tls=True, use_ssl=False)
        msg = self._make_msg()
        with mock.patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = mock.MagicMock()
            mock_server.sendmail.return_value = {}
            mock_smtp_cls.return_value.__enter__.return_value = mock_server
            svc.send(msg)
            mock_server.starttls.assert_called_once()

    def test_default_from_used_when_message_has_none(self):
        svc = self._make_svc(default_from="ninai@example.com")
        msg = EmailMessage(to=["r@x.com"], subject="s", body="b", from_addr=None)
        with mock.patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = mock.MagicMock()
            mock_server.sendmail.return_value = {}
            mock_smtp_cls.return_value.__enter__.return_value = mock_server
            svc.send(msg)
            from_arg = mock_server.sendmail.call_args[0][0]
            assert from_arg == "ninai@example.com"

    def test_custom_from_overrides_default(self):
        svc = self._make_svc(default_from="default@x.com")
        msg = EmailMessage(to=["r@x.com"], subject="s", body="b", from_addr="custom@x.com")
        with mock.patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = mock.MagicMock()
            mock_server.sendmail.return_value = {}
            mock_smtp_cls.return_value.__enter__.return_value = mock_server
            svc.send(msg)
            from_arg = mock_server.sendmail.call_args[0][0]
            assert from_arg == "custom@x.com"
