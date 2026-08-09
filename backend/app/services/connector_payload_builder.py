"""Connector Payload Builder — Phase 88.

Builds correctly formatted request payloads for external systems.
ExternalConnectorService handles the HTTP dispatch; this module handles
the payload structure required by each target system.

Supported targets:
  slack       — Incoming Webhook (Block Kit or simple text)
  jira        — REST API v3 Create Issue
  github      — REST API Create Issue / Create Comment
  notion      — API v1 Create Page (in a parent page or database)
  teams       — Incoming Webhook (Adaptive Card / MessageCard)

Usage::

    from app.services.connector_payload_builder import ConnectorPayloadBuilder

    payload = ConnectorPayloadBuilder.slack(
        text="Memory alert: anomaly detected",
        channel="#alerts",
    )
    await connector.dispatch(action_type="slack", target_url=webhook_url, payload=payload)
"""

from __future__ import annotations

from typing import Any


class ConnectorPayloadBuilder:
    """Factory methods that produce the correct JSON payload for each connector."""

    # ------------------------------------------------------------------
    # Slack
    # ------------------------------------------------------------------

    @staticmethod
    def slack(
        text: str,
        *,
        channel: str | None = None,
        username: str = "Ninai",
        icon_emoji: str = ":brain:",
        blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a Slack Incoming Webhook payload.

        Pass ``blocks`` for rich Block Kit formatting; ``text`` is used as the
        fallback notification text in all cases.
        """
        payload: dict[str, Any] = {
            "text": str(text or ""),
            "username": str(username or "Ninai"),
            "icon_emoji": str(icon_emoji or ":brain:"),
        }
        if channel:
            payload["channel"] = str(channel)
        if blocks:
            payload["blocks"] = blocks
        return payload

    @staticmethod
    def slack_blocks(
        header: str,
        body: str,
        *,
        channel: str | None = None,
        footer: str | None = None,
    ) -> dict[str, Any]:
        """Build a structured Slack Block Kit message."""
        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": str(header)[:150], "emoji": True},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": str(body)[:2900]},
            },
        ]
        if footer:
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": str(footer)[:300]}],
            })
        return ConnectorPayloadBuilder.slack(
            text=header, channel=channel, blocks=blocks
        )

    # ------------------------------------------------------------------
    # Jira
    # ------------------------------------------------------------------

    @staticmethod
    def jira(
        summary: str,
        *,
        project_key: str,
        issue_type: str = "Task",
        description: str | None = None,
        priority: str | None = None,
        labels: list[str] | None = None,
        assignee_account_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a Jira REST API v3 Create Issue payload.

        The ``description`` field uses Atlassian Document Format (ADF) — a
        simplified paragraph node is built automatically from a plain string.
        """
        fields: dict[str, Any] = {
            "project": {"key": str(project_key)},
            "summary": str(summary)[:255],
            "issuetype": {"name": str(issue_type)},
        }
        if description:
            fields["description"] = {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": str(description)}],
                    }
                ],
            }
        if priority:
            fields["priority"] = {"name": str(priority)}
        if labels:
            fields["labels"] = [str(l) for l in labels]
        if assignee_account_id:
            fields["assignee"] = {"accountId": str(assignee_account_id)}
        return {"fields": fields}

    # ------------------------------------------------------------------
    # GitHub
    # ------------------------------------------------------------------

    @staticmethod
    def github_issue(
        title: str,
        *,
        body: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
        milestone: int | None = None,
    ) -> dict[str, Any]:
        """Build a GitHub REST API Create Issue payload.

        POST to: /repos/{owner}/{repo}/issues
        """
        payload: dict[str, Any] = {"title": str(title)[:256]}
        if body:
            payload["body"] = str(body)
        if labels:
            payload["labels"] = [str(l) for l in labels]
        if assignees:
            payload["assignees"] = [str(a) for a in assignees]
        if milestone is not None:
            payload["milestone"] = int(milestone)
        return payload

    @staticmethod
    def github_comment(body: str) -> dict[str, Any]:
        """Build a GitHub Create Issue Comment payload.

        POST to: /repos/{owner}/{repo}/issues/{issue_number}/comments
        """
        return {"body": str(body)}

    # ------------------------------------------------------------------
    # Notion
    # ------------------------------------------------------------------

    @staticmethod
    def notion_page(
        title: str,
        *,
        parent_page_id: str | None = None,
        parent_database_id: str | None = None,
        content: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a Notion API v1 Create Page payload.

        Provide either ``parent_page_id`` (creates a sub-page) or
        ``parent_database_id`` (creates a database entry).
        """
        if not parent_page_id and not parent_database_id:
            raise ValueError("Provide either parent_page_id or parent_database_id")

        if parent_page_id:
            parent = {"type": "page_id", "page_id": str(parent_page_id)}
        else:
            parent = {"type": "database_id", "database_id": str(parent_database_id)}

        page_properties: dict[str, Any] = dict(properties or {})
        page_properties["title"] = {
            "title": [{"type": "text", "text": {"content": str(title)[:2000]}}]
        }

        payload: dict[str, Any] = {"parent": parent, "properties": page_properties}

        if content:
            payload["children"] = [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"type": "text", "text": {"content": str(content)[:2000]}}
                        ]
                    },
                }
            ]
        return payload

    # ------------------------------------------------------------------
    # Microsoft Teams
    # ------------------------------------------------------------------

    @staticmethod
    def teams(
        title: str,
        text: str,
        *,
        theme_color: str = "0078D7",
        facts: list[dict[str, str]] | None = None,
        actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a Microsoft Teams Incoming Webhook MessageCard payload.

        ``theme_color`` is a hex color string (no ``#`` prefix).
        ``facts`` is a list of ``{"name": str, "value": str}`` dicts.
        ``actions`` follows the MessageCard OpenUri/HttpPOST action schema.
        """
        sections: list[dict[str, Any]] = []
        section: dict[str, Any] = {"text": str(text)[:4000]}
        if facts:
            section["facts"] = [
                {"name": str(f.get("name", "")), "value": str(f.get("value", ""))}
                for f in facts
            ]
        sections.append(section)

        payload: dict[str, Any] = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "themeColor": str(theme_color).lstrip("#"),
            "summary": str(title)[:256],
            "title": str(title)[:256],
            "sections": sections,
        }
        if actions:
            payload["potentialAction"] = actions
        return payload

    @staticmethod
    def teams_adaptive_card(
        title: str,
        body_text: str,
        *,
        facts: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Build a Teams Adaptive Card payload (newer format, attachment-wrapped)."""
        body: list[dict[str, Any]] = [
            {"type": "TextBlock", "size": "Medium", "weight": "Bolder", "text": str(title)[:256]},
            {"type": "TextBlock", "text": str(body_text)[:4000], "wrap": True},
        ]
        if facts:
            body.append({
                "type": "FactSet",
                "facts": [
                    {"title": str(f.get("name", "")), "value": str(f.get("value", ""))}
                    for f in facts
                ],
            })
        return {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": body,
                    },
                }
            ],
        }
