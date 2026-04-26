from app.core.security import create_access_token, create_refresh_token, verify_token


def test_access_token_includes_group_team_claims():
    token = create_access_token(
        user_id="u1",
        org_id=None,
        roles=["user"],
        group_id="g1",
        team_id="t1",
    )

    data = verify_token(token, token_type="access")

    assert data is not None
    assert data.user_id == "u1"
    assert data.org_id is None
    assert data.group_id == "g1"
    assert data.team_id == "t1"
    assert data.roles == ["user"]


def test_refresh_token_preserves_group_team_claims():
    token = create_refresh_token(
        user_id="u2",
        org_id="org-1",
        roles=["org_admin"],
        group_id="g2",
        team_id="t2",
    )

    data = verify_token(token, token_type="refresh")

    assert data is not None
    assert data.user_id == "u2"
    assert data.org_id == "org-1"
    assert data.group_id == "g2"
    assert data.team_id == "t2"
    assert data.roles == ["org_admin"]
