"""Tests for the CLI commands.

Uses click's CliRunner and mocks httpx.request to avoid needing a running daemon.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from smriti.cli import main


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = ""
    return resp


class TestStatus:
    def test_healthy(self) -> None:
        runner = CliRunner()
        health_resp = _mock_response(json_data={"status": "ok", "uptime_seconds": 3720.0})
        stats_resp = _mock_response(json_data={
            "batches_processed": 10,
            "events_consumed": 50,
            "events_filtered": 35,
            "memories_created": 20,
            "uptime_seconds": 3720.0,
        })

        with patch("smriti.cli.httpx.request", side_effect=[health_resp, stats_resp]):
            result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "healthy" in result.output
        assert "1h 2m" in result.output
        assert "events consumed: 50" in result.output
        assert "memories created: 20" in result.output

    def test_unhealthy(self) -> None:
        runner = CliRunner()
        resp = _mock_response(status_code=503)

        with patch("smriti.cli.httpx.request", return_value=resp):
            result = runner.invoke(main, ["status"])

        assert result.exit_code == 1
        assert "unhealthy" in result.output

    def test_connection_error(self) -> None:
        runner = CliRunner()

        import httpx
        with patch("smriti.cli.httpx.request", side_effect=httpx.ConnectError("refused")):
            result = runner.invoke(main, ["status"])

        assert result.exit_code == 1
        assert "cannot connect" in result.output


class TestSearch:
    def test_returns_results(self) -> None:
        runner = CliRunner()
        resp = _mock_response(json_data={
            "results": [
                {
                    "content": "Debugged CORS issue in auth service",
                    "tier": "episodic",
                    "importance": 0.91,
                    "score": 0.85,
                    "created_at": "2026-03-08T12:00:00+00:00",
                },
            ]
        })

        with patch("smriti.cli.httpx.request", return_value=resp):
            result = runner.invoke(main, ["search", "CORS debugging"])

        assert result.exit_code == 0
        assert "Debugged CORS" in result.output
        assert "episodic" in result.output
        assert "0.91" in result.output

    def test_no_results(self) -> None:
        runner = CliRunner()
        resp = _mock_response(json_data={"results": []})

        with patch("smriti.cli.httpx.request", return_value=resp):
            result = runner.invoke(main, ["search", "nonexistent topic"])

        assert result.exit_code == 0
        assert "No memories found" in result.output

    def test_passes_top_k(self) -> None:
        runner = CliRunner()
        resp = _mock_response(json_data={"results": []})

        with patch("smriti.cli.httpx.request", return_value=resp) as mock_req:
            runner.invoke(main, ["search", "-k", "5", "test query"])

        call_kwargs = mock_req.call_args
        assert call_kwargs.kwargs["json"]["top_k"] == 5

    def test_passes_tier_filter(self) -> None:
        runner = CliRunner()
        resp = _mock_response(json_data={"results": []})

        with patch("smriti.cli.httpx.request", return_value=resp) as mock_req:
            runner.invoke(main, ["search", "--tier", "semantic", "test query"])

        call_kwargs = mock_req.call_args
        assert call_kwargs.kwargs["json"]["tier"] == "semantic"

    def test_search_api_error(self) -> None:
        runner = CliRunner()
        resp = _mock_response(status_code=502)
        resp.text = "Embedding failed"

        with patch("smriti.cli.httpx.request", return_value=resp):
            result = runner.invoke(main, ["search", "test"])

        assert result.exit_code == 1
        assert "Search failed" in result.output


class TestDelete:
    def test_delete_with_confirmation(self) -> None:
        runner = CliRunner()
        resp = _mock_response(json_data={"deleted": True, "id": "abc-123"})

        with patch("smriti.cli.httpx.request", return_value=resp):
            result = runner.invoke(main, ["delete", "abc-123"], input="y\n")

        assert result.exit_code == 0
        assert "Deleted memory abc-123" in result.output

    def test_delete_skips_confirmation_with_yes_flag(self) -> None:
        runner = CliRunner()
        resp = _mock_response(json_data={"deleted": True, "id": "abc-123"})

        with patch("smriti.cli.httpx.request", return_value=resp):
            result = runner.invoke(main, ["delete", "--yes", "abc-123"])

        assert result.exit_code == 0
        assert "Deleted memory abc-123" in result.output

    def test_delete_aborted(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["delete", "abc-123"], input="n\n")

        assert result.exit_code == 1
        assert "Aborted" in result.output

    def test_delete_not_found(self) -> None:
        runner = CliRunner()
        resp = _mock_response(status_code=404)

        with patch("smriti.cli.httpx.request", return_value=resp):
            result = runner.invoke(main, ["delete", "--yes", "abc-123"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_delete_api_error(self) -> None:
        runner = CliRunner()
        resp = _mock_response(status_code=500)
        resp.text = "Internal error"

        with patch("smriti.cli.httpx.request", return_value=resp):
            result = runner.invoke(main, ["delete", "--yes", "abc-123"])

        assert result.exit_code == 1
        assert "Delete failed" in result.output


class TestCustomUrl:
    def test_url_option(self) -> None:
        runner = CliRunner()
        health_resp = _mock_response(json_data={"status": "ok", "uptime_seconds": 60.0})
        stats_resp = _mock_response(json_data={
            "batches_processed": 0,
            "events_consumed": 0,
            "events_filtered": 0,
            "memories_created": 0,
            "uptime_seconds": 60.0,
        })

        with patch("smriti.cli.httpx.request", side_effect=[health_resp, stats_resp]) as mock_req:
            result = runner.invoke(main, ["--url", "http://remote:9898", "status"])

        assert result.exit_code == 0
        first_call_url = mock_req.call_args_list[0].args[1]
        assert "remote:9898" in first_call_url
