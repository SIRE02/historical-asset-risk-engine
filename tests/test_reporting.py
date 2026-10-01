from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from historical_asset_risk import reporting


def test_git_commit_is_resolved_from_the_package_repository(
    monkeypatch,
) -> None:
    repository_root = Path("historical-source-root")
    completed = Mock(stdout="abc123\n")
    run = Mock(return_value=completed)
    monkeypatch.setattr(reporting, "_installed_vcs_commit", lambda: None)
    monkeypatch.setattr(reporting, "_source_repository_root", lambda: repository_root)
    monkeypatch.setattr(reporting.subprocess, "run", run)

    assert reporting._git_commit() == "abc123"
    run.assert_called_once_with(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        cwd=repository_root,
        text=True,
        timeout=5,
    )


def test_installed_vcs_commit_avoids_a_git_subprocess(monkeypatch) -> None:
    run = Mock()
    monkeypatch.setattr(reporting, "_installed_vcs_commit", lambda: "def456")
    monkeypatch.setattr(reporting.subprocess, "run", run)

    assert reporting._git_commit() == "def456"
    run.assert_not_called()


def test_worktree_dirty_reports_tracked_changes_only(monkeypatch) -> None:
    repository_root = Path("historical-source-root")
    monkeypatch.setattr(reporting, "_installed_vcs_commit", lambda: None)
    monkeypatch.setattr(reporting, "_source_repository_root", lambda: repository_root)

    run = Mock(return_value=Mock(stdout=" M src/historical_asset_risk/cli.py\n"))
    monkeypatch.setattr(reporting.subprocess, "run", run)
    assert reporting._git_worktree_dirty() is True
    run.assert_called_once_with(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        cwd=repository_root,
        text=True,
        timeout=5,
    )

    monkeypatch.setattr(reporting.subprocess, "run", Mock(return_value=Mock(stdout="")))
    assert reporting._git_worktree_dirty() is False

    failing = Mock(side_effect=OSError("git not installed"))
    monkeypatch.setattr(reporting.subprocess, "run", failing)
    assert reporting._git_worktree_dirty() is None


def test_installed_copy_reports_dirty_state_as_unknown(monkeypatch) -> None:
    run = Mock()
    monkeypatch.setattr(reporting, "_installed_vcs_commit", lambda: "def456")
    monkeypatch.setattr(reporting.subprocess, "run", run)

    assert reporting._git_worktree_dirty() is None
    run.assert_not_called()
