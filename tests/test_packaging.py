"""Build and clean-target installation checks for the distributable package."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_VERSION = "0.1.0"


def test_wheel_builds_and_imports_outside_the_source_tree(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    project_copy = tmp_path / "project"
    wheel_directory = tmp_path / "wheel"
    install_directory = tmp_path / "installed"
    wheel_directory.mkdir()
    project_copy.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repository_root / filename, project_copy / filename)
    shutil.copytree(
        repository_root / "src",
        project_copy / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(project_copy),
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
            "--wheel-dir",
            str(wheel_directory),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(
        wheel_directory.glob(f"historical_asset_risk_engine-{EXPECTED_VERSION}-*.whl")
    )
    assert len(wheels) == 1

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(wheels[0]),
            "--no-deps",
            "--no-cache-dir",
            "--target",
            str(install_directory),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(install_directory)
    import_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from importlib.metadata import distribution; "
                "import historical_asset_risk; "
                "from historical_asset_risk.artifacts import load_artifact; "
                "from historical_asset_risk.contracts import "
                "Cash, Instrument, Position; "
                "from historical_asset_risk.portfolio import "
                "validate_portfolio_snapshot, value_portfolio; "
                "dist = distribution('historical-asset-risk-engine'); "
                "entry_point = next(ep for ep in dist.entry_points "
                "if ep.group == 'console_scripts' "
                "and ep.name == 'historical-asset-risk'); "
                "assert historical_asset_risk.__version__ == dist.version "
                f"== '{EXPECTED_VERSION}'; "
                "assert all((Cash, Instrument, Position, load_artifact, "
                "validate_portfolio_snapshot, value_portfolio)); "
                "assert callable(entry_point.load())"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_result.returncode == 0, import_result.stderr

    csv_path = tmp_path / "adjusted_prices.csv"
    csv_path.write_text(
        "date,ticker,adjusted_close\n"
        "2024-01-02,AAA,100\n"
        "2024-01-03,AAA,101\n"
        "2024-01-04,AAA,102\n"
        "2024-01-05,AAA,103\n"
        "2024-01-08,AAA,105\n"
        "2024-01-02,BBB,50\n"
        "2024-01-03,BBB,51\n"
        "2024-01-04,BBB,50.5\n"
        "2024-01-05,BBB,52\n"
        "2024-01-08,BBB,53\n",
        encoding="utf-8",
    )
    cli_output_directory = tmp_path / "cli_outputs"
    cli_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "historical_asset_risk",
            "--provider",
            "csv",
            "--csv-path",
            str(csv_path),
            "--tickers",
            "AAA",
            "BBB",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-02-01",
            "--rolling-window",
            "3",
            "--output-dir",
            str(cli_output_directory),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli_result.returncode == 0, cli_result.stdout + cli_result.stderr
    assert (cli_output_directory / "return_summary.csv").is_file()
    assert (cli_output_directory / "data_quality_report.json").is_file()
    assert (cli_output_directory / "run_manifest.json").is_file()

    consumer_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from historical_asset_risk.artifacts import load_artifact; "
                f"root = Path({str(cli_output_directory)!r}); "
                "returns = load_artifact(root / 'simple_returns.csv', "
                "root / 'run_manifest.json'); "
                "prices = load_artifact(root / 'adjusted_prices.csv', "
                "root / 'run_manifest.json'); "
                "assert list(returns.columns) == ['AAA', 'BBB']; "
                "assert returns.tail(3).mean().notna().all(); "
                "slice_ = prices.loc[:prices.index[-2], ['BBB', 'AAA']]; "
                "assert slice_.shape == (4, 2)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert consumer_result.returncode == 0, consumer_result.stderr

    installed_tests = tmp_path / "installed_tests"
    nested_pytest_temp = tmp_path / "nested_pytest_temp"
    installed_tests.mkdir()
    for filename in (
        "test_config.py",
        "test_data_loader.py",
        "test_dependence.py",
        "test_metrics.py",
        "test_providers.py",
        "test_cli.py",
        "test_artifacts.py",
        "test_portfolio.py",
        "test_portfolio_cli.py",
    ):
        shutil.copy2(repository_root / "tests" / filename, installed_tests / filename)
    test_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(nested_pytest_temp),
            str(installed_tests),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert test_result.returncode == 0, test_result.stdout + test_result.stderr
