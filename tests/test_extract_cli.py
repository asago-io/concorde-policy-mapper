import tempfile

from typer.testing import CliRunner

from concorde_policy_mapper.cli import app

runner = CliRunner()


def test_extract_command_exists():
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "Extract risks" in result.stdout or "extract" in result.stdout.lower()


def test_extract_missing_base_url():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("test policy")
        f.flush()
        result = runner.invoke(
            app,
            [
                "extract",
                f.name,
                "-o",
                "/tmp/test-output",
                "--nexus-base-dir",
                "/tmp/nexus",
            ],
            env={"POLICY_MAPPER_BASE_URL": "", "POLICY_MAPPER_MODEL": ""},
        )
    assert result.exit_code != 0


def test_extract_nonexistent_file():
    result = runner.invoke(
        app,
        [
            "extract",
            "/nonexistent/policy.pdf",
            "-o",
            "/tmp/test-output",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "test",
            "--nexus-base-dir",
            "/tmp/nexus",
        ],
    )
    assert result.exit_code != 0


def test_eval_command_exists():
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "Evaluate" in result.stdout or "eval" in result.stdout.lower()
