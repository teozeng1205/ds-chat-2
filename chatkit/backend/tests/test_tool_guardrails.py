from pathlib import Path

from app.tools.guardrails import (
    classify_aws_command,
    classify_path_write,
    classify_shell_command,
)


def test_shell_allows_read_only_commands() -> None:
    assert classify_shell_command("pwd").decision == "allow"
    assert classify_shell_command("rg -n 'foo' app tests").decision == "allow"
    assert classify_shell_command("git status --short").decision == "allow"


def test_shell_denies_destructive_commands() -> None:
    assert classify_shell_command("rm -rf /").decision == "deny"
    assert classify_shell_command("git reset --hard HEAD").decision == "deny"
    assert classify_shell_command("git clean -fd").decision == "deny"
    assert classify_shell_command("git push --force origin main").decision == "deny"
    assert classify_shell_command("shutdown now").decision == "deny"


def test_shell_allows_read_only_search_for_destructive_words() -> None:
    command = "grep -n 'pullFromQueue\\|preRunChecks\\|generateSchedule\\|shutdown()' PEPipeline.java"
    assert classify_shell_command(command).decision == "allow"


def test_shell_requires_approval_for_writes_and_runs() -> None:
    assert classify_shell_command("touch /tmp/example.txt").decision == "approval_required"
    assert classify_shell_command("python -m pytest").decision == "allow"
    assert classify_shell_command("npm run build").decision == "approval_required"
    assert classify_shell_command("pip install pytest").decision == "approval_required"
    assert classify_shell_command("git commit -m change").decision == "approval_required"


def test_aws_read_only_commands_are_allowed() -> None:
    for command in [
        "aws sts get-caller-identity",
        "aws stepfunctions list-state-machines",
        "aws lambda get-function --function-name fn",
        "aws logs start-query --log-group-name /aws/lambda/fn",
        "aws logs get-query-results --query-id abc",
        "aws s3 ls s3://bucket/prefix/",
    ]:
        assert classify_shell_command(command).decision == "allow", command


def test_aws_mutation_and_run_commands_require_approval() -> None:
    for command in [
        "aws ecs run-task --cluster c --task-definition td",
        "aws lambda invoke --function-name fn /tmp/out.json",
        "aws stepfunctions start-execution --state-machine-arn arn --input '{}'",
        "aws sqs send-message --queue-url q --message-body hi",
        "aws sns publish --topic-arn t --message hi",
    ]:
        assert classify_shell_command(command).decision == "approval_required", command


def test_aws_profile_switching_is_denied() -> None:
    assert classify_shell_command("aws --profile 3VPROD s3 ls").decision == "deny"
    assert classify_shell_command("assume 3VPROD").decision == "deny"
    assert classify_aws_command(["aws", "sts", "assume-role", "--role-arn", "arn"]).decision == "deny"


def test_sensitive_write_paths_are_denied() -> None:
    assert classify_path_write(Path("chatkit/.env"), operation="write").decision == "deny"
    assert classify_path_write(Path("safe/output.txt"), operation="write").decision == "allow"
