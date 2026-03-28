"""Tests for proxy skill injection and stripping."""

from clawloop.proxy_skills import SENTINEL, inject_skills, strip_skills


class TestInjectSkills:
    """inject_skills prepends a system message with sentinel + skills text."""

    def test_injects_leading_system_message(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = inject_skills(messages, "You are a helpful assistant.")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"].startswith(SENTINEL)
        assert "You are a helpful assistant." in result[0]["content"]
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_no_injection_when_empty(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = inject_skills(messages, "")
        assert result == messages

    def test_no_injection_when_whitespace_only(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = inject_skills(messages, "   ")
        assert result == messages

    def test_idempotent_replaces_existing(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        first = inject_skills(messages, "Skill v1")
        second = inject_skills(first, "Skill v2")
        # Should have exactly one skills message, not two
        sentinel_msgs = [m for m in second if SENTINEL in m.get("content", "")]
        assert len(sentinel_msgs) == 1
        assert "Skill v2" in sentinel_msgs[0]["content"]
        assert "Skill v1" not in sentinel_msgs[0]["content"]
        # Total length: 1 skills + 1 user
        assert len(second) == 2

    def test_preserves_other_system_messages(self) -> None:
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Hello"},
        ]
        result = inject_skills(messages, "Extra skill")
        # Skills message + original system + user
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert SENTINEL in result[0]["content"]
        assert result[1] == {"role": "system", "content": "You are a bot."}

    def test_does_not_mutate_input(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        original = list(messages)
        inject_skills(messages, "Skill text")
        assert messages == original


class TestStripSkills:
    """strip_skills removes any message containing the sentinel."""

    def test_strips_injected_message(self) -> None:
        messages = [
            {"role": "system", "content": f"{SENTINEL}\nSome skills"},
            {"role": "user", "content": "Hello"},
        ]
        result = strip_skills(messages)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_noop_when_no_skills_present(self) -> None:
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Hello"},
        ]
        result = strip_skills(messages)
        assert result == messages

    def test_strips_sentinel_in_any_position(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": f"{SENTINEL}\nSkills"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = strip_skills(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_does_not_mutate_input(self) -> None:
        messages = [
            {"role": "system", "content": f"{SENTINEL}\nSkills"},
            {"role": "user", "content": "Hello"},
        ]
        original = list(messages)
        strip_skills(messages)
        assert messages == original

    def test_handles_missing_content_key(self) -> None:
        """Messages without 'content' should be preserved, not crash."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "user", "content": "Hello"},
        ]
        result = strip_skills(messages)
        assert result == messages


class TestRoundTrip:
    """inject then strip should return original messages."""

    def test_inject_then_strip_returns_original(self) -> None:
        messages = [
            {"role": "system", "content": "Base system prompt."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        injected = inject_skills(messages, "Some playbook skills")
        stripped = strip_skills(injected)
        assert stripped == messages
