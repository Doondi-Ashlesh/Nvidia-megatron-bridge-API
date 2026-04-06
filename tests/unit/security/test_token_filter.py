"""Security tests — TokenRedactFilter must scrub sensitive values from all log records."""

from __future__ import annotations

import logging

from app.security.token_filter import TokenRedactFilter, _redact


class TestRedactFunction:
    def test_hf_token_redacted(self):
        assert "***REDACTED***" in _redact("hf_token=hf_ABCD1234xyz")

    def test_hf_token_colon_form_redacted(self):
        assert "***REDACTED***" in _redact("hf_token: hf_ABCD1234xyz")

    def test_api_key_redacted(self):
        assert "***REDACTED***" in _redact("api_key=sk-super-secret")

    def test_password_redacted(self):
        assert "***REDACTED***" in _redact("password=hunter2")

    def test_bearer_token_redacted(self):
        assert "***REDACTED***" in _redact("Authorization: Bearer eyJhbGciOiJ....")

    def test_safe_message_unchanged(self):
        msg = "Training step 100/1000, loss=1.234"
        assert _redact(msg) == msg

    def test_raw_token_value_not_present_after_redact(self):
        result = _redact("hf_token=hf_MYSECRET999")
        assert "hf_MYSECRET999" not in result

    def test_multiple_secrets_in_one_message(self):
        msg = "hf_token=abc123 api_key=xyz789"
        result = _redact(msg)
        assert "abc123" not in result
        assert "xyz789" not in result


class TestTokenRedactFilter:
    def _make_record(self, msg: str) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg=msg,
            args=(), exc_info=None,
        )
        return record

    def test_filter_returns_true(self):
        f = TokenRedactFilter()
        record = self._make_record("hello world")
        assert f.filter(record) is True

    def test_filter_scrubs_hf_token_in_msg(self):
        f = TokenRedactFilter()
        record = self._make_record("using hf_token=hf_SECRET123 to download model")
        f.filter(record)
        assert "hf_SECRET123" not in record.msg

    def test_filter_scrubs_tuple_args(self):
        f = TokenRedactFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg="value is %s",
            args=("hf_token=hf_SECRET123",),
            exc_info=None,
        )
        f.filter(record)
        assert "hf_SECRET123" not in str(record.args)

    def test_filter_scrubs_dict_args(self):
        f = TokenRedactFilter()
        # Build LogRecord manually and set args after construction to avoid
        # a Python 3.12 KeyError when args is a dict with non-int keys.
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg="%(token)s",
            args=None,
            exc_info=None,
        )
        record.args = {"token": "hf_token=hf_SECRET123"}
        f.filter(record)
        assert "hf_SECRET123" not in str(record.args)

    def test_filter_preserves_safe_log_message(self):
        f = TokenRedactFilter()
        msg = "Training step 500/1000 loss=2.345"
        record = self._make_record(msg)
        f.filter(record)
        assert record.msg == msg
