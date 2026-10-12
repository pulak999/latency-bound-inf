import json

from harness.engines.llama_server import encode_request, parse_stream_line
from harness.synthetic import synthetic_png


def test_parse_stream_line_skips_blank_lines_and_end_marker():
    assert parse_stream_line("") is None
    assert parse_stream_line("data: [DONE]") is None
    chunk = parse_stream_line('data: {"choices": [{"delta": {"content": "Field"}}]}')
    assert chunk["choices"][0]["delta"]["content"] == "Field"


def test_request_forces_fixed_output_and_no_prompt_cache():
    body = json.loads(encode_request(synthetic_png(0, size=8), "label?", n_output_tokens=4))
    assert body["max_tokens"] == 4
    assert body["ignore_eos"] is True
    assert body["temperature"] == 0
    assert body["cache_prompt"] is False
    image_part, text_part = body["messages"][0]["content"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert text_part["text"] == "label?"


def test_synthetic_images_are_reproducible_and_distinct():
    assert synthetic_png(1, size=16) == synthetic_png(1, size=16)
    assert synthetic_png(1, size=16) != synthetic_png(2, size=16)
