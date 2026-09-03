from __future__ import annotations

from types import SimpleNamespace
import unittest

from engine import ModelEngine, ProviderConfig, result_model_label


class Recorder:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.response


def responses_client(response):
    recorder = Recorder(response)
    return SimpleNamespace(responses=recorder), recorder


def chat_client(response):
    recorder = Recorder(response)
    return SimpleNamespace(chat=SimpleNamespace(completions=recorder)), recorder


def config(provider: str, model: str | None = None) -> ProviderConfig:
    return ProviderConfig.from_env(
        provider,
        model=model,
        environ={},
        require_api_key=False,
    )


class ProviderConfigTests(unittest.TestCase):
    def test_profiles_resolve_current_models_and_environment(self):
        cases = {
            "qwen": "Qwen/Qwen3.8-27B:deepinfra",
            "glm": "zai-org/GLM-5.3:deepinfra",
            "minimax": "MiniMaxAI/MiniMax-M3:deepinfra",
            "kimi": "moonshotai/Kimi-K3:deepinfra",
            "deepseek": "deepseek-ai/DeepSeek-V4-Pro:deepinfra",
            "huggingface": "Qwen/Qwen3.8-27B:deepinfra",
        }
        for provider, model in cases.items():
            with self.subTest(provider=provider):
                resolved = config(provider)
                self.assertEqual(model, resolved.model)
                self.assertEqual("HF_TOKEN", resolved.api_key_env)
                self.assertEqual("https://router.huggingface.co/v1", resolved.base_url)
                self.assertEqual("responses", resolved.profile.api_style)
                self.assertEqual("max_output_tokens", resolved.profile.max_tokens_parameter)

    def test_missing_key_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            ProviderConfig.from_env("deepseek", environ={})

    def test_custom_provider_requires_base_url_and_model(self):
        with self.assertRaisesRegex(ValueError, "explicit model"):
            config("custom")


class RequestAndResponseTests(unittest.TestCase):
    def test_openai_keeps_legacy_sampling_but_omits_it_for_reasoning_models(self):
        response = SimpleNamespace(output_text="answer", output=[])
        client, recorder = responses_client(response)
        ModelEngine(config("openai", "gpt-4o"), client=client).generate("puzzle")
        self.assertEqual(0, recorder.request["temperature"])
        self.assertEqual(0, recorder.request["top_p"])

        client, recorder = responses_client(response)
        ModelEngine(config("openai", "gpt-5.4"), client=client).generate("puzzle")
        self.assertNotIn("temperature", recorder.request)
        self.assertNotIn("top_p", recorder.request)

    def test_qwen_responses_reasoning_and_normalization(self):
        response = SimpleNamespace(
            output_text="[b c d e]",
            output=[SimpleNamespace(type="reasoning", summary=[{"text": "summary"}])],
        )
        client, recorder = responses_client(response)
        result = ModelEngine(config("qwen"), client=client).generate(
            "puzzle", effort="xhigh", max_output_tokens=2048
        )
        self.assertEqual("[b c d e]", result.text)
        self.assertEqual("summary", result.reasoning)
        self.assertEqual({"effort": "xhigh"}, recorder.request["reasoning"])
        self.assertEqual(2048, recorder.request["max_output_tokens"])
        self.assertNotIn("temperature", recorder.request)

    def test_glm_uses_hf_responses_contract(self):
        response = SimpleNamespace(
            output_text="answer",
            output=[SimpleNamespace(type="reasoning", summary=[{"text": "why"}])],
        )
        client, recorder = responses_client(response)
        result = ModelEngine(config("glm"), client=client).generate(
            "puzzle", effort="max", max_output_tokens=4096
        )
        self.assertEqual("why", result.reasoning)
        self.assertEqual({"effort": "max"}, recorder.request["reasoning"])
        self.assertEqual(4096, recorder.request["max_output_tokens"])
        self.assertNotIn("extra_body", recorder.request)

    def test_minimax_uses_hf_responses_without_direct_api_fields(self):
        response = SimpleNamespace(
            output_text="answer",
            output=[SimpleNamespace(type="reasoning", summary=[{"text": "first"}, {"text": "second"}])],
        )
        client, recorder = responses_client(response)
        result = ModelEngine(config("minimax"), client=client).generate(
            "puzzle", max_output_tokens=2048
        )
        self.assertEqual("first\nsecond", result.reasoning)
        self.assertEqual(2048, recorder.request["max_output_tokens"])
        self.assertNotIn("extra_body", recorder.request)
        self.assertNotIn("temperature", recorder.request)

    def test_minimax_rejects_undocumented_effort_control(self):
        response = {"choices": [{"message": {"content": "answer"}}]}
        client, _ = chat_client(response)
        with self.assertRaisesRegex(ValueError, "no documented --effort mapping"):
            ModelEngine(config("minimax"), client=client).generate("puzzle", effort="high")

    def test_kimi_uses_hf_output_limit_and_fixed_sampling(self):
        response = SimpleNamespace(output_text="answer", output=[])
        client, recorder = responses_client(response)
        ModelEngine(config("kimi"), client=client).generate(
            "puzzle", effort="max", max_output_tokens=8192
        )
        self.assertEqual({"effort": "max"}, recorder.request["reasoning"])
        self.assertEqual(8192, recorder.request["max_output_tokens"])
        self.assertNotIn("temperature", recorder.request)
        self.assertNotIn("top_p", recorder.request)

    def test_deepseek_uses_hf_responses_without_direct_api_fields(self):
        response = SimpleNamespace(
            output_text="answer",
            output=[SimpleNamespace(type="reasoning", summary=[{"text": "why"}])],
        )
        client, recorder = responses_client(response)
        result = ModelEngine(config("deepseek"), client=client).generate("puzzle", effort="high")
        self.assertEqual("why", result.reasoning)
        self.assertEqual({"effort": "high"}, recorder.request["reasoning"])
        self.assertNotIn("extra_body", recorder.request)

    def test_huggingface_model_id_is_safe_in_result_path(self):
        self.assertEqual(
            "Qwen__Qwen3.8-27B__deepinfra_xhigh",
            result_model_label("huggingface", "Qwen/Qwen3.8-27B:deepinfra", "xhigh"),
        )

    def test_huggingface_uses_responses_reasoning_contract(self):
        response = SimpleNamespace(output_text="answer", output=[])
        client, recorder = responses_client(response)
        ModelEngine(config("huggingface"), client=client).generate("puzzle", effort="high")
        self.assertEqual({"effort": "high"}, recorder.request["reasoning"])

    def test_kimi_rejects_unsupported_effort(self):
        response = SimpleNamespace(output_text="answer", output=[])
        client, _ = responses_client(response)
        with self.assertRaisesRegex(ValueError, "Unsupported reasoning effort"):
            ModelEngine(config("kimi"), client=client).generate("puzzle", effort="medium")

    def test_custom_chat_response_normalization_is_still_available(self):
        response = {
            "choices": [
                {"message": {"content": "answer", "reasoning_content": "why"}}
            ]
        }
        client, _ = chat_client(response)
        custom = ProviderConfig.from_env(
            "custom",
            model="local-model",
            base_url="http://localhost:8000/v1",
            environ={},
            require_api_key=False,
        )
        result = ModelEngine(custom, client=client).generate("puzzle")
        self.assertEqual("answer", result.text)
        self.assertEqual("why", result.reasoning)


if __name__ == "__main__":
    unittest.main()
