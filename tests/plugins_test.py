# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for named provider discovery and override precedence."""

from importlib import metadata
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from langextract import plugins
from langextract.core import base_model


class OverrideProvider(base_model.BaseLanguageModel):
  """Local provider that can replace a built-in entry point."""

  def infer(self, batch_prompts, **unused_kwargs):
    return [[] for _ in batch_prompts]


class ProviderDiscoveryTest(parameterized.TestCase):
  """Exercise precedence through package metadata and class loading."""

  def setUp(self):
    super().setUp()
    clear_discovery = plugins._discovered.cache_clear  # pylint: disable=protected-access
    clear_discovery()
    plugins.get_provider_class.cache_clear()
    self.addCleanup(clear_discovery)
    self.addCleanup(plugins.get_provider_class.cache_clear)
    self.plugin_spec = f"{__name__}:OverrideProvider"
    entry_points = metadata.EntryPoints(
        metadata.EntryPoint(
            name=name,
            value=self.plugin_spec,
            group="langextract.providers",
        )
        for name in ("gemini", "ollama", "openai", "custom")
    )
    self.enter_context(
        mock.patch.object(metadata, "entry_points", return_value=entry_points)
    )

  @parameterized.named_parameters(
      ("gemini", "gemini", "langextract.providers.gemini:GeminiLanguageModel"),
      ("ollama", "ollama", "langextract.providers.ollama:OllamaLanguageModel"),
      ("openai", "openai", "langextract.providers.openai:OpenAILanguageModel"),
  )
  def test_builtins_take_precedence_by_default(self, name, expected_spec):
    self.assertEqual(plugins.available_providers()[name], expected_spec)
    self.assertEqual(
        plugins.available_providers(allow_override=False)[name], expected_spec
    )

  @parameterized.named_parameters(
      ("gemini", "gemini"),
      ("ollama", "ollama"),
      ("openai", "openai"),
  )
  def test_allow_override_selects_plugin(self, name):
    self.assertEqual(
        plugins.available_providers(allow_override=True)[name], self.plugin_spec
    )
    self.assertIs(
        plugins.get_provider_class(name, allow_override=True), OverrideProvider
    )

  @parameterized.product(
      allow_override=(False, True), include_optional=(False, True)
  )
  def test_nonconflicting_plugins_remain_available(
      self, allow_override, include_optional
  ):
    self.assertEqual(
        plugins.available_providers(allow_override, include_optional)["custom"],
        self.plugin_spec,
    )

  @parameterized.product(
      allow_override=(False, True), include_optional=(False, True)
  )
  def test_builtins_without_plugins(self, allow_override, include_optional):
    with mock.patch.object(
        metadata, "entry_points", return_value=metadata.EntryPoints()
    ):
      providers = plugins.available_providers(allow_override, include_optional)
    self.assertEqual(
        providers["gemini"], "langextract.providers.gemini:GeminiLanguageModel"
    )
    self.assertEqual(
        providers["ollama"], "langextract.providers.ollama:OllamaLanguageModel"
    )
    self.assertEqual("openai" in providers, include_optional)


if __name__ == "__main__":
  absltest.main()
