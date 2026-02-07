# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
# ==============================================================================

import json
import os

import pytest

from brainevent._cli import _build_parser, _filter_primitives, _resolve_dtype, main


class TestBuildParser:
    def test_parser_creates(self):
        parser = _build_parser()
        assert parser is not None

    def test_benchmark_performance_args(self):
        parser = _build_parser()
        args = parser.parse_args([
            'benchmark-performance',
            '--platform', 'cpu',
            '--data', 'csr',
            '--n-pre', '100',
            '--n-post', '200',
            '--prob', '0.05',
        ])
        assert args.command == 'benchmark-performance'
        assert args.platform == 'cpu'
        assert args.data == 'csr'
        assert args.n_pre == 100
        assert args.n_post == 200
        assert args.prob == 0.05

    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([
            'benchmark-performance',
            '--platform', 'gpu',
        ])
        assert args.data == 'all'
        assert args.n_pre == 1000
        assert args.n_post == 1000
        assert args.prob == 0.1
        assert args.n_warmup == 5
        assert args.n_runs == 20
        assert args.dtype == 'float32'
        assert args.output is None
        assert args.persist is False

    def test_persist_flag(self):
        parser = _build_parser()
        args = parser.parse_args([
            'benchmark-performance',
            '--platform', 'cpu',
            '--persist',
        ])
        assert args.persist is True

    def test_no_persist_flag(self):
        parser = _build_parser()
        args = parser.parse_args([
            'benchmark-performance',
            '--platform', 'cpu',
            '--no-persist',
        ])
        assert args.no_persist is True


class TestResolveDtype:
    def test_float32(self):
        import jax.numpy as jnp
        assert _resolve_dtype('float32') == jnp.float32

    def test_float64(self):
        import jax.numpy as jnp
        assert _resolve_dtype('float64') == jnp.float64


class TestFilterPrimitives:
    def test_all_returns_everything(self):
        registry = {'a': object(), 'b': object()}
        result = _filter_primitives(registry, 'all')
        assert result == registry

    def test_filter_by_tag(self):
        class FakePrim:
            def __init__(self, tags):
                self._tags = tags

        registry = {
            'csr_a': FakePrim({'csr', 'binary'}),
            'coo_b': FakePrim({'coo', 'binary'}),
            'dense_c': FakePrim({'dense', 'float'}),
        }
        result = _filter_primitives(registry, 'csr')
        assert 'csr_a' in result
        assert 'coo_b' not in result
        assert 'dense_c' not in result

    def test_filter_comma_separated(self):
        class FakePrim:
            def __init__(self, tags):
                self._tags = tags

        registry = {
            'csr_a': FakePrim({'csr', 'binary'}),
            'coo_b': FakePrim({'coo', 'binary'}),
        }
        result = _filter_primitives(registry, 'csr,coo')
        assert 'csr_a' in result
        assert 'coo_b' in result

    def test_filter_no_match(self):
        class FakePrim:
            def __init__(self, tags):
                self._tags = tags

        registry = {
            'csr_a': FakePrim({'csr', 'binary'}),
        }
        result = _filter_primitives(registry, 'nonexistent')
        assert len(result) == 0


class TestMainEntryPoint:
    def test_no_command_returns_0(self):
        assert main([]) == 0

    def test_invalid_command(self):
        # argparse will error on invalid subcommand
        with pytest.raises(SystemExit):
            main(['invalid-command'])

    def test_missing_required_platform(self):
        with pytest.raises(SystemExit):
            main(['benchmark-performance'])


class TestXLACustomKernelPersist:
    def test_set_default_persist(self, tmp_path, monkeypatch):
        """set_default with persist=True should write to config."""
        config_path = str(tmp_path / 'brainevent' / 'defaults.json')
        monkeypatch.setattr('brainevent._config.get_config_path', lambda: config_path)
        from brainevent._config import invalidate_cache
        invalidate_cache()

        import brainevent
        prim = brainevent.binary_csrmv_p
        prim.set_default('cpu', 'numba', persist=True)

        assert os.path.isfile(config_path)
        with open(config_path) as f:
            data = json.load(f)
        assert data['defaults']['binary_csrmv']['cpu'] == 'numba'
        invalidate_cache()

    def test_set_default_no_persist(self, tmp_path, monkeypatch):
        """set_default without persist should not write to config."""
        config_path = str(tmp_path / 'brainevent' / 'defaults.json')
        monkeypatch.setattr('brainevent._config.get_config_path', lambda: config_path)
        from brainevent._config import invalidate_cache
        invalidate_cache()

        import brainevent
        prim = brainevent.binary_csrmv_p
        prim.set_default('cpu', 'numba', persist=False)

        assert not os.path.isfile(config_path)
        invalidate_cache()


class TestXLACustomKernelTags:
    def test_def_tags(self):
        import brainevent
        prim = brainevent.binary_csrmv_p
        assert 'csr' in prim._tags
        assert 'binary' in prim._tags

    def test_def_benchmark_data(self):
        import brainevent
        prim = brainevent.binary_csrmv_p
        assert prim._benchmark_data_fn is not None


class TestXLACustomKernelUserDefaults:
    def test_apply_user_defaults(self, tmp_path, monkeypatch):
        """Lazy user defaults should be applied on first dispatch lookup."""
        config_path = str(tmp_path / 'brainevent' / 'defaults.json')
        monkeypatch.setattr('brainevent._config.get_config_path', lambda: config_path)
        from brainevent._config import invalidate_cache, save_user_defaults
        invalidate_cache()

        # Write a config with numba as default for cpu
        save_user_defaults({'binary_csrmv': {'cpu': 'numba'}})
        invalidate_cache()

        import brainevent
        prim = brainevent.binary_csrmv_p
        # Reset the lazy flag to test
        prim._user_defaults_applied = False
        prim._apply_user_defaults()
        assert prim._defaults.get('cpu') == 'numba'
        invalidate_cache()
