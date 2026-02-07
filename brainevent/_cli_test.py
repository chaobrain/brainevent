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

from brainevent._cli import _build_parser, _filter_primitives, main


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
        ])
        assert args.command == 'benchmark-performance'
        assert args.platform == 'cpu'
        assert args.data == 'csr'

    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([
            'benchmark-performance',
            '--platform', 'gpu',
        ])
        assert args.data == 'all'
        assert args.n_warmup == 5
        assert args.n_runs == 20
        assert args.output is None


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
        assert callable(prim._benchmark_data_fn)


class TestMultiBenchmarkData:
    def test_def_benchmark_data_stores_fn(self):
        """def_benchmark_data(fn) stores the function."""
        from brainevent._op.main import XLACustomKernel
        from brainevent._op.benchmark import BenchmarkConfig
        prim = XLACustomKernel('_test_stores_fn')
        fn = lambda *, platform: [BenchmarkConfig("default", ())]
        prim.def_benchmark_data(fn)
        assert prim._benchmark_data_fn is fn

    def test_def_benchmark_data_none_initially(self):
        """_benchmark_data_fn is None when no fn registered."""
        from brainevent._op.main import XLACustomKernel
        prim = XLACustomKernel('_test_none_initially')
        assert prim._benchmark_data_fn is None

    def test_def_benchmark_data_overwrite(self):
        """Second def_benchmark_data call overwrites the first."""
        from brainevent._op.main import XLACustomKernel
        from brainevent._op.benchmark import BenchmarkConfig
        prim = XLACustomKernel('_test_overwrite')
        fn1 = lambda *, platform: [BenchmarkConfig("a", ())]
        fn2 = lambda *, platform: [BenchmarkConfig("b", ())]
        prim.def_benchmark_data(fn1)
        prim.def_benchmark_data(fn2)
        assert prim._benchmark_data_fn is fn2

    def test_benchmark_fn_returns_list(self):
        """Benchmark data fn should return list of BenchmarkConfig instances."""
        from brainevent._op.main import XLACustomKernel
        from brainevent._op.benchmark import BenchmarkConfig
        prim = XLACustomKernel('_test_returns_list')

        def bench_fn(*, platform):
            return [
                BenchmarkConfig("config_a", (1, 2), {"key": "val"}),
                BenchmarkConfig("config_b", (3, 4), {"key": "val2"}),
            ]

        prim.def_benchmark_data(bench_fn)
        configs = prim._benchmark_data_fn(platform='cpu')
        assert len(configs) == 2
        assert configs[0].name == "config_a"
        assert configs[1].name == "config_b"
        for config in configs:
            assert isinstance(config, BenchmarkConfig)
            assert isinstance(config.name, str)
            assert isinstance(config.args, tuple)
            assert isinstance(config.kwargs, dict)


class TestMultiBenchmarkIntegration:
    def test_real_primitive_has_benchmark_data(self):
        """Real primitives should have a benchmark data fn."""
        import brainevent
        prim = brainevent.binary_csrmv_p
        assert prim._benchmark_data_fn is not None
        assert callable(prim._benchmark_data_fn)

    def test_benchmark_fn_returns_configs(self):
        """Benchmark data fn should return list of BenchmarkConfig instances."""
        import brainevent
        from brainevent._op.benchmark import BenchmarkConfig
        prim = brainevent.binary_csrmv_p
        configs = prim._benchmark_data_fn(platform='cpu')
        assert isinstance(configs, list)
        assert len(configs) >= 1
        for config in configs:
            assert isinstance(config, BenchmarkConfig)
            assert isinstance(config.name, str)
            assert isinstance(config.args, tuple)
            assert isinstance(config.kwargs, dict)
            assert 'shape' in config.kwargs

    def test_registry_primitives_benchmark_data_structure(self):
        """All primitives with benchmark data should return proper BenchmarkConfig lists."""
        from brainevent._registry import get_registry
        from brainevent._op.benchmark import BenchmarkConfig
        registry = get_registry()
        for name, prim in registry.items():
            if prim._benchmark_data_fn is None:
                continue
            assert callable(prim._benchmark_data_fn), (
                f"Primitive '{name}': _benchmark_data_fn should be callable"
            )
            configs = prim._benchmark_data_fn(platform='cpu')
            assert isinstance(configs, list), (
                f"Primitive '{name}': benchmark data fn should return a list"
            )
            for config in configs:
                assert isinstance(config, BenchmarkConfig), (
                    f"Primitive '{name}': each entry should be a BenchmarkConfig instance"
                )
                assert isinstance(config.name, str), (
                    f"Primitive '{name}': config name should be a string"
                )
                assert isinstance(config.args, tuple), (
                    f"Primitive '{name}': args should be a tuple"
                )
                assert isinstance(config.kwargs, dict), (
                    f"Primitive '{name}': kwargs should be a dict"
                )


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
