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
import warnings

import pytest

from brainevent.config import (
    _SCHEMA_VERSION,
    _read_config_file,
    _write_config_file,
    clear_user_defaults,
    get_config_path,
    get_user_default,
    invalidate_cache,
    load_user_defaults,
    save_user_defaults,
    set_user_default,
)


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Redirect config to a temp directory and clear cache for each test."""
    config_path = str(tmp_path / 'brainevent' / 'defaults.json')
    monkeypatch.setattr('brainevent._config.get_config_path', lambda: config_path)
    invalidate_cache()
    yield config_path
    invalidate_cache()


class TestGetConfigPath:
    def test_returns_string(self):
        path = get_config_path()
        assert isinstance(path, str)

    def test_ends_with_defaults_json(self):
        path = get_config_path()
        assert path.endswith('defaults.json')


class TestReadConfigFile:
    def test_missing_file_returns_default(self, isolate_config):
        data = _read_config_file('/nonexistent/path/defaults.json')
        assert data['schema_version'] == _SCHEMA_VERSION
        assert data['defaults'] == {}

    def test_corrupted_json(self, isolate_config):
        path = isolate_config
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('not valid json{{{')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            data = _read_config_file(path)
            assert any('Corrupted' in str(warning.message) for warning in w)
        assert data['defaults'] == {}

    def test_unsupported_schema_version(self, isolate_config):
        path = isolate_config
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'schema_version': 999, 'defaults': {'foo': {'cpu': 'bar'}}}, f)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            data = _read_config_file(path)
            assert any('schema version' in str(warning.message) for warning in w)
        assert data['defaults'] == {}

    def test_valid_file(self, isolate_config):
        path = isolate_config
        expected = {
            'schema_version': 1,
            'defaults': {'binary_csrmv': {'cpu': 'numba'}},
            'benchmark_metadata': {},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(expected, f)
        data = _read_config_file(path)
        assert data == expected


class TestWriteConfigFile:
    def test_creates_directory_and_file(self, isolate_config):
        path = isolate_config
        data = {'schema_version': 1, 'defaults': {}, 'benchmark_metadata': {}}
        _write_config_file(path, data)
        assert os.path.isfile(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_write(self, isolate_config):
        path = isolate_config
        data1 = {'schema_version': 1, 'defaults': {'a': {'cpu': 'b'}}, 'benchmark_metadata': {}}
        data2 = {'schema_version': 1, 'defaults': {'c': {'gpu': 'd'}}, 'benchmark_metadata': {}}
        _write_config_file(path, data1)
        _write_config_file(path, data2)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data2


class TestLoadSaveUserDefaults:
    def test_load_empty(self, isolate_config):
        defaults = load_user_defaults()
        assert defaults == {}

    def test_save_and_load(self, isolate_config):
        save_user_defaults({'binary_csrmv': {'cpu': 'numba', 'gpu': 'pallas'}})
        invalidate_cache()
        defaults = load_user_defaults()
        assert defaults['binary_csrmv']['cpu'] == 'numba'
        assert defaults['binary_csrmv']['gpu'] == 'pallas'

    def test_save_merges(self, isolate_config):
        save_user_defaults({'prim_a': {'cpu': 'numba'}})
        save_user_defaults({'prim_b': {'gpu': 'warp'}})
        invalidate_cache()
        defaults = load_user_defaults()
        assert defaults['prim_a']['cpu'] == 'numba'
        assert defaults['prim_b']['gpu'] == 'warp'

    def test_save_with_metadata(self, isolate_config):
        metadata = {'last_run': '2026-02-07T12:00:00Z', 'platform': 'cpu'}
        save_user_defaults({'prim': {'cpu': 'numba'}}, metadata=metadata)
        path = isolate_config
        with open(path) as f:
            data = json.load(f)
        assert data['benchmark_metadata'] == metadata

    def test_caching(self, isolate_config):
        save_user_defaults({'prim': {'cpu': 'numba'}})
        # First load caches
        d1 = load_user_defaults()
        # Second load should use cache (same object)
        d2 = load_user_defaults()
        assert d1 is d2


class TestGetSetUserDefault:
    def test_get_nonexistent(self, isolate_config):
        assert get_user_default('nonexistent', 'cpu') is None

    def test_set_and_get(self, isolate_config):
        set_user_default('binary_csrmv', 'cpu', 'numba')
        invalidate_cache()
        assert get_user_default('binary_csrmv', 'cpu') == 'numba'

    def test_set_overwrites(self, isolate_config):
        set_user_default('prim', 'cpu', 'backend_a')
        set_user_default('prim', 'cpu', 'backend_b')
        invalidate_cache()
        assert get_user_default('prim', 'cpu') == 'backend_b'


class TestClearUserDefaults:
    def test_clear_removes_file(self, isolate_config):
        save_user_defaults({'prim': {'cpu': 'numba'}})
        assert os.path.isfile(isolate_config)
        clear_user_defaults()
        assert not os.path.isfile(isolate_config)

    def test_clear_invalidates_cache(self, isolate_config):
        save_user_defaults({'prim': {'cpu': 'numba'}})
        load_user_defaults()
        clear_user_defaults()
        defaults = load_user_defaults()
        assert defaults == {}

    def test_clear_nonexistent_file(self, isolate_config):
        # Should not raise even if file doesn't exist
        clear_user_defaults()
