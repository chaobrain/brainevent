"""Record RAM and VRAM usage for isolated benchmark workers."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any, Callable, TextIO


SAMPLE_INTERVAL_S = 0.1
SAMPLE_FIELDS = (
    "case_id",
    "sample_index",
    "timestamp_utc",
    "elapsed_s",
    "worker_pid",
    "process_count",
    "ram_rss_mib",
    "vram_used_mib",
    "gpu_uuids",
    "sample_status",
    "sample_error",
)
RESULT_FIELDS = (
    "ram_peak_mib",
    "vram_peak_mib",
    "resource_sample_count",
    "resource_status",
    "resource_error",
)


class ResourceMonitor:
    """Record resource samples for isolated benchmark workers.

    Parameters
    ----------
    samples_path : pathlib.Path
        CSV path receiving incremental resource samples.
    sample_interval_s : float, optional
        Delay in seconds between resource samples.
    proc_root : pathlib.Path, optional
        Linux process filesystem root.
    gpu_query : callable, optional
        Function returning per-process NVIDIA CSV rows.
    gpu_inventory_query : callable, optional
        Function returning NVIDIA device inventory CSV rows.
    """

    def __init__(
        self,
        samples_path: Path,
        *,
        sample_interval_s: float = SAMPLE_INTERVAL_S,
        proc_root: Path = Path("/proc"),
        gpu_query: Callable[[], str] | None = None,
        gpu_inventory_query: Callable[[], str] | None = None,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        self.samples_path = Path(samples_path)
        self.sample_interval_s = float(sample_interval_s)
        self.proc_root = Path(proc_root)
        self._gpu_query = gpu_query or self._query_gpu_processes
        self._gpu_inventory_query = (
            gpu_inventory_query or self._query_gpu_inventory
        )
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._states: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> ResourceMonitor:
        """Open the incremental sample CSV.

        Returns
        -------
        ResourceMonitor
            Active monitor writing to ``samples_path``.
        """
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.samples_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=SAMPLE_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the incremental sample CSV.

        Parameters
        ----------
        *args : Any
            Context-manager exception details, when present.
        """
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None

    @staticmethod
    def _query_gpu_processes() -> str:
        command = (
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout

    @staticmethod
    def _query_gpu_inventory() -> str:
        command = (
            "nvidia-smi",
            "--query-gpu=driver_version,index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        )
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout

    def _process_tree(self, worker_pid: int) -> set[int]:
        process_ids: set[int] = set()
        pending = [worker_pid]
        while pending:
            pid = pending.pop()
            if pid in process_ids:
                continue
            process_ids.add(pid)
            children_path = (
                self.proc_root / str(pid) / "task" / str(pid) / "children"
            )
            try:
                children = children_path.read_text(encoding="utf-8").split()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            pending.extend(int(child) for child in children)
        return process_ids

    def _read_ram_mib(self, process_ids: set[int]) -> tuple[float | str, list[str]]:
        total_kib = 0
        errors = []
        readable = 0
        for pid in process_ids:
            try:
                lines = (
                    self.proc_root / str(pid) / "status"
                ).read_text(encoding="utf-8").splitlines()
                rss_line = next(line for line in lines if line.startswith("VmRSS:"))
                total_kib += int(rss_line.split()[1])
                readable += 1
            except (
                FileNotFoundError,
                PermissionError,
                ProcessLookupError,
                StopIteration,
                ValueError,
            ) as error:
                errors.append(f"RAM pid {pid}: {type(error).__name__}: {error}")
        return (total_kib / 1024.0 if readable else ""), errors

    def _read_vram_mib(
        self, process_ids: set[int]
    ) -> tuple[float | str, str, list[str]]:
        try:
            output = self._gpu_query()
            total_mib = 0.0
            gpu_uuids = set()
            for line in output.splitlines():
                if not line.strip():
                    continue
                pid_text, gpu_uuid, used_text = (
                    part.strip() for part in line.split(",", maxsplit=2)
                )
                if int(pid_text) in process_ids:
                    total_mib += float(used_text)
                    gpu_uuids.add(gpu_uuid)
            return total_mib, ";".join(sorted(gpu_uuids)), []
        except (
            FileNotFoundError,
            PermissionError,
            subprocess.SubprocessError,
            TypeError,
            ValueError,
        ) as error:
            message = f"{type(error).__name__}: {error}"
            return "", "", [message]

    def record_sample(
        self, case_id: str, worker_pid: int, *, elapsed_s: float
    ) -> dict[str, Any]:
        """Capture and persist one resource observation.

        Parameters
        ----------
        case_id : str
            Stable case identity within the benchmark run.
        worker_pid : int
            Root worker process identifier.
        elapsed_s : float
            Monotonic seconds since the worker started.

        Returns
        -------
        dict
            Serializable sample row written to the resource CSV.
        """
        if self._writer is None or self._file is None:
            raise RuntimeError("ResourceMonitor must be used as a context manager")

        state = self._states.setdefault(
            case_id,
            {
                "count": 0,
                "ram_peak": None,
                "vram_peak": None,
                "errors": set(),
                "all_ok": True,
                "any_valid": False,
            },
        )
        process_ids = self._process_tree(worker_pid)
        ram_mib, ram_errors = self._read_ram_mib(process_ids)
        vram_mib, gpu_uuids, vram_errors = self._read_vram_mib(process_ids)
        errors = [*ram_errors, *vram_errors]
        valid_count = sum(value != "" for value in (ram_mib, vram_mib))
        status = "ok" if not errors else "partial" if valid_count else "unavailable"

        state["count"] += 1
        if ram_mib != "":
            ram_value = float(ram_mib)
            state["ram_peak"] = (
                ram_value
                if state["ram_peak"] is None
                else max(state["ram_peak"], ram_value)
            )
        if vram_mib != "":
            vram_value = float(vram_mib)
            state["vram_peak"] = (
                vram_value
                if state["vram_peak"] is None
                else max(state["vram_peak"], vram_value)
            )
        state["errors"].update(errors)
        state["all_ok"] = state["all_ok"] and status == "ok"
        state["any_valid"] = state["any_valid"] or valid_count > 0
        row = {
            "case_id": case_id,
            "sample_index": state["count"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": elapsed_s,
            "worker_pid": worker_pid,
            "process_count": len(process_ids),
            "ram_rss_mib": ram_mib,
            "vram_used_mib": vram_mib,
            "gpu_uuids": gpu_uuids,
            "sample_status": status,
            "sample_error": "; ".join(errors),
        }
        self._writer.writerow(row)
        self._file.flush()
        return row

    def summary(self, case_id: str | None = None) -> dict[str, Any]:
        """Return peak resource usage for one observed case.

        Parameters
        ----------
        case_id : str, optional
            Case to summarize. It may be omitted when only one case exists.

        Returns
        -------
        dict
            Peak RAM, peak VRAM, sample count, and collection status.

        Raises
        ------
        ValueError
            If the case is ambiguous or has not been observed.
        """
        if case_id is None:
            if len(self._states) != 1:
                raise ValueError("case_id is required unless exactly one case exists")
            case_id = next(iter(self._states))
        if case_id not in self._states:
            raise ValueError(f"unknown case_id: {case_id}")
        state = self._states[case_id]
        errors = sorted(state["errors"])
        if state["count"] and state["all_ok"]:
            status = "ok"
        elif state["any_valid"]:
            status = "partial"
        else:
            status = "unavailable"
        return {
            "ram_peak_mib": (
                state["ram_peak"] if state["ram_peak"] is not None else ""
            ),
            "vram_peak_mib": (
                state["vram_peak"] if state["vram_peak"] is not None else ""
            ),
            "resource_sample_count": state["count"],
            "resource_status": status,
            "resource_error": "; ".join(errors),
        }

    def monitor_process(
        self, case_id: str, process: Any, receiver: Any
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Sample a worker until it returns a result or exits.

        Parameters
        ----------
        case_id : str
            Stable case identity within the benchmark run.
        process : multiprocessing.Process
            Started worker process exposing ``pid`` and ``is_alive``.
        receiver : multiprocessing.Connection
            Parent-side connection receiving the workload result.

        Returns
        -------
        result : dict or None
            Worker result, or ``None`` when the worker exits without one.
        resources : dict
            Peak resource fields for the case result row.
        """
        if process.pid is None:
            raise ValueError("worker process has no pid")
        started = time.monotonic()
        result = None
        while True:
            self.record_sample(
                case_id,
                process.pid,
                elapsed_s=time.monotonic() - started,
            )
            try:
                if receiver.poll(self.sample_interval_s):
                    result = receiver.recv()
                    break
            except (EOFError, OSError):
                break
            if not process.is_alive():
                break
        return result, self.summary(case_id)

    def write_metadata(self, output: Path) -> None:
        """Write reproducibility metadata for the monitored run.

        Parameters
        ----------
        output : pathlib.Path
            JSON file receiving host, environment, and GPU inventory data.
        """
        warnings = []
        gpus = []
        try:
            for line in self._gpu_inventory_query().splitlines():
                if not line.strip():
                    continue
                driver, index, uuid, name, total = (
                    part.strip() for part in line.split(",", maxsplit=4)
                )
                gpus.append({
                    "driver_version": driver,
                    "index": int(index),
                    "uuid": uuid,
                    "name": name,
                    "memory_total_mib": float(total),
                })
        except (
            FileNotFoundError,
            PermissionError,
            subprocess.SubprocessError,
            TypeError,
            ValueError,
        ) as error:
            warnings.append(f"GPU inventory: {type(error).__name__}: {error}")
        metadata = {
            "schema_version": 1,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "sample_interval_s": self.sample_interval_s,
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                    "XLA_PYTHON_CLIENT_PREALLOCATE"
                ),
            },
            "gpus": gpus,
            "warnings": warnings,
            "sampling_limit": (
                "Allocations shorter than one sampling interval may be missed."
            ),
        }
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
