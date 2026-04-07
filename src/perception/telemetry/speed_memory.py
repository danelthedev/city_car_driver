import time


class SpeedTelemetryReader:
    def __init__(
        self,
        process_name: str = "starter.exe",
        module_name: str = "pdd.dll",
        speed_offset: int = 0xE322B0,
        poll_interval_ms: float = 50.0,
    ):
        self.process_name = str(process_name)
        self.module_name = str(module_name)
        self.speed_offset = int(speed_offset)
        self.poll_interval_s = max(0.0, float(poll_interval_ms) / 1000.0)

        self._pm = None
        self._speed_address = None
        self._next_poll_ts = 0.0

    def _connect(self):
        pymem = __import__("pymem")
        pymem_process = __import__("pymem.process", fromlist=["module_from_name"])

        self._pm = pymem.Pymem(self.process_name)
        module = pymem_process.module_from_name(self._pm.process_handle, self.module_name)
        if module is None:
            raise RuntimeError(f"{self.module_name} not found in process {self.process_name}")
        base = module.lpBaseOfDll
        self._speed_address = int(base) + int(self.speed_offset)

    def read_speed_if_due(self, now_ts: float | None = None):
        ts = time.perf_counter() if now_ts is None else float(now_ts)
        if ts < self._next_poll_ts:
            return None

        self._next_poll_ts = ts + self.poll_interval_s
        try:
            if self._pm is None or self._speed_address is None:
                self._connect()
            value = abs(float(self._pm.read_float(self._speed_address)))
            if 0.0 <= value <= 500.0:
                return value
            return None
        except Exception:
            self._pm = None
            self._speed_address = None
            return None

    def close(self):
        self._pm = None
        self._speed_address = None
