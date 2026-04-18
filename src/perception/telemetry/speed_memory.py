import time


class DistTurnKalmanFilter:
    """1-D Kalman filter for distance-to-turn.

    Prediction uses the current vehicle speed as a control input so the
    estimate keeps decreasing even between sporadic memory reads.

    State  : estimated distance to turn (metres)
    Control: speed (m/s) × Δt  →  expected distance decrease per step
    """

    def __init__(
        self,
        process_noise: float = 2.0,
        measurement_noise: float = 15.0,
    ):
        self.Q = float(process_noise)       # process noise variance (m²)
        self.R = float(measurement_noise)   # measurement noise variance (m²)

        self._x: float | None = None        # state estimate
        self._P: float = 1.0                # estimate covariance
        self._last_ts: float | None = None  # timestamp of last update

    @property
    def estimate(self) -> float | None:
        return self._x

    def reset(self) -> None:
        self._x = None
        self._P = 1.0
        self._last_ts = None

    def predict(self, speed_ms: float, now_ts: float) -> None:
        """Advance the state using current speed (no measurement)."""
        if self._x is None or self._last_ts is None:
            self._last_ts = now_ts
            return
        dt = max(0.0, now_ts - self._last_ts)
        self._last_ts = now_ts
        if dt <= 0.0:
            return
        # State transition: distance decreases by speed * dt
        self._x = max(0.0, self._x - speed_ms * dt)
        # Covariance grows with time
        self._P += self.Q * dt

    def update(self, measurement: float, speed_ms: float, now_ts: float) -> float:
        """Incorporate a new raw measurement and return the updated estimate."""
        if self._x is None or self._last_ts is None:
            # First measurement: initialise directly
            self._x = measurement
            self._P = self.R
            self._last_ts = now_ts
            return self._x

        # Predict forward to current time
        dt = max(0.0, now_ts - self._last_ts)
        self._last_ts = now_ts
        x_pred = max(0.0, self._x - speed_ms * dt)
        P_pred = self._P + self.Q * dt

        # Kalman gain
        K = P_pred / (P_pred + self.R)

        # Update
        self._x = x_pred + K * (measurement - x_pred)
        self._P = (1.0 - K) * P_pred
        return self._x


class SpeedTelemetryReader:
    def __init__(
        self,
        process_name: str = "starter.exe",
        module_name: str = "pdd.dll",
        speed_offset: int = 0xE322B0,
        dist_turn_offset: int = 0xF10C70,
        dist_dest_offset: int = 0xF10C4C,
        poll_interval_ms: float = 50.0,
    ):
        self.process_name = str(process_name)
        self.module_name = str(module_name)
        self.speed_offset = int(speed_offset)
        self.dist_turn_offset = int(dist_turn_offset)
        self.dist_dest_offset = int(dist_dest_offset)
        self.poll_interval_s = max(0.0, float(poll_interval_ms) / 1000.0)

        self._pm = None
        self._speed_address = None
        self._dist_turn_address = None
        self._dist_dest_address = None
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
        self._dist_turn_address = int(base) + int(self.dist_turn_offset)
        self._dist_dest_address = int(base) + int(self.dist_dest_offset)

    def read_telemetry_if_due(self, now_ts: float | None = None):
        """Return (speed_kmh, dist_turn_m, dist_dest_m) tuple; each value is None if unavailable."""
        ts = time.perf_counter() if now_ts is None else float(now_ts)
        if ts < self._next_poll_ts:
            return None, None, None

        self._next_poll_ts = ts + self.poll_interval_s
        try:
            if self._pm is None or self._speed_address is None or self._dist_turn_address is None or self._dist_dest_address is None:
                self._connect()

            raw_speed = abs(float(self._pm.read_float(self._speed_address)))
            speed = raw_speed if 0.0 <= raw_speed <= 500.0 else None

            raw_dist = float(self._pm.read_float(self._dist_turn_address))
            dist_turn = raw_dist if 0.0 <= raw_dist <= 100000.0 else None

            raw_dest = float(self._pm.read_float(self._dist_dest_address))
            dist_dest = raw_dest if 0.0 <= raw_dest <= 100000.0 else None

            return speed, dist_turn, dist_dest
        except Exception:
            self._pm = None
            self._speed_address = None
            self._dist_turn_address = None
            self._dist_dest_address = None
            return None, None, None

    def close(self):
        self._pm = None
        self._speed_address = None
        self._dist_turn_address = None
        self._dist_dest_address = None
