from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__, event_type="config")


def _to_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_int(value: str | int | None, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: str | float | int | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_optional_str(value: str | None, default: str | None = None) -> str | None:
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def _first_env(*keys: str, default: str | None = None) -> str | None:
    """
    Return the first non-empty environment variable among aliases.

    This keeps canonical EXCHANGE_* keys while supporting exchange-specific
    aliases such as BINANCE_* for existing .env files.
    """
    for key in keys:
        value = os.getenv(key)
        if value is not None and value.strip() != "":
            return value
    return default


def _load_env_file(env_file: str | Path = ".env") -> None:
    """
    Мінімальний .env loader без сторонніх залежностей.
    Не перезаписує вже існуючі environment variables.
    """
    env_path = Path(env_file)
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value

        logger.info("Environment file loaded | path=%s", str(env_path))
    except Exception:
        logger.exception("Failed to load environment file | path=%s", str(env_path))
        raise


@dataclass(slots=True)
class AppConfig:
    name: str = "trading_system"
    env: str = "dev"
    debug: bool = False
    timezone: str = "Europe/Kiev"
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    json_logs: bool = False
    enable_file_logging: bool = False
    log_dir: str = "logs"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


@dataclass(slots=True)
class EventBusConfig:
    max_queue_size: int = 100000
    worker_count: int = 12
    queue_full_policy: str = "drop_oldest"
    max_retries: int = 1
    retry_delay: float = 0.02
    enable_metrics: bool = True


@dataclass(slots=True)
class SchedulerConfig:
    tick_interval: float = 0.2
    graceful_shutdown_timeout: float = 10.0
    wait_running_jobs_on_shutdown: bool = True


@dataclass(slots=True)
class RedisConfig:
    enabled: bool = True
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    ssl: bool = False
    key_prefix: str = "trading_system"


@dataclass(slots=True)
class PostgresConfig:
    enabled: bool = True
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_system"
    user: str = "postgres"
    password: str | None = None
    pool_min_size: int = 1
    pool_max_size: int = 10


@dataclass(slots=True)
class ExchangeCredentials:
    api_key: str | None = None
    api_secret: str | None = None
    passphrase: str | None = None
    testnet: bool = False


@dataclass(slots=True)
class ExchangeConfig:
    enabled: bool = True
    name: str = "binance"
    ws_url: str | None = None
    rest_url: str | None = None
    timeout_seconds: float = 10.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 20
    credentials: ExchangeCredentials = field(default_factory=ExchangeCredentials)


@dataclass(slots=True)
class RiskConfig:
    max_risk_per_trade_pct: float = 1.0
    max_daily_drawdown_pct: float = 5.0
    max_open_positions: int = 3
    max_total_exposure_pct: float = 30.0
    kill_switch_enabled: bool = True


@dataclass(slots=True)
class AIConfig:
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-5"
    api_key: str | None = None
    timeout_seconds: float = 20.0
    max_signal_tokens: int = 2000


@dataclass(slots=True)
class DashboardConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    cors_enabled: bool = True
    cors_allow_origins: str = "*"


@dataclass(slots=True)
class StorageConfig:
    parquet_enabled: bool = True
    parquet_dir: str = "data/parquet"
    flush_interval_seconds: float = 15.0
    batch_size: int = 1000


@dataclass(slots=True)
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Config":
        if env_file:
            _load_env_file(env_file)

        base_dir = Path(_get_env("APP_BASE_DIR", str(Path.cwd()))).resolve()
        data_dir = Path(_get_env("APP_DATA_DIR", str(base_dir / "data"))).resolve()
        logs_dir = Path(_get_env("APP_LOGS_DIR", str(base_dir / "logs"))).resolve()

        config = cls(
            app=AppConfig(
                name=_get_env("APP_NAME", "trading_system") or "trading_system",
                env=_get_env("APP_ENV", "dev") or "dev",
                debug=_to_bool(_get_env("APP_DEBUG"), False),
                timezone=_get_env("APP_TIMEZONE", "Europe/Kiev") or "Europe/Kiev",
                base_dir=base_dir,
                data_dir=data_dir,
                logs_dir=logs_dir,
            ),
            logging=LoggingConfig(
                level=(_get_env("LOG_LEVEL", "INFO") or "INFO").upper(),
                json_logs=_to_bool(_get_env("LOG_JSON"), False),
                enable_file_logging=_to_bool(_get_env("LOG_TO_FILE"), False),
                log_dir=_get_env("LOG_DIR", str(logs_dir)) or str(logs_dir),
                max_bytes=_to_int(_get_env("LOG_MAX_BYTES"), 10 * 1024 * 1024),
                backup_count=_to_int(_get_env("LOG_BACKUP_COUNT"), 5),
            ),
            event_bus=EventBusConfig(
                max_queue_size=_to_int(_get_env("EVENT_BUS_MAX_QUEUE_SIZE"), 100000),
                worker_count=_to_int(_get_env("EVENT_BUS_WORKER_COUNT"), 12),
                queue_full_policy=_get_env("EVENT_BUS_QUEUE_FULL_POLICY", "drop_oldest") or "drop_oldest",
                max_retries=_to_int(_get_env("EVENT_BUS_MAX_RETRIES"), 1),
                retry_delay=_to_float(_get_env("EVENT_BUS_RETRY_DELAY"), 0.02),
                enable_metrics=_to_bool(_get_env("EVENT_BUS_ENABLE_METRICS"), True),
            ),
            scheduler=SchedulerConfig(
                tick_interval=_to_float(_get_env("SCHEDULER_TICK_INTERVAL"), 0.2),
                graceful_shutdown_timeout=_to_float(_get_env("SCHEDULER_SHUTDOWN_TIMEOUT"), 10.0),
                wait_running_jobs_on_shutdown=_to_bool(
                    _get_env("SCHEDULER_WAIT_RUNNING_JOBS_ON_SHUTDOWN"),
                    True,
                ),
            ),
            redis=RedisConfig(
                enabled=_to_bool(_get_env("REDIS_ENABLED"), True),
                host=_get_env("REDIS_HOST", "localhost") or "localhost",
                port=_to_int(_get_env("REDIS_PORT"), 6379),
                db=_to_int(_get_env("REDIS_DB"), 0),
                password=_to_optional_str(_get_env("REDIS_PASSWORD")),
                ssl=_to_bool(_get_env("REDIS_SSL"), False),
                key_prefix=_get_env("REDIS_KEY_PREFIX", "trading_system") or "trading_system",
            ),
            postgres=PostgresConfig(
                enabled=_to_bool(_get_env("POSTGRES_ENABLED"), True),
                host=_get_env("POSTGRES_HOST", "localhost") or "localhost",
                port=_to_int(_get_env("POSTGRES_PORT"), 5432),
                database=_get_env("POSTGRES_DB", "trading_system") or "trading_system",
                user=_get_env("POSTGRES_USER", "postgres") or "postgres",
                password=_to_optional_str(_get_env("POSTGRES_PASSWORD")),
                pool_min_size=_to_int(_get_env("POSTGRES_POOL_MIN_SIZE"), 1),
                pool_max_size=_to_int(_get_env("POSTGRES_POOL_MAX_SIZE"), 10),
            ),
            exchange=ExchangeConfig(
                enabled=_to_bool(_first_env("EXCHANGE_ENABLED", "BINANCE_ENABLED"), True),
                name=_first_env("EXCHANGE_NAME", default="binance") or "binance",
                ws_url=_to_optional_str(
                    _first_env(
                        "EXCHANGE_WS_URL",
                        "BINANCE_WS_URL",
                        "BINANCE_FUTURES_WS_URL",
                    )
                ),
                rest_url=_to_optional_str(
                    _first_env(
                        "EXCHANGE_REST_URL",
                        "BINANCE_REST_URL",
                        "BINANCE_FUTURES_REST_URL",
                    )
                ),
                timeout_seconds=_to_float(_first_env("EXCHANGE_TIMEOUT_SECONDS", "BINANCE_TIMEOUT_SECONDS"), 10.0),
                reconnect_delay=_to_float(_first_env("EXCHANGE_RECONNECT_DELAY", "BINANCE_RECONNECT_DELAY"), 5.0),
                max_reconnect_attempts=_to_int(
                    _first_env("EXCHANGE_MAX_RECONNECT_ATTEMPTS", "BINANCE_MAX_RECONNECT_ATTEMPTS"),
                    20,
                ),
                credentials=ExchangeCredentials(
                    api_key=_to_optional_str(
                        _first_env(
                            "EXCHANGE_API_KEY",
                            "BINANCE_API_KEY",
                            "BINANCE_FUTURES_API_KEY",
                        )
                    ),
                    api_secret=_to_optional_str(
                        _first_env(
                            "EXCHANGE_API_SECRET",
                            "BINANCE_API_SECRET",
                            "BINANCE_FUTURES_API_SECRET",
                        )
                    ),
                    passphrase=_to_optional_str(
                        _first_env(
                            "EXCHANGE_PASSPHRASE",
                            "BINANCE_PASSPHRASE",
                            "BINANCE_API_PASSPHRASE",
                        )
                    ),
                    testnet=_to_bool(
                        _first_env("EXCHANGE_TESTNET", "BINANCE_TESTNET", "BINANCE_FUTURES_TESTNET"),
                        False,
                    ),
                ),
            ),
            risk=RiskConfig(
                max_risk_per_trade_pct=_to_float(_get_env("RISK_MAX_RISK_PER_TRADE_PCT"), 1.0),
                max_daily_drawdown_pct=_to_float(_get_env("RISK_MAX_DAILY_DRAWDOWN_PCT"), 5.0),
                max_open_positions=_to_int(_get_env("RISK_MAX_OPEN_POSITIONS"), 3),
                max_total_exposure_pct=_to_float(_get_env("RISK_MAX_TOTAL_EXPOSURE_PCT"), 30.0),
                kill_switch_enabled=_to_bool(_get_env("RISK_KILL_SWITCH_ENABLED"), True),
            ),
            ai=AIConfig(
                enabled=_to_bool(_get_env("AI_ENABLED"), False),
                provider=_get_env("AI_PROVIDER", "openai") or "openai",
                model=_get_env("AI_MODEL", "gpt-5") or "gpt-5",
                api_key=_to_optional_str(_get_env("AI_API_KEY")),
                timeout_seconds=_to_float(_get_env("AI_TIMEOUT_SECONDS"), 20.0),
                max_signal_tokens=_to_int(_get_env("AI_MAX_SIGNAL_TOKENS"), 2000),
            ),
            dashboard=DashboardConfig(
                enabled=_to_bool(_get_env("DASHBOARD_ENABLED"), True),
                host=_get_env("DASHBOARD_HOST", "0.0.0.0") or "0.0.0.0",
                port=_to_int(_get_env("DASHBOARD_PORT"), 8000),
                cors_enabled=_to_bool(_get_env("DASHBOARD_CORS_ENABLED"), True),
                cors_allow_origins=_get_env("DASHBOARD_CORS_ALLOW_ORIGINS", "*") or "*",
            ),
            storage=StorageConfig(
                parquet_enabled=_to_bool(_get_env("STORAGE_PARQUET_ENABLED"), True),
                parquet_dir=_get_env("STORAGE_PARQUET_DIR", str(data_dir / "parquet")) or str(data_dir / "parquet"),
                flush_interval_seconds=_to_float(_get_env("STORAGE_FLUSH_INTERVAL_SECONDS"), 15.0),
                batch_size=_to_int(_get_env("STORAGE_BATCH_SIZE"), 1000),
            ),
        )

        config.validate()
        config.prepare_directories()

        logger.info(
            "Configuration loaded | app=%s env=%s exchange=%s debug=%s",
            config.app.name,
            config.app.env,
            config.exchange.name,
            config.app.debug,
        )
        return config

    def validate(self) -> None:
        errors: list[str] = []

        if self.event_bus.max_queue_size <= 0:
            errors.append("event_bus.max_queue_size must be > 0")

        if self.event_bus.worker_count <= 0:
            errors.append("event_bus.worker_count must be > 0")

        if self.event_bus.queue_full_policy not in {"wait", "drop_new", "drop_oldest"}:
            errors.append("event_bus.queue_full_policy must be one of: wait, drop_new, drop_oldest")

        if self.scheduler.tick_interval <= 0:
            errors.append("scheduler.tick_interval must be > 0")

        if self.redis.port <= 0:
            errors.append("redis.port must be > 0")

        if self.postgres.port <= 0:
            errors.append("postgres.port must be > 0")

        if self.postgres.pool_min_size < 0:
            errors.append("postgres.pool_min_size must be >= 0")

        if self.postgres.pool_max_size <= 0:
            errors.append("postgres.pool_max_size must be > 0")

        if self.postgres.pool_min_size > self.postgres.pool_max_size:
            errors.append("postgres.pool_min_size must be <= postgres.pool_max_size")

        if self.risk.max_risk_per_trade_pct <= 0:
            errors.append("risk.max_risk_per_trade_pct must be > 0")

        if self.risk.max_daily_drawdown_pct <= 0:
            errors.append("risk.max_daily_drawdown_pct must be > 0")

        if self.risk.max_open_positions <= 0:
            errors.append("risk.max_open_positions must be > 0")

        if self.risk.max_total_exposure_pct <= 0:
            errors.append("risk.max_total_exposure_pct must be > 0")

        if self.dashboard.port <= 0:
            errors.append("dashboard.port must be > 0")

        if self.storage.flush_interval_seconds <= 0:
            errors.append("storage.flush_interval_seconds must be > 0")

        if self.storage.batch_size <= 0:
            errors.append("storage.batch_size must be > 0")

        if errors:
            for error in errors:
                logger.error("Config validation error | error=%s", error)
            raise ValueError("Invalid configuration: " + "; ".join(errors))

    def prepare_directories(self) -> None:
        dirs = {
            self.app.data_dir,
            self.app.logs_dir,
            Path(self.storage.parquet_dir),
            Path(self.logging.log_dir),
        }

        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("Configuration directories prepared")

    @property
    def is_dev(self) -> bool:
        return self.app.env.lower() in {"dev", "local", "development"}

    @property
    def is_prod(self) -> bool:
        return self.app.env.lower() in {"prod", "production"}

    @property
    def postgres_dsn(self) -> str:
        password_part = self.postgres.password or ""
        return (
            f"postgresql://{self.postgres.user}:{password_part}"
            f"@{self.postgres.host}:{self.postgres.port}/{self.postgres.database}"
        )

    @property
    def redis_url(self) -> str:
        scheme = "rediss" if self.redis.ssl else "redis"

        auth = ""
        if self.redis.password:
            auth = f":{self.redis.password}@"

        return f"{scheme}://{auth}{self.redis.host}:{self.redis.port}/{self.redis.db}"