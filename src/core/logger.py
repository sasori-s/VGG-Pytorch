
import logging
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import os

os.makedirs('logs', exist_ok=True)

# =========================
# Generate Run ID
# =========================
RUN_ID = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

class RunIdFilter(logging.Filter):
    def filter(self, record):
        record.run_id = RUN_ID
        return True

# =========================
# Logger
# =========================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =========================
# Formatter
# =========================
formatter = logging.Formatter(
    '[%(run_id)s] '
    '%(levelname)s '
    '%(filename)s:%(lineno)d - %(funcName)s - %(message)s'
)

run_id_filter = RunIdFilter()

# =========================
# Console Handler
# =========================
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
console_handler.addFilter(run_id_filter)

# =========================
# INFO File Handler (INFO only)
# =========================
info_handler = TimedRotatingFileHandler(
    "logs/info.log",
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)
info_handler.addFilter(run_id_filter)

class InfoOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

info_handler.addFilter(InfoOnlyFilter())

# =========================
# ERROR File Handler (WARNING+)
# =========================
error_handler = TimedRotatingFileHandler(
    "logs/error.log",
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)
error_handler.setLevel(logging.WARNING)
error_handler.setFormatter(formatter)
error_handler.addFilter(run_id_filter)

# =========================
# Attach handlers
# =========================
logger.addHandler(console_handler)
logger.addHandler(info_handler)
logger.addHandler(error_handler)
