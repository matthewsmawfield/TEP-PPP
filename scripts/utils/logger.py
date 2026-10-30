#!/usr/bin/env python3
"""
TEP Logging Utilities
================================

Standardized logging infrastructure for the TEP-GNSS-RINEX analysis pipeline.
Provides color-coded console output, file logging, and custom log levels for
consistent, professional status reporting across all analysis steps.

Custom Log Levels:
    DETAIL  (24): Dark cyan  - verbose diagnostic / intermediate results
    PROCESS (25): Blue       - ongoing operations
    SUCCESS (26): Green      - completed operations
    TEST    (27): Magenta    - test/validation results

Key Components:
    TEPLogger:      Main logger class with console and file handlers
    TEPFormatter:   Color-coded console formatter with date+time timestamps
    TEPFileFormatter: Clean formatter for log files (no ANSI codes)
    print_status:   Global function for status messages
    section_banner: Formatted section header for major pipeline phases
    step_banner:    Full-page step header with metadata box

Usage:
    >>> from scripts.utils.logger import TEPLogger, print_status, set_step_logger
    >>> logger = TEPLogger("step_2_0", log_file_path=Path("logs/step_2_0.log"))
    >>> set_step_logger(logger)
    >>> print_status("Analysis complete", "SUCCESS")
    >>> logger.section_banner("Phase A: Data Acquisition")
    >>> logger.step_banner("Step 02", "Clock Residual Extraction",
    ...                    metadata={"Stations": "79", "Days": "7"})

Author: Matthew Lukin Smawfield
Date: 9 December 2025
License: CC-BY-4.0
"""

import logging
import sys
import time
from pathlib import Path
import os
from typing import Optional

# Project root for relative path calculations
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

# Register custom log levels so logging.addLevelName recognises them
logging.addLevelName(24, "DETAIL")
logging.addLevelName(25, "PROCESS")
logging.addLevelName(26, "SUCCESS")
logging.addLevelName(27, "TEST")

class TEPFormatter(logging.Formatter):
    """Formatter with color support for professional console output.

    Timestamps include both date and time (YYYY-MM-DD HH:MM:SS) for full
    traceability across long pipeline runs.
    """

    # ANSI color codes
    COLORS = {
        'DETAIL': '\033[0;36m',    # Dark cyan
        'SUCCESS': '\033[1;32m',   # Green bold
        'WARNING': '\033[1;33m',   # Yellow bold
        'ERROR': '\033[1;31m',     # Red bold
        'INFO': '\033[0;37m',      # White
        'DEBUG': '\033[0;90m',     # Dark gray
        'PROCESS': '\033[0;34m',   # Blue
        'TEST': '\033[1;35m',      # Magenta bold
        'TITLE': '\033[1;36m',     # Cyan bold for titles
        'CRITICAL': '\033[1;41m'   # White on red background
    }
    RESET = '\033[0m'

    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        self.use_colors = use_colors

    def format(self, record):
        message = record.getMessage()

        level_mapping = {
            24: ('DETAIL', self.COLORS['DETAIL']),
            25: ('PROCESS', self.COLORS['PROCESS']),
            26: ('SUCCESS', self.COLORS['SUCCESS']),
            27: ('TEST', self.COLORS['TEST']),
            logging.INFO: ('INFO', self.COLORS['INFO']),
            logging.WARNING: ('WARNING', self.COLORS['WARNING']),
            logging.ERROR: ('ERROR', self.COLORS['ERROR']),
            logging.DEBUG: ('DEBUG', self.COLORS['DEBUG']),
            logging.CRITICAL: ('CRITICAL', self.COLORS['CRITICAL'])
        }

        level_name, color = level_mapping.get(record.levelno, ('INFO', self.COLORS['INFO']))

        timestamp = self.formatTime(record, self.datefmt)

        if self.use_colors:
            return f"{color}[{timestamp}] [{level_name:8s}] {message}{self.RESET}"
        else:
            return f"[{timestamp}] [{level_name:8s}] {message}"

class TEPFileFormatter(logging.Formatter):
    """Clean formatter for file output without ANSI color codes.

    Timestamps include both date and time for full traceability in log files.
    """

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record):
        message = record.getMessage()

        level_mapping = {
            24: 'DETAIL',
            25: 'PROCESS',
            26: 'SUCCESS',
            27: 'TEST',
            logging.INFO: 'INFO',
            logging.WARNING: 'WARNING',
            logging.ERROR: 'ERROR',
            logging.DEBUG: 'DEBUG',
            logging.CRITICAL: 'CRITICAL'
        }

        level_name = level_mapping.get(record.levelno, 'INFO')
        timestamp = self.formatTime(record, self.datefmt)
        return f"[{timestamp}] [{level_name:8s}] {message}"

class TEPLogger:
    """Professional logging class with console + file handlers and rich formatting.

    Provides standard log levels (info, warning, error, debug, critical) plus
    custom levels (detail, process, success, test) and high-level formatting
    helpers (section_banner, step_banner, summary_box).
    """

    def __init__(self, name: str = "tep_gnss", level: str = "INFO",
                 log_file_path: Optional[Path] = None, reset_log: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_log_level(level))
        self._step_name = name
        self._step_start_time = None
        self._step_metadata = {}

        # Clear any existing handlers to prevent duplication
        self.logger.handlers.clear()

        # Console handler with immediate flushing
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.stream.reconfigure(line_buffering=True)
        ch.setFormatter(TEPFormatter())
        self.logger.addHandler(ch)
        self.logger.propagate = False

        # File handler
        if log_file_path is None:
            default_log_dir = PACKAGE_ROOT / "logs"
            default_log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = default_log_dir / "general_tep_gnss.log"
            reset_log = False

        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        if reset_log:
            try:
                with open(log_file_path, 'w') as f:
                    f.write("")
            except Exception as e:
                print(f"Warning: Could not reset log file {log_file_path}: {e}")

        fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(TEPFileFormatter())
        self.logger.addHandler(fh)

    def _get_log_level(self, level_name: str):
        return getattr(logging, level_name.upper(), logging.INFO)

    def _flush(self):
        for handler in self.logger.handlers:
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
                handler.stream.flush()

    # --- Standard log levels ---

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def critical(self, message: str):
        self.logger.critical(message)

    # --- Custom log levels ---

    def detail(self, message: str):
        """Verbose diagnostic output (level 24, between INFO and PROCESS)."""
        self.logger.log(24, message)
        self._flush()

    def process(self, message: str):
        """Ongoing operation status (level 25)."""
        self.logger.log(25, message)
        self._flush()

    def success(self, message: str):
        """Successful completion message (level 26)."""
        self.logger.log(26, message)

    def test(self, message: str):
        """Test or validation result (level 27)."""
        self.logger.log(27, message)

    def debug_msg(self, message: str):
        self.logger.debug(message)

    # --- High-level formatting helpers ---

    def section_banner(self, title: str, width: int = 80):
        """Print a formatted section header with decorative borders.

        Parameters
        ----------
        title : str
            Section title to display.
        width : int
            Total banner width in characters.
        """
        bar = '=' * width
        pad = max(0, (width - len(title)) // 2)
        centered = ' ' * pad + title
        self.logger.info('')
        self.logger.info(bar)
        self.logger.info(centered)
        self.logger.info(bar)
        self.logger.info('')
        self._flush()

    def step_banner(self, step_id: str, title: str,
                    metadata: Optional[dict] = None,
                    width: int = 80):
        """Print a full step header with metadata box.

        Parameters
        ----------
        step_id : str
            Short identifier, e.g. "Step 04".
        title : str
            Human-readable step title.
        metadata : dict, optional
            Key-value pairs displayed in a bordered metadata box.
        width : int
            Total banner width.
        """
        self._step_name = step_id
        self._step_start_time = time.time()
        self._step_metadata = metadata or {}

        bar = '=' * width
        inner = '=' * (width - 4)
        self.logger.info('')
        self.logger.info(bar)
        self.logger.info(f"  {step_id}: {title}")
        self.logger.info(bar)
        if metadata:
            self.logger.info(f"  +{'-' * (width - 6)}+")
            for key, val in metadata.items():
                line = f"  | {key:<20s} {val}"
                self.logger.info(line)
            self.logger.info(f"  +{'-' * (width - 6)}+")
        self.logger.info('')
        self._flush()

    def summary_box(self, title: str, items: list,
                    width: int = 80):
        """Print a summary box with key-value items at step completion.

        Parameters
        ----------
        title : str
            Summary section title.
        items : list of (str, str)
            Key-value pairs to display.
        width : int
            Total box width.
        """
        bar = '=' * width
        self.logger.info('')
        self.logger.info(bar)
        pad = max(0, (width - len(title)) // 2)
        self.logger.info(' ' * pad + title)
        self.logger.info(bar)
        for key, val in items:
            self.logger.info(f"  {key:<36s} {val}")
        elapsed = ''
        if self._step_start_time is not None:
            elapsed_s = time.time() - self._step_start_time
            if elapsed_s >= 60:
                elapsed = f"{elapsed_s/60:.1f} min"
            else:
                elapsed = f"{elapsed_s:.1f} s"
            self.logger.info(f"  {'Elapsed':<36s} {elapsed}")
        self.logger.info(bar)
        self.logger.info('')
        self._flush()

# Global variable to track the current step logger (set by each step script)
# This is used by print_status to know which logger to use
_current_step_logger = None

def set_step_logger(logger: TEPLogger):
    """Set the current step logger that print_status will use."""
    global _current_step_logger
    _current_step_logger = logger

# Module-level logger instance for backward compatibility
_global_teplogger_instance = None

# Expose print_status and check_memory_usage as global functions
def print_status(message: str, level: str = "INFO") -> None:
    """Print a status message to the console and log to the current step logger.

    This function is the primary interface for all pipeline status output.
    Supported levels: INFO, DETAIL, PROCESS, SUCCESS, WARNING, ERROR,
    DEBUG, TEST, TITLE, CRITICAL.
    """
    if _current_step_logger is not None:
        if level == "SUCCESS":
            _current_step_logger.success(message)
        elif level == "ERROR":
            _current_step_logger.error(message)
        elif level == "WARNING":
            _current_step_logger.warning(message)
        elif level == "PROCESS":
            _current_step_logger.process(message)
        elif level == "DETAIL":
            _current_step_logger.detail(message)
        elif level == "DEBUG":
            _current_step_logger.debug(message)
        elif level == "TEST":
            _current_step_logger.test(message)
        elif level == "TITLE":
            _current_step_logger.info("")
            _current_step_logger.info("="*80)
            _current_step_logger.info(message)
            _current_step_logger.info("="*80)
            _current_step_logger.info("")
        else:
            _current_step_logger.info(message)
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatter_instance = TEPFormatter()
        color = formatter_instance.COLORS.get(level, '')
        reset = formatter_instance.RESET

        if level == "TITLE":
            print(f"\n{color}{'='*80}{reset}")
            print(f"{color}{message}{reset}")
            print(f"{color}{'='*80}{reset}\n")
        else:
            print(f"{color}[{timestamp}] [{level:8s}] {message}{reset}")

    sys.stdout.flush()

def reset_master_log() -> None:
    """Legacy function retained for backward compatibility. No-op."""
    pass

def check_memory_usage(context: str = "Unknown") -> None:
    """
    Logs the current memory usage of the process using the current step logger.
    This function is exposed for consistent use across all scripts.
    """
    import psutil
    import gc

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)
    
    if _current_step_logger is not None:
        _current_step_logger.debug(f"Memory usage in {context}: RSS={rss_mb:.2f} MB, VMS={vms_mb:.2f} MB")
    else:
        print(f"[DEBUG] Memory usage in {context}: RSS={rss_mb:.2f} MB, VMS={vms_mb:.2f} MB")
    gc.collect()
