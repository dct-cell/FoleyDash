import logging
import sys
from pathlib import Path

import colorlog
from tqdm import tqdm

from .distribute import local_rank

from os import PathLike


class DistributedLogger(logging.Logger):
    def __init__(
        self,
        name: str,
        level=logging.NOTSET,
        # Different ranks should output to different log_files.
        # If only rank0 outputs, prefer setting log_file=None for all other ranks
        log_file: PathLike = None,
        rank0_only: bool = False,
        redirect_stderr: bool = False,
        redirect_stdout: bool = False,
        formatter: str | logging.Formatter = None,
        formatter_console: str | logging.Formatter = None,
        formatter_file: str | logging.Formatter = None,
        datefmt: str = "%Y-%m-%d %H:%M:%S",
    ):
        """
        Args:
            name (str): Logger name that may work as part of output format.
            level: Logger level so that messages with higher level of it will output.
            log_file: Path to log file. If set to None, no output to files.
            rank0_only: If set to True, messages from locak_rank != 0 will not output.
            redirect_stderr: Redirect stderr (e.g. raise RuntimeError) to log messages.
            redirect_stdout: Redirect stdout (e.g. print) to log messages. Note that logger outputs to console whatsoever.
            formatter: Formatter.
            formatter_console:
            formatter_file:
            datefmt:
        """
        super().__init__(name, level)
        self._rank = local_rank
        self._rank0_only = rank0_only
        self._filter = None
        self.rank0_only(self._rank0_only)

        # Set formatter_xxx to formatter (as default) if the former is omitted
        if formatter is not None:
            if formatter_console is None:
                formatter_console = formatter
            if formatter_file is None:
                formatter_file = formatter
        self._formatter_console = self._construct_formatter(formatter_console, datefmt)
        self._formatter_file = self._construct_formatter(formatter_file, datefmt)
        handler_console = logging.StreamHandler(sys.__stdout__)
        if self._formatter_console is not None:
            handler_console.setFormatter(self._formatter_console)
        handler_console.setLevel(level)
        self.addHandler(handler_console)
        if log_file is not None:
            handler_file = logging.FileHandler(log_file, encoding="utf-8")
            handler_file.setLevel(level)
            if self._formatter_file is not None:
                handler_file.setFormatter(self._formatter_file)
            self.addHandler(handler_file)
        if redirect_stderr:
            sys.stderr = _StreamToLogger(self.error)
        if redirect_stdout:
            sys.stdout = _StreamToLogger(self.info)
        # print(1)
        # sleep(1)

    @staticmethod
    def _construct_formatter(
        formatter: str | logging.Formatter = None,
        datefmt: str = "%Y-%m-%d %H:%M:%S",
    ) -> logging.Formatter | None:
        match formatter:
            case f if f is None or isinstance(f, logging.Formatter):
                return formatter
            case str():
                return logging.Formatter(fmt=formatter, datefmt=datefmt)
            case _:
                raise TypeError(f"formatter type {type(formatter)} not supported!")

    def rank0_only(self, enable: bool = True):
        self._rank0_only = enable
        if self._filter is not None:
            self.removeFilter(self._filter)
        self._filter = _Rank0Filter(self._rank0_only)
        self.addFilter(self._filter)


class _Rank0Filter(logging.Filter):
    def __init__(self, rank0_only=True):
        super().__init__()
        self._rank = local_rank
        self._rank0_only = rank0_only

    def filter(self, record):
        return (not self._rank0_only) or (self._rank == 0)


class _StreamToLogger:
    """Redirect write() to logger"""

    def __init__(self, log_func):
        self.log_func = log_func
        self._buffer = ""

    def write(self, message):
        # buffer message until '\n' occurs
        self._buffer += message
        if "\n" in self._buffer:
            lines = self._buffer.splitlines(True)
            for line in lines[:-1]:
                self.log_func(line.rstrip("\n"))
            self._buffer = lines[-1]

    def flush(self):
        if self._buffer:
            self.log_func(self._buffer)
            self._buffer = ""


def create_logger():
    color_time = "cyan"
    color_name = "purple"
    formatter = colorlog.ColoredFormatter(
        fmt=f"[%({color_time})s%(asctime)s%(reset)s][%({color_name})s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    log = DistributedLogger(
        "test_logger",
        log_file=None,
        rank0_only=True,
        redirect_stderr=True,
        redirect_stdout=True,
        formatter=formatter,
        formatter_file="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    return log


_logger = create_logger()


def get_logger():
    return _logger


def distributed_tqdm(*args, **kwargs):
    return tqdm(*args, position=local_rank, file=sys.__stderr__, **kwargs)


if __name__ == "__main__":
    log = get_logger()

    log.info("test DistrubutedLogger")
    print(123)
    try:
        raise TypeError("type error")
    except Exception as e:
        print(e)
    raise RuntimeError("test error")
