import time
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

# PROGRESS BAR
def get_progress_bar(accelerator):
    return Progress(
    TextColumn("[bold blue]{task.completed:g}/{task.total:g}"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    "[green]Loss: {task.fields[loss]:.3f}",
    "•",
    "[red]Speed: {task.fields[speed]:.2f}s/it",
    "•",
    TimeRemainingColumn(),
    disable=not accelerator.is_local_main_process
)
