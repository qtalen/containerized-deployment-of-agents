from pathlib import Path
import inspect

def get_project_root() -> Path:
    """
    Get the project's root directory, which is the folder containing the .git directory.
    :return:
    """
    current_path = Path(__file__).resolve()

    for parent in [current_path, *current_path.parents]:
        if Path(parent / ".git").exists():
            return parent

    raise FileNotFoundError("I can not locate the root path through .git")

def get_current_directory() -> Path:
    """
    Get the directory where the Python file that is calling this method is located.
    :return:
    """
    current_frame = inspect.currentframe()
    if current_frame is None:
        raise ValueError("Can not locate current frame")

    caller_frame = current_frame.f_back
    if caller_frame is None:
        raise ValueError("Can not locate caller frame")

    caller_file = caller_frame.f_code.co_filename
    return Path(caller_file).resolve().parent