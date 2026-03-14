"""
A tool for executing Python code in a sandboxed virtual environment.

This tool provides a secure way to run arbitrary Python code by:
1.  Parsing the code to detect dependencies.
2.  Creating a cached, sandboxed virtual environment using 'uv'.
3.  Installing the detected dependencies into the environment.
4.  Executing the code and returning its output.
"""
import ast
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile
import shutil
from typing import List, Type, Optional, Dict

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


# A mapping of common import names to their corresponding PyPI package names.
IMPORT_TO_PACKAGE_MAP = {
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "requests": "requests",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tensorflow": "tensorflow",
    "torch": "torch",
    "keras": "keras",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "starlette": "starlette",
    "pydantic": "pydantic",
    "sqlalchemy": "SQLAlchemy",
    "alembic": "alembic",
    "boto3": "boto3",
}

class _ImportVisitor(ast.NodeVisitor):
    """AST visitor to find all imported modules in a Python code string."""

    def __init__(self):
        self.imports = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)


def _check_dependencies(code: str) -> List[str]:
    """
    Parses Python code to detect and return a list of dependencies.

    Args:
        code: The Python code string.

    Returns:
        A sorted list of unique PyPI package names.
    """
    try:
        tree = ast.parse(code)
        visitor = _ImportVisitor()
        visitor.visit(tree)
        
        # Map import names to package names
        packages = set()
        for imp in visitor.imports:
            packages.add(IMPORT_TO_PACKAGE_MAP.get(imp, imp))
        
        # Exclude standard library modules
        std_lib_modules = set(sys.stdlib_module_names)
        
        return sorted(list(packages - std_lib_modules))

    except SyntaxError as e:
        # If the code has a syntax error, we can't parse it.
        # Return an empty list and let the execution fail.
        return []


def _get_or_create_venv(dependencies: List[str]) -> str:
    """
    Creates or retrieves a cached virtual environment for a given set of dependencies.

    Args:
        dependencies: A list of PyPI package names.

    Returns:
        The path to the Python interpreter in the virtual environment.
    """
    base_path = pathlib.Path.home() / ".radops" / "py_venvs"
    base_path.mkdir(parents=True, exist_ok=True)

    # Create a hash for the specific dependency set
    dep_string = "".join(sorted(dependencies))
    hash_id = hashlib.sha256(dep_string.encode()).hexdigest()
    venv_path = base_path / hash_id

    python_executable = venv_path / "bin" / "python"
    success_marker = venv_path / ".success"

    if python_executable.exists() and success_marker.exists():
        # Environment already exists and was set up successfully
        return str(python_executable)
        
    # Clean up any partial environment from a failed previous attempt
    if venv_path.exists():
        shutil.rmtree(venv_path, ignore_errors=True)

    # Environment does not exist, create it
    try:
        subprocess.run(["uv", "venv", str(venv_path)], check=True, capture_output=True, text=True)
        if dependencies:
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python_executable)] + dependencies,
                check=True,
                capture_output=True,
                text=True
            )
        success_marker.touch()
    except subprocess.CalledProcessError as e:
        shutil.rmtree(venv_path, ignore_errors=True) # Clean up failed attempt
        raise RuntimeError(f"Failed to create virtual environment: {e.stderr}") from e
    except Exception as e:
        shutil.rmtree(venv_path, ignore_errors=True) # Clean up on unexpected errors (e.g. uv not installed)
        raise RuntimeError(f"Unexpected error creating virtual environment: {e}") from e
    
    return str(python_executable)


class PythonExecutorInput(BaseModel):
    code: str = Field(description="The Python code to execute.")
    dependencies: Optional[List[str]] = Field(description="An optional list of package dependencies to install.")

class PythonExecutorTool(BaseTool):
    """A tool to safely execute Python code in a sandboxed environment."""

    name: str = "system__python_executor"
    description: str = (
        "Executes Python code in a sandboxed environment. "
        "It automatically detects and installs required dependencies. "
        "The code should be self-contained. "
        "The output (stdout) of the code will be returned."
    )
    args_schema: Type[BaseModel] = PythonExecutorInput

    def _run(self, code: str, dependencies: Optional[List[str]] = None) -> str:
        """Execute the code."""
        try:
            # Determine dependencies
            if dependencies:
                all_deps = sorted(list(set(dependencies)))
            else:
                all_deps = _check_dependencies(code)

            # Get or create the environment
            python_executable = _get_or_create_venv(all_deps)

            # Write code to a temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(code)
                script_path = tmp_file.name
            
            # Execute the script
            result = subprocess.run(
                [python_executable, script_path],
                capture_output=True,
                text=True,
                timeout=300 # 5-minute timeout
            )

            # Clean up the temporary file
            os.unlink(script_path)

            # Format and return the output
            output = ""
            if result.stdout:
                output += f"--- STDOUT ---\n{result.stdout}\n"
            if result.stderr:
                output += f"--- STDERR ---\n{result.stderr}\n"
            
            output += f"--- Exit Code: {result.returncode} ---"
            return output

        except Exception as e:
            return f"An unexpected error occurred: {e}"

# Create an instance of the tool so it can be imported and registered.
system__python_executor = PythonExecutorTool()
