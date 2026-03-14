
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import subprocess

from src.services.tools.executors import shell_executor
from src.services.tools.executors import python_executor


class TestShellExecutorTool(unittest.TestCase):

    @patch('docker.from_env')
    def test_run_command_success(self, mock_from_env):
        # Arrange
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_from_env.return_value = mock_client
        mock_client.containers.run.return_value = mock_container
        mock_container.decode.return_value = "hello world"

        tool = shell_executor.ShellExecutorTool()
        command = "echo 'hello world'"

        # Act
        result = tool._run(command)

        # Assert
        mock_client.containers.run.assert_called_once_with(
            "alpine:latest",
            command=f"/bin/sh -c 'apk add --no-cache curl iputils && {command}'",
            detach=False,
            remove=True,
            stdout=True,
            stderr=True,
        )
        self.assertIn("""--- STDOUT ---
hello world
""", result)
        self.assertIn("--- Exit Code: 0 ---", result)

    @patch('docker.from_env')
    def test_run_command_error(self, mock_from_env):
        # Arrange
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        
        # Simulate a container error
        mock_container = MagicMock()
        mock_container.logs.return_value.decode.return_value = "error message"

        from docker.errors import ContainerError
        mock_client.containers.run.side_effect = ContainerError(
            mock_container, 1, f"/bin/sh -c 'bad command'", "alpine:latest", "error message"
        )

        tool = shell_executor.ShellExecutorTool()
        command = "bad command"

        # Act
        result = tool._run(command)

        # Assert
        mock_client.containers.run.assert_called_once_with(
            "alpine:latest",
            command=f"/bin/sh -c 'apk add --no-cache curl iputils && {command}'",
            detach=False,
            remove=True,
            stdout=True,
            stderr=True,
        )
        self.assertIn("""--- STDERR ---
error message
""", result)
        self.assertIn("--- Exit Code: 1 ---", result)

    @patch('docker.from_env')
    def test_docker_not_found(self, mock_from_env):
        # Arrange
        from docker.errors import DockerException
        mock_from_env.side_effect = DockerException("Docker not found")

        tool = shell_executor.ShellExecutorTool()
        command = "any command"

        # Act
        result = tool._run(command)

        # Assert
        self.assertIn("A Docker error occurred: Docker not found", result)


class TestPythonExecutorHelpers(unittest.TestCase):
    """Tests for helper functions in python_executor."""

    def test_check_dependencies(self):
        # Test case 1: Standard library
        code = "import os\nimport sys"
        self.assertEqual(python_executor._check_dependencies(code), [])

        # Test case 2: Common packages
        code = "import numpy as np\nimport pandas as pd"
        self.assertEqual(python_executor._check_dependencies(code), ["numpy", "pandas"])

        # Test case 3: Mapped packages
        code = "from PIL import Image\nimport sklearn"
        self.assertEqual(python_executor._check_dependencies(code), ["Pillow", "scikit-learn"])

        # Test case 4: Mixed and duplicates
        code = "import numpy\nimport numpy as np\nfrom os import path"
        self.assertEqual(python_executor._check_dependencies(code), ["numpy"])

        # Test case 5: Syntax error
        code = "import numpy as"
        self.assertEqual(python_executor._check_dependencies(code), [])


class TestPythonExecutorTool(unittest.TestCase):
    """Tests for the PythonExecutorTool class itself."""

    @patch('src.services.tools.executors.python_executor._get_or_create_venv')
    @patch('subprocess.run')
    @patch('os.unlink')
    @patch('tempfile.NamedTemporaryFile')
    def test_run_success(self, mock_tempfile, mock_unlink, mock_subprocess, mock_get_venv):
        # Arrange
        mock_get_venv.return_value = "/fake/venv/bin/python"
        
        mock_process_result = MagicMock()
        mock_process_result.stdout = "Success output"
        mock_process_result.stderr = ""
        mock_process_result.returncode = 0
        mock_subprocess.return_value = mock_process_result

        # Mock the temporary file context manager
        mock_file_obj = MagicMock()
        mock_file_obj.name = "/tmp/fake_script.py"
        mock_temp_context = MagicMock()
        mock_temp_context.__enter__.return_value = mock_file_obj
        mock_temp_context.__exit__.return_value = None
        mock_tempfile.return_value = mock_temp_context

        tool = python_executor.PythonExecutorTool()
        code = "print('Success output')"

        # Act
        result = tool._run(code)

        # Assert
        mock_get_venv.assert_called_once()
        mock_subprocess.assert_called_once_with(
            ["/fake/venv/bin/python", "/tmp/fake_script.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        mock_unlink.assert_called_once_with("/tmp/fake_script.py")
        self.assertIn("--- STDOUT ---\nSuccess output", result)
        self.assertIn("--- Exit Code: 0 ---", result)
        self.assertNotIn("--- STDERR ---", result)

    @patch('src.services.tools.executors.python_executor._check_dependencies')
    @patch('src.services.tools.executors.python_executor._get_or_create_venv')
    def test_run_uses_explicit_dependencies(self, mock_get_venv, mock_check_deps):
        # Arrange
        # We need to patch subprocess.run as well, even if it's not the focus of this test
        with patch('subprocess.run'), patch('os.unlink'), patch('tempfile.NamedTemporaryFile'):
            tool = python_executor.PythonExecutorTool()
            code = "import numpy"
            deps = ["numpy", "pandas"]

            # Act
            tool._run(code, dependencies=deps)

            # Assert
            mock_check_deps.assert_not_called()
            mock_get_venv.assert_called_once_with(["numpy", "pandas"])

    @patch('src.services.tools.executors.python_executor._get_or_create_venv')
    def test_run_venv_creation_fails(self, mock_get_venv):
        # Arrange
        mock_get_venv.side_effect = RuntimeError("Failed to create venv")
        tool = python_executor.PythonExecutorTool()
        code = "print('hello')"

        # Act
        result = tool._run(code)

        # Assert
        self.assertIn("An unexpected error occurred: Failed to create venv", result)

    @patch('src.services.tools.executors.python_executor._get_or_create_venv')
    @patch('subprocess.run')
    @patch('os.unlink')
    @patch('tempfile.NamedTemporaryFile')
    def test_run_script_failure(self, mock_tempfile, mock_unlink, mock_subprocess, mock_get_venv):
        # Arrange
        mock_get_venv.return_value = "/fake/venv/bin/python"
        
        # Mock a failed subprocess result
        mock_process_result = MagicMock()
        mock_process_result.stdout = ""
        mock_process_result.stderr = "Traceback: File not found"
        mock_process_result.returncode = 1
        mock_subprocess.return_value = mock_process_result

        mock_file_obj = MagicMock()
        mock_file_obj.name = "/tmp/fake_script.py"
        mock_temp_context = MagicMock()
        mock_temp_context.__enter__.return_value = mock_file_obj
        mock_temp_context.__exit__.return_value = None
        mock_tempfile.return_value = mock_temp_context

        tool = python_executor.PythonExecutorTool()
        code = "import non_existent_module"

        # Act
        result = tool._run(code)

        # Assert
        self.assertIn("--- STDERR ---\nTraceback: File not found", result)
        self.assertIn("--- Exit Code: 1 ---", result)
        self.assertNotIn("--- STDOUT ---", result)
        mock_unlink.assert_called_once_with("/tmp/fake_script.py")


class TestGetOrCreateVenv(unittest.TestCase):
    """Tests for the _get_or_create_venv helper function."""

    @patch('src.services.tools.executors.python_executor.shutil.rmtree')
    @patch('src.services.tools.executors.python_executor.subprocess.run')
    @patch('src.services.tools.executors.python_executor.pathlib.Path')
    def test_venv_exists_cache_hit(self, mock_path_class, mock_subprocess, mock_rmtree):
        # Arrange
        mock_venv_path = MagicMock(name="venv_path")
        mock_path_class.home.return_value.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_venv_path

        mock_python_executable = MagicMock(name="python_executable")
        mock_python_executable.exists.return_value = True
        
        mock_success_marker = MagicMock(name="success_marker")
        mock_success_marker.exists.return_value = True

        mock_bin_path = MagicMock(name="bin_path")
        mock_bin_path.__truediv__.return_value = mock_python_executable
        
        def venv_path_truediv_side_effect(arg):
            if arg == "bin":
                return mock_bin_path
            if arg == ".success":
                return mock_success_marker
            return MagicMock()
            
        mock_venv_path.__truediv__.side_effect = venv_path_truediv_side_effect

        # Act
        result = python_executor._get_or_create_venv(["numpy"])

        # Assert
        self.assertEqual(result, str(mock_python_executable))
        mock_subprocess.assert_not_called()

    @patch('src.services.tools.executors.python_executor.shutil.rmtree')
    @patch('src.services.tools.executors.python_executor.subprocess.run')
    @patch('src.services.tools.executors.python_executor.pathlib.Path')
    def test_venv_creation_success(self, mock_path_class, mock_subprocess, mock_rmtree):
        # Arrange
        mock_venv_path = MagicMock(name="venv_path")
        mock_venv_path.exists.return_value = False # For the cleanup check
        mock_path_class.home.return_value.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_venv_path

        mock_python_executable = MagicMock(name="python_executable")
        mock_python_executable.exists.return_value = False
        
        mock_success_marker = MagicMock(name="success_marker")
        mock_success_marker.exists.return_value = False
        mock_success_marker.touch.return_value = None

        mock_bin_path = MagicMock(name="bin_path")
        mock_bin_path.__truediv__.return_value = mock_python_executable
        
        def venv_path_truediv_side_effect(arg):
            if arg == "bin":
                return mock_bin_path
            if arg == ".success":
                return mock_success_marker
            return MagicMock()
            
        mock_venv_path.__truediv__.side_effect = venv_path_truediv_side_effect

        # Act
        python_executor._get_or_create_venv(["numpy", "pandas"])

        # Assert
        self.assertEqual(mock_subprocess.call_count, 2)
        mock_subprocess.assert_any_call(["uv", "venv", str(mock_venv_path)], check=True, capture_output=True, text=True)
        mock_subprocess.assert_any_call(
            ["uv", "pip", "install", "--python", str(mock_python_executable)] + ["numpy", "pandas"],
            check=True,
            capture_output=True,
            text=True
        )
        mock_success_marker.touch.assert_called_once()

    @patch('src.services.tools.executors.python_executor.shutil.rmtree')
    @patch('src.services.tools.executors.python_executor.subprocess.run')
    @patch('src.services.tools.executors.python_executor.pathlib.Path')
    def test_venv_creation_fails(self, mock_path_class, mock_subprocess, mock_rmtree):
        # Arrange
        mock_venv_path = MagicMock(name="venv_path")
        mock_venv_path.exists.return_value = True # Let's say it exists before failure
        mock_path_class.home.return_value.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_venv_path

        mock_python_executable = MagicMock(name="python_executable")
        mock_python_executable.exists.return_value = False
        mock_success_marker = MagicMock(name="success_marker")
        mock_success_marker.exists.return_value = False

        mock_bin_path = MagicMock(name="bin_path")
        mock_bin_path.__truediv__.return_value = mock_python_executable
        
        mock_venv_path.__truediv__.side_effect = lambda arg: mock_bin_path if arg == "bin" else mock_success_marker

        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "uv", stderr="uv failed")

        # Act & Assert
        with self.assertRaises(RuntimeError) as cm:
            python_executor._get_or_create_venv(["numpy"])
        
        # Check that cleanup is called twice: once for the pre-existing
        # directory, and once after the failed creation attempt.
        self.assertEqual(mock_rmtree.call_count, 2)
