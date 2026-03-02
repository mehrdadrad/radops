import unittest
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
import sys
import os
from core.skill import SkillRunner

# Helper class to mock the object returned by frontmatter.load
class MockFrontmatter:
    def __init__(self, metadata, content):
        self.metadata = metadata
        self.content = content

class TestSkillRunner(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.md_path = "skills/test_skill.md"

    @patch("os.path.exists")
    def test_init_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            SkillRunner("non_existent.md")

    @patch("core.skill.frontmatter.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_init_success(self, mock_exists, mock_file, mock_fm_load):
        mock_exists.return_value = True
        mock_fm_load.return_value = MockFrontmatter({"name": "test_skill"}, "some content")
        
        runner = SkillRunner(self.md_path)
        
        self.assertEqual(runner.name, "test_skill")
        self.assertEqual(runner.content, "some content")
        self.assertEqual(runner.type, "script") # Default

    @patch("core.skill.frontmatter.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    async def test_execute_inline_python(self, mock_exists, mock_file, mock_fm_load):
        mock_exists.return_value = True
        content = """
# Test Skill
```python
import sys
name = locals().get('name', 'Unknown')
arg = sys.argv[1] if len(sys.argv) > 1 else 'NoArg'
print(f"Hello {name}, Arg: {arg}")
```
"""
        mock_fm_load.return_value = MockFrontmatter({}, content)
        
        runner = SkillRunner(self.md_path)
        
        # Test with inputs
        # Note: inputs values are passed to sys.argv. Order depends on dict iteration order.
        inputs = {"name": "World", "extra": "CLI_ARG"}
        
        result = await runner.execute(inputs=inputs)
        
        self.assertIn("Hello World", result)
        # sys.argv[1] corresponds to the first value in inputs (World)
        self.assertIn("Arg: World", result) 

    @patch("core.skill.frontmatter.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("subprocess.run")
    async def test_execute_inline_bash(self, mock_subprocess, mock_exists, mock_file, mock_fm_load):
        mock_exists.return_value = True
        content = """
```bash
echo "Hello $NAME"
```
"""
        mock_fm_load.return_value = MockFrontmatter({}, content)
        
        # Mock subprocess output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Hello Bash User"
        
        runner = SkillRunner(self.md_path)
        result = await runner.execute(inputs={"NAME": "Bash User"})
        
        self.assertEqual(result, "Hello Bash User")
        
        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        self.assertEqual(kwargs['executable'], "/bin/bash")
        self.assertIn("NAME", kwargs['env'])
        self.assertEqual(kwargs['env']['NAME'], "Bash User")

    @patch("core.skill.frontmatter.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("subprocess.run")
    async def test_execute_script_file(self, mock_subprocess, mock_exists, mock_file, mock_fm_load):
        # os.path.exists called twice: once for md file, once for script file
        mock_exists.side_effect = [True, True]
        
        metadata = {"script": "myscript.py"}
        mock_fm_load.return_value = MockFrontmatter(metadata, "")
        
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Script Output"
        
        runner = SkillRunner(self.md_path)
        result = await runner.execute(inputs={"arg": "val"})
        
        self.assertIn("Script Output", result)
        
        mock_subprocess.assert_called()
        args, kwargs = mock_subprocess.call_args
        # Check if python executable is used
        self.assertEqual(args[0][0], sys.executable)
        self.assertTrue(args[0][1].endswith("skills/myscript.py"))
        self.assertIn("arg", kwargs['env'])

    @patch("core.skill.frontmatter.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    async def test_execute_mcp(self, mock_exists, mock_file, mock_fm_load):
        mock_exists.return_value = True
        metadata = {
            "type": "mcp",
            "mcp_server": "test-server",
            "mcp_tool": "test-tool"
        }
        mock_fm_load.return_value = MockFrontmatter(metadata, "")
        
        # Mock MCP Client
        mock_client = MagicMock()
        mock_client.name = "test-server"
        mock_client.session = AsyncMock()
        
        # Mock tool result
        mock_result = MagicMock()
        mock_result.isError = False
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "MCP Output"
        mock_result.content = [mock_content]
        
        mock_client.session.call_tool.return_value = mock_result
        
        runner = SkillRunner(self.md_path)
        result = await runner.execute(inputs={"param": "value"}, mcp_clients=[mock_client])
        
        self.assertIn("MCP Output", result)
        mock_client.session.call_tool.assert_called_with("test-tool", arguments={"param": "value"})

    @patch("core.skill.frontmatter.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    async def test_execute_mixed_methods(self, mock_exists, mock_file, mock_fm_load):
        # Test that multiple methods (MCP, Script, Inline) are executed if present
        mock_exists.side_effect = [True, True] # md file, script file
        
        metadata = {
            "mcp_server": "test-server",
            "mcp_tool": "test-tool",
            "script": "myscript.py"
        }
        content = """
```python
print("Inline Output")
```
"""
        mock_fm_load.return_value = MockFrontmatter(metadata, content)
        
        # Mock MCP
        mock_client = MagicMock()
        mock_client.name = "test-server"
        mock_client.session = AsyncMock()
        mock_result = MagicMock()
        mock_result.isError = False
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "MCP Output"
        mock_result.content = [mock_content]
        mock_client.session.call_tool.return_value = mock_result
        
        # Mock Subprocess (for script)
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "Script Output"
            
            runner = SkillRunner(self.md_path)
            result = await runner.execute(inputs={}, mcp_clients=[mock_client])
            
            self.assertIn("MCP Output", result)
            self.assertIn("Script Output", result)
            self.assertIn("Inline Output", result)

    @patch("core.skill.frontmatter.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    async def test_execute_multiple_inline_blocks(self, mock_exists, mock_file, mock_fm_load):
        mock_exists.return_value = True
        content = """
```python
print("Block 1")
```
```python
print("Block 2")
```
"""
        mock_fm_load.return_value = MockFrontmatter({}, content)
        
        runner = SkillRunner(self.md_path)
        result = await runner.execute()
        
        self.assertIn("Block 1", result)
        self.assertIn("Block 2", result)

    @patch("core.skill.frontmatter.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    async def test_execute_inline_unsupported(self, mock_exists, mock_file, mock_fm_load):
        mock_exists.return_value = True
        content = """
```java
System.out.println("Hello");
```
"""
        mock_fm_load.return_value = MockFrontmatter({}, content)
        
        runner = SkillRunner(self.md_path)
        result = await runner.execute()
        
        self.assertIn("Error: Unsupported inline language 'java'", result)
