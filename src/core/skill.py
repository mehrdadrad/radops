import io
import logging
import sys
import frontmatter
import subprocess
import os
from typing import Any, Dict, List
from markdown_it import MarkdownIt

logger = logging.getLogger(__name__)

class SkillRunner:
    def __init__(self, md_path: str):
        self.md_path = md_path
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Skill file not found: {md_path}")

        with open(md_path, "r", encoding="utf-8") as f:
            self.parsed_file = frontmatter.load(f)
            
        self.metadata = self.parsed_file.metadata
        self.content = self.parsed_file.content
        self.name = self.metadata.get("name", "Unknown Skill")
        self.description = self.metadata.get("description", "")
        self.type = self.metadata.get("type", "script")  # script, mcp, inline
        
    def _extract_inline_code(self) -> List[Dict[str, str]]:
        """Parses the Markdown AST to find all code blocks."""
        md = MarkdownIt()
        tokens = md.parse(self.content)
        
        blocks = []
        for token in tokens:
            if token.type == "fence": # 'fence' is a ``` code block
                blocks.append({"language": token.info.strip(), "code": token.content})
        return blocks

    async def execute(self, inputs: Dict[str, Any] = None, mcp_clients: List[Any] = None) -> str:
        """
        Executes the skill based on its configuration.

        Args:
            inputs: A dictionary of input variables.
            mcp_clients: A list of available MCPClient instances (required for MCP skills).
        """
        inputs = inputs or {}
        results = []
        executed = False

        # MCP Call
        if self.type == "mcp" or "mcp_tool" in self.metadata:
            results.append(await self._execute_mcp(inputs, mcp_clients))
            executed = True

        # External Script
        if "script" in self.metadata:
            results.append(self._execute_script_file(inputs))
            executed = True

        # Inline Code
        blocks = self._extract_inline_code()
        if blocks:
            results.append(self._execute_inline(inputs))
            executed = True

        if not executed:
            return "Error: No valid execution method found (MCP, Script, or Inline code)."

        return "\n".join(results)

    async def _execute_mcp(self, inputs: Dict[str, Any], mcp_clients: List[Any]) -> str:
        """Executes an MCP tool call."""
        if not mcp_clients:
            return "Error: No MCP clients provided for MCP skill execution."

        server_name = self.metadata.get("mcp_server")
        tool_name = self.metadata.get("mcp_tool")

        if not server_name or not tool_name:
            return "Error: 'mcp_server' and 'mcp_tool' must be defined in metadata for MCP skills."

        client = next((c for c in mcp_clients if c.name.lower() == server_name.lower()), None)
        if not client:
            return f"Error: MCP server '{server_name}' not found."

        if not client.session:
            return f"Error: MCP server '{server_name}' is not connected."

        try:
            logger.info("Executing MCP skill %s on server %s", tool_name, server_name)
            result = await client.session.call_tool(tool_name, arguments=inputs)

            output = []
            if hasattr(result, "content"):
                for item in result.content:
                    if item.type == "text":
                        output.append(item.text)
            
            if result.isError:
                return f"MCP Tool Error: {output}"

            return "\n".join(output) if output else "Success (No output)"

        except Exception as e:
            logger.error("MCP execution failed: %s", e)
            return f"Error executing MCP tool: {e}"

    def _execute_script_file(self, inputs: Dict[str, Any]) -> str:
        """Executes an external script file referenced in metadata."""
        script_rel_path = self.metadata["script"]
        base_dir = os.path.dirname(self.md_path)
        script_path = os.path.join(base_dir, script_rel_path)

        if not os.path.exists(script_path):
            return f"Error: Script file not found at {script_path}"

        # Pass inputs as environment variables
        env = os.environ.copy()
        for k, v in inputs.items():
            env[str(k)] = str(v)

        try:
            cmd = []
            if script_path.endswith(".py"):
                cmd = [sys.executable, script_path]
            elif script_path.endswith(".sh"):
                cmd = ["/bin/bash", script_path]
            else:
                if os.access(script_path, os.X_OK):
                    cmd = [script_path]
                else:
                    return f"Error: Cannot determine interpreter for {script_rel_path}"

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                return f"Script Error (Exit Code {result.returncode}):\n{result.stderr}"
            
            return result.stdout.strip()

        except Exception as e:
            return f"Error executing script: {e}"

    def _execute_inline(self, inputs: Dict[str, Any]) -> str:
        """Executes inline code found in the markdown."""
        blocks = self._extract_inline_code()

        if not blocks:
            return "Error: No inline code block found."

        results = []
        for block in blocks:
            code = block["code"]
            language = block["language"]
            if language == "python":
                results.append(self._execute_inline_python(code, inputs))
            elif language in ["bash", "sh", "shell"]:
                results.append(self._execute_inline_bash(code, inputs))
            else:
                results.append(f"Error: Unsupported inline language '{language}'.")
        
        return "\n".join(results)

    def _execute_inline_python(self, code: str, inputs: Dict[str, Any]) -> str:
        """Executes inline Python code using exec() with captured stdout."""
        output_capture = io.StringIO()
        
        # Inject inputs into local scope
        exec_locals = inputs.copy()
        exec_globals = {}

        original_stdout = sys.stdout
        sys.stdout = output_capture
        
        # Mock sys.argv for scripts that rely on it
        original_argv = sys.argv
        sys.argv = ["skill_script"]
        if inputs:
            sys.argv.extend([str(v) for v in inputs.values()])

        try:
            exec(code, exec_globals, exec_locals)
        except Exception as e:
            return f"Python Execution Error: {e}"
        finally:
            sys.stdout = original_stdout
            sys.argv = original_argv

        return output_capture.getvalue().strip()

    def _execute_inline_bash(self, code: str, inputs: Dict[str, Any]) -> str:
        """Executes inline Bash code."""
        env = os.environ.copy()
        for k, v in inputs.items():
            env[str(k)] = str(v)

        try:
            result = subprocess.run(
                code,
                shell=True,
                executable="/bin/bash",
                env=env,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return f"Bash Error: {result.stderr}"
            return result.stdout.strip()
        except Exception as e:
            return f"Bash Execution Error: {e}"