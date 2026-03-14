
import docker
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Type

class ShellExecutorInput(BaseModel):
    command: str = Field(description="The shell command to execute.")

class ShellExecutorTool(BaseTool):
    """A tool to safely execute shell commands in a sandboxed Docker container."""

    name: str = "system__shell_executor"
    description: str = (
        "Executes a shell command in a sandboxed Alpine Linux Docker container. "
        "The output (stdout and stderr) of the command will be returned."
    )
    args_schema: Type[BaseModel] = ShellExecutorInput

    def _run(self, command: str) -> str:
        """Execute the shell command in a Docker container."""
        try:
            client = docker.from_env()
            
            # Use a lightweight image
            image = "alpine:latest"
            
            # Ensure the image is available locally
            try:
                client.images.get(image)
            except docker.errors.ImageNotFound:
                print(f"Pulling image: {image}")
                client.images.pull(image)

            try:
                container = client.containers.run(
                    image,
                    command=f"/bin/sh -c 'apk add --no-cache curl iputils && {command}'",
                    detach=False,
                    remove=True,
                    stdout=True,
                    stderr=True,
                    # Set a timeout for the container execution
                    # Note: The 'timeout' parameter for run() is not directly available in docker-py
                    # A workaround is to use detach=True and then wait with a timeout, but for simplicity
                    # and since we want to capture output directly, we'll rely on the command timing out
                    # itself if it runs for too long. For more robust timeout, a more complex setup is needed.
                )
                
                # The result is returned as bytes, so we need to decode it
                stdout = container.decode('utf-8')
                stderr = "" # Stderr is not captured separately in this mode, it's merged with stdout
                exit_code = 0 # If run completes without error, exit code is 0
                
            except docker.errors.ContainerError as e:
                # This error is raised when the container exits with a non-zero exit code
                stdout = e.container.logs(stdout=True).decode('utf-8')
                stderr = e.container.logs(stderr=True).decode('utf-8')
                exit_code = e.exit_status

            # Format and return the output
            output = ""
            if stdout:
                output += f"--- STDOUT ---\n{stdout}\n"
            if stderr:
                output += f"--- STDERR ---\n{stderr}\n"
            
            output += f"--- Exit Code: {exit_code} ---"
            return output

        except docker.errors.DockerException as e:
            return f"A Docker error occurred: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

# Create an instance of the tool so it can be imported and registered.
system__shell_executor = ShellExecutorTool()
