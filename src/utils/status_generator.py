import re

class StatusGenerator:
    # 1. Broad Categories for Icons (Easy to maintain)
    ICONS = {
        "aws": "☁️", "ec2": "☁️", "lambda": "⚡",
        "network": "🌐", "ping": "📡", "traceroute": "🗺️",
        "jira": "🎫", "ticket": "🎫",
        "github": "🐙", "git": "🐙",
        "pagerduty": "md", "alert": "md",
        "notion": "md", "kb": "📚",
        "system": "⚙️", "mcp": "🔌" # Generic icon for MCP tools
    }

    # 2. Action Mapping (Noun -> Progressive Verb)
    VERBS = {
        "list": "Listing",
        "get": "Fetching",
        "read": "Reading",
        "search": "Searching",
        "create": "Creating",
        "update": "Updating",
        "delete": "Deleting",
        "ping": "Pinging",
        "run": "Running",
        "check": "Checking"
    }

    @staticmethod
    def parse_tool_call(tool_name: str, tool_args: dict) -> str:
        """
        Dynamically generates a human-friendly status string.
        """
        # --- Step A: Determine Icon ---
        # Look for keywords in the tool name (e.g., 'aws' in 'aws_list_instances')
        icon = "⚙️" # Default
        for key, emoji in StatusGenerator.ICONS.items():
            if key in tool_name.lower():
                icon = emoji
                break
        
        # --- Step B: Determine Action (Verb) ---
        # Split "aws_list_ec2" -> ["aws", "list", "ec2"]
        parts = re.split(r'[__]', tool_name) 
        action = "Processing" # Default
        
        for part in parts:
            if part.lower() in StatusGenerator.VERBS:
                action = StatusGenerator.VERBS[part.lower()]
                break
        
        # --- Step C: Determine Subject (The "Noun") ---
        # Use the tool name to find what we are acting on, excluding the verb
        # e.g. "aws_list_ec2" -> "Ec2"
        clean_name = tool_name.replace("_", " ").replace("__", " ").title()
        # (Optional: specialized logic to strip 'Aws' or 'Mcp' if you want cleaner text)

        # --- Step D: Add "Intelligence" (Context from Args) ---
        # Grab the first meaningful argument to make it look smart
        context = ""
        if tool_args:
            # Get first value that is a string and short enough to show
            for k, v in tool_args.items():
                if isinstance(v, str) and len(v) < 20:
                    context = f": {v}" 
                    break
                elif isinstance(v, list):
                    context = f": {len(v)} items"
                    break

        # --- Final Assembly ---
        # "☁️ Listing Aws Ec2 Instances: i-12345..."
        return f"{icon} {action} {clean_name}{context}..."