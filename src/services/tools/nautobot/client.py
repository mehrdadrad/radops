import os
import urllib3
import pynautobot

from config.tools import tool_settings


def get_nautobot_client() -> pynautobot.core.api.Api:
    """
    Initialize and return the Nautobot client.
    """
    nautobot_settings = getattr(tool_settings, "nautobot", {})

    url = nautobot_settings.get("url") or os.environ.get("NAUTOBOT_URL")
    token = nautobot_settings.get("token") or os.environ.get("NAUTOBOT_TOKEN")
    verify_ssl = nautobot_settings.get("verify_ssl")

    if verify_ssl is None:
        verify_ssl = os.environ.get("NAUTOBOT_VERIFY_SSL", "true").lower() == "true"

    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if not url or not token:
        raise ValueError(
            "Nautobot credentials not found. "
            "Please set NAUTOBOT_URL and NAUTOBOT_TOKEN environment variables."
        )
    return pynautobot.api(url=url, token=token, verify=verify_ssl)