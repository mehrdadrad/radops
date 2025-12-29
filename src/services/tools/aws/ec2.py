import json
import logging
from typing import Optional

import boto3
from langchain_core.tools import tool

from config.tools import tool_settings

logger = logging.getLogger(__name__)


def get_aws_client(service_name: str):
    return boto3.client(
        service_name,
        aws_access_key_id=tool_settings.aws.access_key_id,
        aws_secret_access_key=tool_settings.aws.secret_access_key,
        region_name=tool_settings.aws.region
    )


@tool
def aws__list_ec2_instances(state: Optional[str] = None) -> str:
    """
    Lists EC2 instances in the configured region.
    Args:
        state: Optional filter for instance state (e.g., 'running', 'stopped'). Do NOT set this argument if the user wants ALL instances. Only use it if the user explicitly asks to filter by state.
    Returns:
        A JSON string containing a list of instances, each with details like InstanceId, Name, State, IPs, VpcId, and SubnetId.
    """
    ec2 = get_aws_client('ec2')
    try:
        filters = []
        if state:
            filters.append({'Name': 'instance-state-name', 'Values': [state]})

        response = ec2.describe_instances(Filters=filters)
        instances = []
        for reservation in response.get('Reservations', []):
            for instance in reservation.get('Instances', []):
                name = next((tag['Value'] for tag in instance.get('Tags', []) if tag['Key'] == 'Name'), None)
                instances.append({
                    'InstanceId': instance.get('InstanceId'),
                    'Name': name,
                    'InstanceType': instance.get('InstanceType'),
                    'State': instance.get('State', {}).get('Name'),
                    'PublicIpAddress': instance.get('PublicIpAddress'),
                    'PrivateIpAddress': instance.get('PrivateIpAddress'),
                    'VpcId': instance.get('VpcId'),
                    'SubnetId': instance.get('SubnetId'),
                    'LaunchTime': str(instance.get('LaunchTime'))
                })
        return json.dumps(instances, indent=2)
    except Exception as e:
        logger.error(f"Error in aws__list_ec2_instances: {e}")
        return json.dumps({"error": str(e)}, indent=2)


@tool
def aws__manage_ec2_instance(instance_id: str, action: str) -> str:
    """
    Manages an EC2 instance state.
    Args:
        instance_id: The ID of the EC2 instance (e.g., i-1234567890abcdef0).
        action: The action to perform. Must be one of: 'start', 'stop', 'reboot', 'terminate'.
    """
    ec2 = get_aws_client('ec2')
    try:
        action = action.lower()
        if action == 'start':
            ec2.start_instances(InstanceIds=[instance_id])
            return json.dumps({"status": "starting", "instance_id": instance_id}, indent=2)
        elif action == 'stop':
            ec2.stop_instances(InstanceIds=[instance_id])
            return json.dumps({"status": "stopping", "instance_id": instance_id}, indent=2)
        elif action == 'reboot':
            ec2.reboot_instances(InstanceIds=[instance_id])
            return json.dumps({"status": "rebooting", "instance_id": instance_id}, indent=2)
        elif action == 'terminate':
            ec2.terminate_instances(InstanceIds=[instance_id])
            return json.dumps({"status": "terminating", "instance_id": instance_id}, indent=2)
        else:
            return json.dumps({"error": f"Invalid action: {action}. Must be start, stop, reboot, or terminate."}, indent=2)
    except Exception as e:
        logger.error(f"Error in aws__manage_ec2_instance: {e}")
        return json.dumps({"error": str(e)}, indent=2)
