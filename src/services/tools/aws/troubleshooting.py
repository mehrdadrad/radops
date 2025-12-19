import json
import logging
from typing import List

import boto3
from botocore.exceptions import ClientError
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
def get_cloudformation_stack_events(stack_name: str) -> str:
    """
    Retrieves the recent events for a CloudFormation stack, which is useful for diagnosing 'ROLLBACK_COMPLETE' failures.
    Args:
        stack_name: The name or unique stack ID of the CloudFormation stack.
    Returns:
        A JSON string containing a list of the most recent stack events, highlighting any failures.
    """
    cfn = get_aws_client('cloudformation')
    try:
        response = cfn.describe_stack_events(StackName=stack_name)
        events = []
        for event in response.get('StackEvents', []):
            events.append({
                'Timestamp': str(event.get('Timestamp')),
                'LogicalResourceId': event.get('LogicalResourceId'),
                'ResourceType': event.get('ResourceType'),
                'ResourceStatus': event.get('ResourceStatus'),
                'ResourceStatusReason': event.get('ResourceStatusReason')
            })
        # Return the most recent 15 events to keep it concise
        return json.dumps(events[:15], indent=2)
    except ClientError as e:
        logger.error(f"Error in get_cloudformation_stack_events: {e}")
        return json.dumps({"error": str(e)}, indent=2)


@tool
def get_target_group_health(target_group_arn: str) -> str:
    """
    Checks the health of registered targets in an Elastic Load Balancer (ELB) target group. Useful for diagnosing 'Unhealthy' instances.
    Args:
        target_group_arn: The Amazon Resource Name (ARN) of the target group.
    Returns:
        A JSON string describing the health status of each target in the group.
    """
    elbv2 = get_aws_client('elbv2')
    try:
        response = elbv2.describe_target_health(TargetGroupArn=target_group_arn)
        return json.dumps(response.get('TargetHealthDescriptions', []), default=str, indent=2)
    except ClientError as e:
        logger.error(f"Error in get_target_group_health: {e}")
        return json.dumps({"error": str(e)}, indent=2)


@tool
def simulate_iam_policy(principal_arn: str, action_names: List[str], resource_arns: List[str]) -> str:
    """
    Simulates IAM policies to check if a principal (user or role) has specific permissions for given actions on specified resources.
    Useful for diagnosing '403 Access Denied', 'sts:AssumeRole', and other permission errors.
    Args:
        principal_arn: The ARN of the IAM user, group, or role whose policies you want to simulate.
        action_names: A list of action strings to simulate, e.g., ["s3:GetObject", "sts:AssumeRole"].
        resource_arns: A list of resource ARNs to simulate the actions on. Use ["*"] to represent all resources.
    Returns:
        A JSON string containing the simulation results, indicating whether each action is allowed or denied.
    """
    iam = get_aws_client('iam')
    try:
        response = iam.simulate_principal_policy(
            PolicySourceArn=principal_arn,
            ActionNames=action_names,
            ResourceArns=resource_arns
        )
        return json.dumps(response.get('EvaluationResults', []), default=str, indent=2)
    except ClientError as e:
        logger.error(f"Error in simulate_iam_policy: {e}")
        return json.dumps({"error": str(e)}, indent=2)