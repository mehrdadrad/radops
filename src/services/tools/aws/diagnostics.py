import logging
import time
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

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
def analyze_reachability(source_id: str, dest_id: str, port: int) -> str:
    """
    Triggers AWS VPC Reachability Analyzer between two points.
    Returns: JSON describing the blocking component (SG, NACL, Route Table).
    """
    ec2 = get_aws_client('ec2')
    try:
        # Create Path
        path = ec2.create_network_insights_path(
            Source=source_id,
            Destination=dest_id,
            Protocol='tcp',
            DestinationPort=port
        )
        path_id = path['NetworkInsightsPath']['NetworkInsightsPathId']

        # Start Analysis
        analysis = ec2.start_network_insights_analysis(
            NetworkInsightsPathId=path_id
        )
        analysis_id = analysis['NetworkInsightsAnalysis']['NetworkInsightsAnalysisId']

        # Wait for completion
        logger.info(f"Started Reachability Analysis {analysis_id}. Waiting for completion...")
        while True:
            response = ec2.describe_network_insights_analyses(
                NetworkInsightsAnalysisIds=[analysis_id]
            )
            status = response['NetworkInsightsAnalyses'][0]['Status']
            if status in ['succeeded', 'failed']:
                break
            time.sleep(2)

        result = response['NetworkInsightsAnalyses'][0]

        # Cleanup Analysis and Path to avoid clutter.
        # The analysis must be deleted before the path.
        try:
            ec2.delete_network_insights_analysis(
                NetworkInsightsAnalysisId=analysis_id)
            ec2.delete_network_insights_path(NetworkInsightsPathId=path_id)
            logger.info(
                f"Successfully deleted insights analysis {analysis_id} and path {path_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up network insights resources: {e}")

        return json.dumps({
            "status": status,
            "network_path_found": result.get('NetworkPathFound', False),
            "explanations": result.get('Explanations', [])
        }, indent=2)
    except Exception as e:
        logger.error(f"Error in analyze_reachability: {e}")
        return json.dumps({"error": str(e)}, indent=2)


@tool
def query_logs(log_group: str, query: str) -> str:
    """
    Runs a CloudWatch Logs Insights query.
    Example Query: "fields @message | filter @message like /Error/"
    """
    logs = get_aws_client('logs')
    try:
        # Default to last 1 hour
        start_time = int((datetime.now() - timedelta(hours=1)).timestamp())
        end_time = int(datetime.now().timestamp())

        response = logs.start_query(
            logGroupName=log_group,
            startTime=start_time,
            endTime=end_time,
            queryString=query,
        )
        query_id = response['queryId']

        while True:
            results = logs.get_query_results(queryId=query_id)
            if results['status'] in ['Complete', 'Failed', 'Cancelled']:
                break
            time.sleep(1)

        parsed_results = []
        for row in results.get('results', []):
            parsed_row = {field['field']: field['value'] for field in row}
            parsed_results.append(parsed_row)

        return json.dumps(parsed_results, indent=2)
    except Exception as e:
        logger.error(f"Error in query_logs: {e}")
        return json.dumps([{"error": str(e)}], indent=2)


@tool
def check_recent_changes(resource_id: str) -> str:
    """
    Queries CloudTrail/Config for changes to this resource in the last 24h.
    """
    ct = get_aws_client('cloudtrail')
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        response = ct.lookup_events(
            LookupAttributes=[
                {'AttributeKey': 'ResourceName', 'AttributeValue': resource_id}
            ],
            StartTime=start_time,
            EndTime=end_time,
            MaxResults=50
        )

        events = []
        for event in response.get('Events', []):
            events.append({
                'EventName': event.get('EventName'),
                'EventTime': str(event.get('EventTime')),
                'Username': event.get('Username'),
                'EventSource': event.get('EventSource')
            })
        return json.dumps(events, indent=2)
    except Exception as e:
        logger.error(f"Error in check_recent_changes: {e}")
        return json.dumps([{"error": str(e)}], indent=2)


@tool
def get_ec2_health(instance_id: str) -> str:
    """
    Checks Status Checks (System/Instance) and CPU utilization metrics.
    """
    ec2 = get_aws_client('ec2')
    cw = get_aws_client('cloudwatch')
    try:
        # Status Checks
        status_response = ec2.describe_instance_status(
            InstanceIds=[instance_id]
        )
        instance_status = "Unknown"
        system_status = "Unknown"
        if status_response['InstanceStatuses']:
            status = status_response['InstanceStatuses'][0]
            instance_status = status['InstanceStatus']['Status']
            system_status = status['SystemStatus']['Status']

        # CPU Metrics (Last 1 hour, 5 min period)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        metrics = cw.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Average']
        )

        datapoints = sorted(metrics['Datapoints'], key=lambda x: x['Timestamp'])
        cpu_stats = [{"time": str(dp['Timestamp']), "average": dp['Average']} for dp in datapoints]

        return json.dumps({
            "instance_status": instance_status,
            "system_status": system_status,
            "cpu_utilization": cpu_stats
        }, indent=2)
    except Exception as e:
        logger.error(f"Error in get_ec2_health: {e}")
        return json.dumps({"error": str(e)}, indent=2)