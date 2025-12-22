import json
import logging
from typing import Optional, Literal

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
def aws__manage_vpc(action: Literal["create", "delete", "list"], vpc_id: Optional[str] = None, cidr_block: Optional[str] = None, name_tag: Optional[str] = None, confirm: bool = False) -> str:
    """
    Creates, deletes, or lists AWS VPCs.
    Args:
        action: 'create', 'delete', or 'list'.
        vpc_id: Required for 'delete'.
        cidr_block: Required for 'create' (e.g., 10.0.0.0/16).
        name_tag: Optional Name tag for 'create'.
        confirm: Set to True to confirm deletion.
    """
    ec2 = get_aws_client('ec2')
    if action == "create":
        if not cidr_block:
            return json.dumps({"error": "cidr_block is required for create"}, indent=2)
        try:
            response = ec2.create_vpc(CidrBlock=cidr_block)
            vpc = response['Vpc']
            vpc_id = vpc['VpcId']
            if name_tag:
                ec2.create_tags(Resources=[vpc_id], Tags=[{'Key': 'Name', 'Value': name_tag}])
                vpc['Tags'] = [{'Key': 'Name', 'Value': name_tag}]
            return json.dumps(vpc, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error in aws__manage_vpc (create): {e}")
            return json.dumps({"error": str(e)}, indent=2)
    elif action == "delete":
        if not vpc_id:
            return json.dumps({"error": "vpc_id is required for delete"}, indent=2)
        if not confirm:
            return f"Action requires confirmation. Please call this tool again with confirm=True to delete VPC {vpc_id}."
        try:
            ec2.delete_vpc(VpcId=vpc_id)
            return json.dumps({"status": "deleted", "vpc_id": vpc_id}, indent=2)
        except Exception as e:
            logger.error(f"Error in aws__manage_vpc (delete): {e}")
            return json.dumps({"error": str(e)}, indent=2)
    elif action == "list":
        try:
            response = ec2.describe_vpcs()
            vpcs = []
            for vpc in response.get('Vpcs', []):
                name = next((tag['Value'] for tag in vpc.get('Tags', []) if tag['Key'] == 'Name'), None)
                vpcs.append({
                    'VpcId': vpc.get('VpcId'),
                    'CidrBlock': vpc.get('CidrBlock'),
                    'State': vpc.get('State'),
                    'Name': name,
                    'IsDefault': vpc.get('IsDefault')
                })
            return json.dumps(vpcs, indent=2)
        except Exception as e:
            logger.error(f"Error in aws__manage_vpc (list): {e}")
            return json.dumps({"error": str(e)}, indent=2)
    return json.dumps({"error": f"Invalid action: {action}"}, indent=2)


@tool
def aws__manage_subnet(action: Literal["create", "delete"], subnet_id: Optional[str] = None, vpc_id: Optional[str] = None, cidr_block: Optional[str] = None, availability_zone: Optional[str] = None, name_tag: Optional[str] = None, confirm: bool = False) -> str:
    """
    Creates or deletes a subnet.
    Args:
        action: 'create' or 'delete'.
        subnet_id: Required for 'delete'.
        vpc_id: Required for 'create'.
        cidr_block: Required for 'create'.
        availability_zone: Optional for 'create'.
        name_tag: Optional for 'create'.
        confirm: Set to True to confirm deletion.
    """
    ec2 = get_aws_client('ec2')
    if action == "create":
        if not vpc_id or not cidr_block:
            return json.dumps({"error": "vpc_id and cidr_block are required for create"}, indent=2)
        try:
            kwargs = {'VpcId': vpc_id, 'CidrBlock': cidr_block}
            if availability_zone:
                kwargs['AvailabilityZone'] = availability_zone
            response = ec2.create_subnet(**kwargs)
            subnet = response['Subnet']
            subnet_id = subnet['SubnetId']
            if name_tag:
                ec2.create_tags(Resources=[subnet_id], Tags=[{'Key': 'Name', 'Value': name_tag}])
                subnet['Tags'] = [{'Key': 'Name', 'Value': name_tag}]
            return json.dumps(subnet, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error in aws__manage_subnet (create): {e}")
            return json.dumps({"error": str(e)}, indent=2)
    elif action == "delete":
        if not subnet_id:
            return json.dumps({"error": "subnet_id is required for delete"}, indent=2)
        if not confirm:
            return f"Action requires confirmation. Please call this tool again with confirm=True to delete subnet {subnet_id}."
        try:
            ec2.delete_subnet(SubnetId=subnet_id)
            return json.dumps({"status": "deleted", "subnet_id": subnet_id}, indent=2)
        except Exception as e:
            logger.error(f"Error in aws__manage_subnet (delete): {e}")
            return json.dumps({"error": str(e)}, indent=2)
    return json.dumps({"error": f"Invalid action: {action}"}, indent=2)


@tool
def aws__manage_internet_gateway(action: Literal["create", "delete", "attach", "detach"], internet_gateway_id: Optional[str] = None, vpc_id: Optional[str] = None, name_tag: Optional[str] = None, confirm: bool = False) -> str:
    """
    Manages Internet Gateways (create, delete, attach, detach).
    Args:
        action: 'create', 'delete', 'attach', or 'detach'.
        internet_gateway_id: Required for delete, attach, detach.
        vpc_id: Required for attach, detach.
        name_tag: Optional for create.
        confirm: Required for delete, detach.
    """
    ec2 = get_aws_client('ec2')
    try:
        if action == "create":
            response = ec2.create_internet_gateway()
            igw = response['InternetGateway']
            igw_id = igw['InternetGatewayId']
            if name_tag:
                ec2.create_tags(Resources=[igw_id], Tags=[{'Key': 'Name', 'Value': name_tag}])
                igw['Tags'] = [{'Key': 'Name', 'Value': name_tag}]
            return json.dumps(igw, indent=2, default=str)
        
        if not internet_gateway_id:
            return json.dumps({"error": "internet_gateway_id is required"}, indent=2)

        if action == "delete":
            if not confirm:
                return f"Action requires confirmation. Please call this tool again with confirm=True to delete IGW {internet_gateway_id}."
            ec2.delete_internet_gateway(InternetGatewayId=internet_gateway_id)
            return json.dumps({"status": "deleted", "internet_gateway_id": internet_gateway_id}, indent=2)
        
        if action == "attach":
            if not vpc_id:
                return json.dumps({"error": "vpc_id is required for attach"}, indent=2)
            ec2.attach_internet_gateway(InternetGatewayId=internet_gateway_id, VpcId=vpc_id)
            return json.dumps({"status": "attached", "internet_gateway_id": internet_gateway_id, "vpc_id": vpc_id}, indent=2)
        
        if action == "detach":
            if not vpc_id:
                return json.dumps({"error": "vpc_id is required for detach"}, indent=2)
            if not confirm:
                return f"Action requires confirmation. Please call this tool again with confirm=True to detach IGW {internet_gateway_id} from VPC {vpc_id}."
            ec2.detach_internet_gateway(InternetGatewayId=internet_gateway_id, VpcId=vpc_id)
            return json.dumps({"status": "detached", "internet_gateway_id": internet_gateway_id, "vpc_id": vpc_id}, indent=2)

    except Exception as e:
        logger.error(f"Error in aws__manage_internet_gateway ({action}): {e}")
        return json.dumps({"error": str(e)}, indent=2)
    return json.dumps({"error": f"Invalid action: {action}"}, indent=2)


@tool
def aws__manage_route(action: Literal["create", "delete"], route_table_id: str, destination_cidr_block: str, gateway_id: Optional[str] = None, nat_gateway_id: Optional[str] = None, confirm: bool = False) -> str:
    """
    Creates or deletes a route.
    Args:
        action: 'create' or 'delete'.
        route_table_id: The ID of the route table.
        destination_cidr_block: The IPv4 CIDR address block used for the destination match.
        gateway_id: (Create only) The ID of an internet gateway or virtual private gateway.
        nat_gateway_id: (Create only) The ID of a NAT gateway.
        confirm: (Delete only) Set to True to confirm deletion.
    """
    ec2 = get_aws_client('ec2')
    if action == "create":
        try:
            kwargs = {
                'RouteTableId': route_table_id,
                'DestinationCidrBlock': destination_cidr_block
            }
            if gateway_id:
                kwargs['GatewayId'] = gateway_id
            elif nat_gateway_id:
                kwargs['NatGatewayId'] = nat_gateway_id
            else:
                return json.dumps({"error": "Must provide either gateway_id or nat_gateway_id for create"}, indent=2)

            ec2.create_route(**kwargs)
            return json.dumps({"status": "created", "route_table_id": route_table_id, "destination": destination_cidr_block}, indent=2)
        except Exception as e:
            logger.error(f"Error in aws__manage_route (create): {e}")
            return json.dumps({"error": str(e)}, indent=2)
    elif action == "delete":
        if not confirm:
            return f"Action requires confirmation. Please call this tool again with confirm=True to delete route to {destination_cidr_block} in {route_table_id}."
        try:
            ec2.delete_route(RouteTableId=route_table_id, DestinationCidrBlock=destination_cidr_block)
            return json.dumps({"status": "deleted", "route_table_id": route_table_id, "destination": destination_cidr_block}, indent=2)
        except Exception as e:
            logger.error(f"Error in aws__manage_route (delete): {e}")
            return json.dumps({"error": str(e)}, indent=2)
    return json.dumps({"error": f"Invalid action: {action}"}, indent=2)


@tool
def aws__manage_route_table(action: Literal["create", "delete", "associate", "disassociate"], route_table_id: Optional[str] = None, vpc_id: Optional[str] = None, subnet_id: Optional[str] = None, association_id: Optional[str] = None, name_tag: Optional[str] = None, confirm: bool = False) -> str:
    """
    Manages route tables (create, delete, associate, disassociate).
    Args:
        action: 'create', 'delete', 'associate', or 'disassociate'.
        route_table_id: Required for delete, associate.
        vpc_id: Required for create.
        subnet_id: Required for associate.
        association_id: Required for disassociate.
        name_tag: Optional for create.
        confirm: Required for delete, disassociate.
    """
    ec2 = get_aws_client('ec2')
    try:
        if action == "create":
            if not vpc_id:
                return json.dumps({"error": "vpc_id is required for create"}, indent=2)
            response = ec2.create_route_table(VpcId=vpc_id)
            rt = response['RouteTable']
            rt_id = rt['RouteTableId']
            if name_tag:
                ec2.create_tags(Resources=[rt_id], Tags=[{'Key': 'Name', 'Value': name_tag}])
                rt['Tags'] = [{'Key': 'Name', 'Value': name_tag}]
            return json.dumps(rt, indent=2, default=str)
        
        if action == "delete":
            if not route_table_id:
                return json.dumps({"error": "route_table_id is required for delete"}, indent=2)
            if not confirm:
                return f"Action requires confirmation. Please call this tool again with confirm=True to delete route table {route_table_id}."
            ec2.delete_route_table(RouteTableId=route_table_id)
            return json.dumps({"status": "deleted", "route_table_id": route_table_id}, indent=2)
        
        if action == "associate":
            if not route_table_id or not subnet_id:
                return json.dumps({"error": "route_table_id and subnet_id are required for associate"}, indent=2)
            response = ec2.associate_route_table(RouteTableId=route_table_id, SubnetId=subnet_id)
            return json.dumps({"status": "associated", "association_id": response['AssociationId']}, indent=2)
        
        if action == "disassociate":
            if not association_id:
                return json.dumps({"error": "association_id is required for disassociate"}, indent=2)
            if not confirm:
                return f"Action requires confirmation. Please call this tool again with confirm=True to remove association {association_id}."
            ec2.disassociate_route_table(AssociationId=association_id)
            return json.dumps({"status": "disassociated", "association_id": association_id}, indent=2)
            
    except Exception as e:
        logger.error(f"Error in aws__manage_route_table ({action}): {e}")
        return json.dumps({"error": str(e)}, indent=2)
    return json.dumps({"error": f"Invalid action: {action}"}, indent=2)