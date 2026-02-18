import unittest
from pydantic import ValidationError
from core.state import SupervisorAgentPlanOutput, Requirement

class TestStateExtended(unittest.TestCase):
    def test_supervisor_plan_output_validation(self):
        # Valid plan
        plan = SupervisorAgentPlanOutput(
            current_step_id=1,
            current_step_status="pending",
            next_worker="system",
            response_to_user="Plan created",
            instructions_for_worker="Execute step 1",
            detected_requirements=[
                Requirement(id=1, instruction="Do X", assigned_agent="system")
            ]
        )
        self.assertEqual(len(plan.detected_requirements), 1)

    def test_supervisor_plan_output_missing_requirements(self):
        # Missing detected_requirements
        with self.assertRaises(ValidationError):
            SupervisorAgentPlanOutput(
                current_step_id=1,
                current_step_status="pending",
                next_worker="system",
                response_to_user="Plan created",
                instructions_for_worker="Execute step 1"
            )

    def test_requirement_validation(self):
        # Invalid agent
        with self.assertRaises(ValidationError):
            Requirement(id=1, instruction="Do X", assigned_agent="invalid_agent")