import unittest
from pydantic import ValidationError
from core.state import SupervisorAgentPlanOutput, Requirement, SupervisorAgentOutputBase

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

    def test_supervisor_output_base_validation_short_instruction(self):
        # Instructions too short for non-end worker
        with self.assertRaises(ValidationError):
            SupervisorAgentOutputBase(
                current_step_id=1,
                current_step_status="pending",
                next_worker="system",
                response_to_user="Response",
                instructions_for_worker="Do" # Too short (< 5 chars)
            )

    def test_supervisor_output_base_validation_end_worker(self):
        # Short instructions allowed if next_worker is 'end'
        model = SupervisorAgentOutputBase(
            current_step_id=1,
            current_step_status="completed",
            next_worker="end",
            response_to_user="Done",
            instructions_for_worker="" 
        )
        self.assertEqual(model.next_worker, "end")