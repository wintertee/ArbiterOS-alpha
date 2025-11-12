"""Core components for ArbiterOS-alpha.

This module contains the main classes and functionality for policy-driven
governance of LangGraph execution.
"""

import datetime
import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langgraph.types import Command

from .instructions import InstructionType
from .policy import PolicyChecker, PolicyRouter

logger = logging.getLogger(__name__)


@dataclass
class History:
    """The minimal OS metadata for tracking instruction execution.

    Attributes:
        timestamp: When the instruction was executed.
        instruction: The instruction type that was executed.
        input_state: The state passed to the instruction.
        output_state: The state returned by the instruction.
        check_policy_results: Results of policy checkers (name -> passed/failed).
        route_policy_results: Results of policy routers (name -> target or None).
    """

    timestamp: datetime.datetime
    instruction: InstructionType
    input_state: dict[str, Any]
    output_state: dict[str, Any] = field(default_factory=dict)
    check_policy_results: dict[str, bool] = field(default_factory=dict)
    route_policy_results: dict[str, str | None] = field(default_factory=dict)


class ArbiterOSAlpha:
    """Main ArbiterOS coordinator for policy-driven LangGraph execution.

    ArbiterOSAlpha provides a lightweight governance layer on top of LangGraph,
    enabling policy-based validation and dynamic routing without modifying
    the underlying graph structure.

    Attributes:
        history: List of execution history entries with timestamps and I/O.
        policy_checkers: List of PolicyChecker instances for validation.
        policy_routers: List of PolicyRouter instances for dynamic routing.

    Example:
        >>> os = ArbiterOSAlpha(backend="langgraph")
        >>> os.add_policy_checker(HistoryPolicyChecker("require_verification",["generate", "execute"]))
        >>> os.add_policy_router(ConfidencePolicyRouter("confidence", 0.5, "retry"))
        >>> @os.instruction("generate")
        ... def generate(state): return {"result": "output"}
    """

    def __init__(self, backend: Literal["langgraph", "vanilla"] = "langgraph"):
        """Initialize the ArbiterOSAlpha instance.

        Args:
            backend: The execution backend to use.
                - "langgraph": Use an agent based on the LangGraph framework.
                - "vanilla": Use the framework-less ('from scratch') agent implementation.
        """
        self.backend = backend
        self.history: list[History] = []
        self.policy_checkers: list[PolicyChecker] = []
        self.policy_routers: list[PolicyRouter] = []

    def add_policy_checker(self, checker: PolicyChecker) -> None:
        """Register a policy checker for validation.

        Args:
            checker: A PolicyChecker instance to validate execution constraints.
        """
        logger.debug(f"Adding policy checker: {checker}")
        self.policy_checkers.append(checker)

    def add_policy_router(self, router: PolicyRouter) -> None:
        """Register a policy router for dynamic flow control.

        Policy routers are only supported when using the "langgraph" backend.

        Args:
            router: A PolicyRouter instance to dynamically route execution.

        Raises:
            RuntimeError: If the backend is not "langgraph".
        """
        if self.backend != "langgraph":
            raise RuntimeError(
                "Policy routers are only supported with the 'langgraph' backend."
            )
        logger.debug(f"Adding policy router: {router}")
        self.policy_routers.append(router)

    def _check_before(self) -> tuple[dict[str, bool], bool]:
        """Execute all policy checkers before instruction execution.

        Returns:
            A dictionary mapping checker names to their validation results.
            A final boolean indicating if all checkers passed.
        """
        results = {}
        logger.debug(f"Running {len(self.policy_checkers)} policy checkers (before)")
        for checker in self.policy_checkers:
            result = checker.check_before(self.history)

            if result is False:
                results[checker.name] = result
                logger.error(f"Policy checker {checker} failed validation.")

        return results, all(results.values())

    def _route_after(self) -> tuple[dict[str, str | None], str | None]:
        """Determine if execution should be routed to a different node.

        Consults all registered policy routers in order. Returns the first
        non-None routing decision.

        Returns:
            A dictionary mapping checker names to their route destination.
            A final str indicating the final route destination.
        """
        results = {}
        destination = None
        used_router = None
        logger.debug(f"Checking {len(self.policy_routers)} policy routers")
        for router in self.policy_routers:
            decision = router.route_after(self.history)

            if decision:
                results[router.name] = decision
                used_router = router
                destination = decision

        decision_count = sum(1 for v in results.values() if v is not None)
        if decision_count > 1:
            logger.error(
                "Multiple routers decided to route. Fallback to first decision."
            )

        if destination is not None:
            logger.warning(f"Router {used_router} decision made to: {destination}")
        return results, destination

    def instruction(
        self, instruction_type: InstructionType
    ) -> Callable[[Callable], Callable]:
        """Decorator to wrap LangGraph node functions with policy governance.

        This decorator adds policy validation, execution history tracking,
        and dynamic routing to LangGraph node functions. It's the core
        integration point between ArbiterOS and LangGraph.

        Args:
            instruction_type: An instruction type from one of the Core enums
                (CognitiveCore, MemoryCore, ExecutionCore, NormativeCore,
                MetacognitiveCore, AdaptiveCore, SocialCore, or AffectiveCore).

        Returns:
            A decorator function that wraps the target node function.

        Example:
            >>> from arbiteros_alpha.instructions import CognitiveCore
            >>> @os.instruction(CognitiveCore.GENERATE)
            ... def generate(state: State) -> State:
            ...     return {"field": "value"}
            >>> # Function now includes policy checks and history tracking
        """
        # Validate that instruction_type is a valid InstructionType enum
        if not isinstance(instruction_type, InstructionType.__args__):
            raise TypeError(
                f"instruction_type must be an instance of one of the Core enums, got {type(instruction_type)}"
            )

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                logger.debug(
                    f"Executing instruction: {instruction_type.__class__.__name__}.{instruction_type.name}"
                )

                self.history.append(
                    History(
                        timestamp=datetime.datetime.now(),
                        instruction=instruction_type,
                        input_state=args[0],
                    )
                )

                self.history[-1].check_policy_results, all_passed = self._check_before()

                result = func(*args, **kwargs)
                logger.debug(f"Instruction {instruction_type.name} returned: {result}")
                self.history[-1].output_state = result

                self.history[-1].route_policy_results, destination = self._route_after()

                if destination:
                    return Command(update=result, goto=destination)

                return result

            return wrapper

        return decorator
