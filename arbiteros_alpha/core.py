"""Core components for ArbiterOS-alpha.

This module contains the main classes and functionality for policy-driven
governance of LangGraph execution.
"""

import datetime
import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml
from langgraph.types import Command
from rich.console import Console

from .policy import PolicyChecker, PolicyRouter

logger = logging.getLogger(__name__)


@dataclass
class History:
    """The minimal OS metadata for tracking instruction execution."""

    timestamp: datetime.datetime
    instruction: str
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
        >>> os = ArbiterOSAlpha()
        >>> os.add_policy_checker(HistoryPolicyChecker().add_blacklist("rule", ["a", "b"]))
        >>> os.add_policy_router(ConfidencePolicyRouter("confidence", 0.5, "retry"))
        >>> @os.instruction("generate")
        ... def generate(state): return {"result": "output"}
    """

    def __init__(self):
        """Initialize ArbiterOSAlpha with empty history and no policies."""
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

        Args:
            router: A PolicyRouter instance to dynamically route execution.
        """
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

    def instruction(self, name: str) -> Callable[[Callable], Callable]:
        """Decorator to wrap LangGraph node functions with policy governance.

        This decorator adds policy validation, execution history tracking,
        and dynamic routing to LangGraph node functions. It's the core
        integration point between ArbiterOS and LangGraph.

        Args:
            name: A unique identifier for this instruction/node.

        Returns:
            A decorator function that wraps the target node function.

        Example:
            >>> @os.instruction("generate")
            ... def generate(state: State) -> State:
            ...     return {"field": "value"}
            >>> # Function now includes policy checks and history tracking
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                logger.debug(f"Executing instruction: {name}")

                self.history.append(
                    History(
                        timestamp=datetime.datetime.now(),
                        instruction=name,
                        input_state=args[0],
                    )
                )

                self.history[-1].check_policy_results, all_passed = self._check_before()

                result = func(*args, **kwargs)
                logger.debug(f"Instruction {name} returned: {result}")
                self.history[-1].output_state = result

                self.history[-1].route_policy_results, destination = self._route_after()

                if destination:
                    return Command(update=result, goto=destination)

                return result

            return wrapper

        return decorator

    def print_history(self) -> None:
        """Print the execution history in a readable format."""
        console = Console()
        console.print("\n[bold cyan]📋 Arbiter OS Execution History[/bold cyan]")
        console.print("=" * 80)

        for i, entry in enumerate(self.history, 1):
            # Format policy results
            check_results = entry.check_policy_results
            route_results = entry.route_policy_results

            # Header with instruction name
            console.print(f"\n[bold cyan][{i}] {entry.instruction}[/bold cyan]")
            console.print(f"[dim]  Timestamp: {entry.timestamp}[/dim]")

            # Format input state as YAML
            console.print("  [yellow]Input:[/yellow]")
            input_yaml = yaml.dump(
                entry.input_state, default_flow_style=False, sort_keys=False
            )
            for line in input_yaml.strip().split("\n"):
                console.print(f"    [dim]{line}[/dim]")

            # Format output state as YAML
            console.print("  [yellow]Output:[/yellow]")
            output_yaml = yaml.dump(
                entry.output_state, default_flow_style=False, sort_keys=False
            )
            for line in output_yaml.strip().split("\n"):
                console.print(f"    [dim]{line}[/dim]")

            # Show detailed policy check results
            console.print("  [yellow]Policy Checks:[/yellow]")
            if check_results:
                for policy_name, result in check_results.items():
                    status = "[green]✓[/green]" if result else "[red]✗[/red]"
                    console.print(f"    {status} {policy_name}")
            else:
                console.print("    [dim](none)[/dim]")

            # Show detailed policy route results
            console.print("  [yellow]Policy Routes:[/yellow]")
            if route_results:
                for policy_name, destination in route_results.items():
                    if destination:
                        console.print(
                            f"    [magenta]→[/magenta] {policy_name} [bold magenta]⇒ {destination}[/bold magenta]"
                        )
                    else:
                        console.print(f"    [dim]— {policy_name}[/dim]")
            else:
                console.print("    [dim](none)[/dim]")

        console.print("\n" + "=" * 80 + "\n")
