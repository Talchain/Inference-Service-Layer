"""Independent exact reference; does not import the candidate classifier or evaluator."""
from __future__ import annotations

from fractions import Fraction as Q


def truth(case: str, x: float, y: float, *, c: float = 0.0, objective: bool = True) -> dict:
    u, v = Q(str(x)), Q(str(y))
    if case == "F9" and u + v > Q(1):
        return {"state": "OUTSIDE_DECLARED_DOMAIN"}
    if case == "F1" or case == "F2" or case == "F3" or case == "F4" or case == "F9":
        a_value, b_value = u, v
    elif case == "F5":
        a_value, b_value = u * v, Q(0)
    elif case == "F6":
        a_value, b_value = u * u - Q(1, 4), Q(0)
    elif case == "F8":
        a_value, b_value = Q(2), Q(0)
    elif case == "F11":
        a_value, b_value = Q(1, 40000) - (u - Q(1, 80)) * (u - Q(1, 80)) - (v - Q(1, 80)) * (v - Q(1, 80)), Q(0)
    elif case == "F13":
        a_value, b_value = u + Q(str(c)), v
    else:
        raise ValueError(case)
    a_feasible = v <= Q(2, 5) if case in ("F3", "F4") else True
    b_feasible = None if case == "F4" else (u <= Q(2, 5) if case == "F3" else True)
    feasible = {"A": "FEASIBLE" if a_feasible else "INFEASIBLE", "B": "UNSUPPORTED" if b_feasible is None else ("FEASIBLE" if b_feasible else "INFEASIBLE")}
    goal = {"A": "ATTAINED" if u >= Q(3, 5) else "MISSED", "B": "ATTAINED" if v >= Q(3, 5) else "MISSED"} if case == "F3" else {"A": "NOT_APPLICABLE", "B": "NOT_APPLICABLE"}
    if not objective:
        preference = "OBJECTIVE_UNSPECIFIED"
    elif b_feasible is None:
        preference = "INCOMPLETE_COMPARISON"
    elif not a_feasible and not b_feasible:
        preference = "NO_FEASIBLE_OPTION"
    elif a_feasible and not b_feasible:
        preference = "SOLE_FEASIBLE:A"
    elif b_feasible and not a_feasible:
        preference = "SOLE_FEASIBLE:B"
    else:
        difference = a_value - b_value
        if difference == 0:
            preference = "EXACT_TIE"
        elif case == "F2" and abs(difference) <= Q(1, 10):
            preference = "PRACTICALLY_EQUIVALENT"
        elif difference > 0:
            preference = "PREFERRED:A"
        else:
            preference = "PREFERRED:B"
    return {"state": "EVALUATED", "feasibility": feasible, "goal": goal, "named_preference": preference}
