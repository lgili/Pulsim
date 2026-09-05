"""The C++ BlockChain Jiles-Atherton adapter is REMOVED (refuses by
name).

This file used to check that the C++ adapter matched the Python
observer step for step. It did — and both were wrong the same way:
the injected EMF had its sign inverted (the magnetisation acted as a
negative inductance: current leading voltage, |I₁| > V/R on a passive
branch) and the one-step-lagged coupling was numerically unstable
above q = L_M/(dt·(R + 2L_0/dt)) ≈ 0.5, i.e. for every shipped use.
Agreement between two copies of the same mistake certified nothing.

The hysteretic inductor is solved inside the Newton loop now
(`add_hysteretic_inductor` builds a Newton flux branch); its physics
anchors live in test_hysteretic_core_in_loop.py.
"""

from __future__ import annotations

import pytest

pulsim = pytest.importorskip("pulsim")

_HAS_CXX_CHAIN = (
    hasattr(pulsim, "_pulsim")
    and hasattr(pulsim._pulsim, "CxxBlockChain")
    and hasattr(pulsim._pulsim.CxxBlockChain(), "add_hysteretic_inductor")
)


@pytest.mark.skipif(not _HAS_CXX_CHAIN, reason="C++ BlockChain not in this build")
def test_cpp_chain_block_refuses_by_name():
    chain = pulsim._pulsim.CxxBlockChain()
    with pytest.raises(Exception, match="INSIDE the Newton loop"):
        chain.add_hysteretic_inductor(
            Ms=4.0e5, a=50.0, alpha=5e-5, c=0.2, k=30.0,
            N_turns=100, l_m=0.05, A_core=1e-4,
            inductor_branch_var_idx=0, bemf_source_idx=1)
