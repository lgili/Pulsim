# Plano: Framework de Validação para PulsimCore v2

**Status:** ✅ IMPLEMENTADO - 71 testes passando

| Nível | Categoria | Testes |
|-------|-----------|--------|
| 1 | Circuitos Lineares (RC, RL, RLC) | 19 |
| 2 | Análise DC | 11 |
| 3 | Não-Lineares (Diodo, Switch, MOSFET) | 21 |
| 4 | Conversores (Buck, Boost) | 20 |
| **Total** | | **71** |

## Objetivo
Criar uma suíte de testes automatizada que valide os resultados do PulsimCore contra:
1. **Soluções analíticas** (onde existem)
2. **NgSpice** (via PySpice) como referência gold-standard
3. **Tolerâncias definidas** para cada tipo de circuito

## Nova API Python

```python
import pulsim as ps

# Criar circuito
ckt = ps.Circuit()
gnd = ps.Circuit.ground()
n1 = ckt.add_node("n1")
n2 = ckt.add_node("n2")

# Adicionar componentes
ckt.add_voltage_source("V1", n1, gnd, 5.0)
ckt.add_resistor("R1", n1, n2, 1000.0)
ckt.add_capacitor("C1", n2, gnd, 1e-6)

# DC Operating Point
dc_result = ps.dc_operating_point(ckt)
print(dc_result.newton_result.solution)

# Transient
times, states, success, msg = ps.run_transient(ckt, 0.0, 5e-3, 10e-6)
```

## Estrutura Proposta

```
python/tests/validation/
├── __init__.py
├── conftest.py              # Fixtures pytest
├── framework/
│   ├── __init__.py
│   ├── base.py              # Classes base para testes
│   ├── spice_runner.py      # Interface com PySpice/NgSpice
│   ├── comparator.py        # Comparação de resultados
│   └── reporters.py         # Geração de relatórios
├── level1_linear/           # Nível 1: Circuitos lineares ✅
│   ├── test_rc_circuits.py  # 6 testes
│   ├── test_rl_circuits.py  # 5 testes
│   └── test_rlc_circuits.py # 8 testes
├── level2_dc_analysis/      # Nível 2: Análise DC ✅
│   └── test_resistor_networks.py  # 11 testes
├── level3_nonlinear/        # Nível 3: Componentes não-lineares ✅
│   ├── test_diode.py        # 6 testes
│   ├── test_switch.py       # 7 testes
│   └── test_mosfet.py       # 8 testes
├── level4_converters/       # Nível 4: Conversores de potência ✅
│   ├── test_buck_converter.py   # 9 testes
│   └── test_boost_converter.py  # 11 testes
└── reports/                 # Relatórios gerados
```

## Framework Base (framework/base.py)

```python
from dataclasses import dataclass
from typing import Optional, List, Callable
from enum import Enum
import numpy as np
import pulsim as ps

class ValidationLevel(Enum):
    LINEAR = 1         # Circuitos lineares com solução analítica
    DC_ANALYSIS = 2    # Análise DC
    NONLINEAR = 3      # Componentes não-lineares
    CONVERTER = 4      # Conversores de potência

@dataclass
class ValidationResult:
    """Resultado de um teste de validação"""
    test_name: str
    passed: bool
    pulsim_times: np.ndarray
    pulsim_values: np.ndarray
    reference_times: np.ndarray
    reference_values: np.ndarray
    max_error: float
    rms_error: float
    max_relative_error: float
    tolerance: float
    execution_time_ms: float
    notes: str = ""

@dataclass
class CircuitDefinition:
    """Definição de um circuito de teste"""
    name: str
    description: str
    level: ValidationLevel
    build_circuit: Callable[[], ps.Circuit]  # Função que retorna Circuit
    spice_netlist: Optional[str] = None      # Netlist SPICE equivalente
    analytical_solution: Optional[Callable] = None  # Solução analítica
    t_start: float = 0.0
    t_stop: float = 1e-3
    dt: float = 1e-6
    compare_nodes: List[int] = None  # Índices dos nós para comparar
    tolerance: float = 0.01          # 1% default

class ValidationTest:
    """Classe base para testes de validação"""

    def __init__(self, circuit_def: CircuitDefinition):
        self.circuit_def = circuit_def

    def run_pulsim_dc(self) -> ps.DCAnalysisResult:
        """Executa análise DC no Pulsim"""
        circuit = self.circuit_def.build_circuit()
        return ps.dc_operating_point(circuit)

    def run_pulsim_transient(self) -> tuple:
        """Executa simulação transiente no Pulsim"""
        circuit = self.circuit_def.build_circuit()

        # DC inicial
        dc_result = ps.dc_operating_point(circuit)
        if not dc_result.success:
            raise RuntimeError(f"DC failed: {dc_result.message}")

        # Transiente
        times, states, success, msg = ps.run_transient(
            circuit,
            self.circuit_def.t_start,
            self.circuit_def.t_stop,
            self.circuit_def.dt,
            dc_result.newton_result.solution
        )

        if not success:
            raise RuntimeError(f"Transient failed: {msg}")

        return np.array(times), np.array([s for s in states])

    def run_analytical(self, times: np.ndarray) -> np.ndarray:
        """Calcula solução analítica"""
        if self.circuit_def.analytical_solution:
            return self.circuit_def.analytical_solution(times)
        return None

    def compare(self, pulsim_times, pulsim_values, ref_times, ref_values) -> ValidationResult:
        """Compara resultados Pulsim vs referência"""
        from scipy import interpolate

        # Interpolar referência para mesmos tempos do Pulsim
        if len(ref_times) != len(pulsim_times):
            f = interpolate.interp1d(ref_times, ref_values, kind='linear',
                                     fill_value='extrapolate')
            ref_interpolated = f(pulsim_times)
        else:
            ref_interpolated = ref_values

        # Calcular erros
        errors = np.abs(pulsim_values - ref_interpolated)
        max_error = np.max(errors)
        rms_error = np.sqrt(np.mean(errors**2))
        max_ref = np.max(np.abs(ref_interpolated))
        max_rel_error = max_error / max_ref if max_ref > 0 else max_error

        passed = max_rel_error <= self.circuit_def.tolerance

        return ValidationResult(
            test_name=self.circuit_def.name,
            passed=passed,
            pulsim_times=pulsim_times,
            pulsim_values=pulsim_values,
            reference_times=ref_times,
            reference_values=ref_interpolated,
            max_error=max_error,
            rms_error=rms_error,
            max_relative_error=max_rel_error,
            tolerance=self.circuit_def.tolerance,
            execution_time_ms=0.0,
            notes=""
        )

    def validate_with_analytical(self) -> ValidationResult:
        """Valida contra solução analítica"""
        import time

        start = time.perf_counter()
        times, states = self.run_pulsim_transient()
        exec_time = (time.perf_counter() - start) * 1000

        # Extrair nó de interesse
        node_idx = self.circuit_def.compare_nodes[0] if self.circuit_def.compare_nodes else 1
        pulsim_values = np.array([s[node_idx] for s in states])

        # Solução analítica
        ref_values = self.run_analytical(times)

        result = self.compare(times, pulsim_values, times, ref_values)
        result.execution_time_ms = exec_time
        return result
```

## Casos de Teste Prioritários (Fase 1)

### Nível 1: Circuitos Lineares com Solução Analítica

| Teste | Circuito | Solução Analítica | Tolerância |
|-------|----------|-------------------|------------|
| RC_step | V -> R -> C -> GND | V(t) = Vf*(1 - e^(-t/τ)), τ=RC | 1% |
| RC_discharge | C(V0) -> R -> GND | V(t) = V0*e^(-t/τ) | 1% |
| RL_step | V -> R -> L -> GND | I(t) = (V/R)*(1 - e^(-t/τ)), τ=L/R | 1% |
| RLC_under | V -> R -> L -> C -> GND (ζ<1) | Oscilação amortecida | 2% |
| RLC_critical | V -> R -> L -> C -> GND (ζ=1) | Amortecimento crítico | 2% |
| RLC_over | V -> R -> L -> C -> GND (ζ>1) | Superamortecido | 2% |

### Nível 2: Análise DC

| Teste | Circuito | Validação | Tolerância |
|-------|----------|-----------|------------|
| Resistor_divider | V -> R1 -> R2 -> GND | V2 = V*R2/(R1+R2) | 0.01% |
| Series_resistors | V -> R1 -> R2 -> R3 -> GND | Lei de Ohm | 0.01% |
| Parallel_resistors | V -> R1||R2 -> GND | 1/Req = 1/R1 + 1/R2 | 0.01% |

### Nível 3: Não-Lineares

| Teste | Circuito | Validação | Tolerância |
|-------|----------|-----------|------------|
| Diode_forward | V -> R -> D -> GND | V_R ≈ V - Vf | 5% |
| Switch_on | V -> R -> SW(closed) -> GND | I = V/R | 1% |
| Switch_off | V -> R -> SW(open) -> GND | I ≈ 0 | 1% |

## Critérios de Aprovação

| Nível | Max Rel Error | RMS Error | Descrição |
|-------|---------------|-----------|-----------|
| 1 | < 1% | < 0.1% | Circuitos lineares simples |
| 2 | < 0.01% | < 0.001% | DC (solução exata) |
| 3 | < 5% | < 1% | Não-lineares |
| 4 | < 10% | < 2% | Conversores (complexos) |

## Implementação - Checklist

### Fase 1: Framework + Circuitos Lineares ✅ COMPLETO
- [x] Criar estrutura de diretórios
- [x] Implementar `framework/base.py`
- [x] Implementar `framework/comparator.py`
- [x] Testes RC (step, discharge) - 6 testes
- [x] Testes RL (step) - 5 testes
- [x] Testes RLC (under/critical/over damped) - 8 testes
- [x] Validar contra soluções analíticas do pulsim (RCAnalytical, RLAnalytical, RLCAnalytical)

### Fase 2: DC Analysis ✅ COMPLETO
- [x] Testes divisor de tensão
- [x] Testes resistores série/paralelo
- [x] Verificar Newton solver accuracy
- Total: 11 testes

### Fase 3: Não-Lineares ✅ COMPLETO
- [x] Testes diodo - 6 testes
- [x] Testes switch - 7 testes
- [x] Testes MOSFET - 8 testes
- [x] Verificar convergence aids

### Fase 4: Conversores de Potência ✅ COMPLETO
- [x] Testes Buck converter - 9 testes
- [x] Testes Boost converter - 11 testes

### Fase 5: Integração NgSpice (Opcional - Não Implementado)
- [ ] Implementar `framework/spice_runner.py`
- [ ] Comparação Pulsim vs NgSpice
- [ ] Relatórios comparativos

## Execução

```bash
# Rodar todos os testes de validação
pytest python/tests/validation/ -v

# Rodar apenas nível 1 (lineares)
pytest python/tests/validation/level1_linear/ -v

# Com cobertura
pytest python/tests/validation/ -v --cov=pulsim

# Gerar relatório
pytest python/tests/validation/ -v --html=validation_report.html
```

## Dependências

```
# requirements-validation.txt (mínimo)
numpy>=1.20
scipy>=1.7
pytest>=7.0
pytest-html>=3.1

# Opcional para comparação NgSpice
pyspice>=1.5
```
