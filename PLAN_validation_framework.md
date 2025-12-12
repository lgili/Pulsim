# Plano: Framework de Validação SPICE para Pulsim

**Decisões do Usuário:**
- **SPICE Runner**: Ambos (PySpice quando disponível, fallback para subprocess)
- **Prioridade**: Circuitos básicos (RC/RL/RLC com solução analítica primeiro)

## Objetivo
Criar uma suíte de testes automatizada que valide os resultados do Pulsim contra:
1. **Soluções analíticas** (onde existem)
2. **NgSpice** (via PySpice) como referência gold-standard
3. **Tolerâncias definidas** para cada tipo de circuito

## Estrutura Proposta

```
python/tests/validation/
├── __init__.py
├── framework/
│   ├── __init__.py
│   ├── base.py              # Classes base para testes de validação
│   ├── spice_runner.py      # Interface com PySpice/NgSpice
│   ├── comparator.py        # Comparação de resultados com métricas
│   ├── analytical.py        # Soluções analíticas conhecidas
│   └── reporters.py         # Geração de relatórios
├── level1_components/       # Nível 1: Componentes básicos isolados
│   ├── test_resistor.py
│   ├── test_capacitor.py
│   ├── test_inductor.py
│   ├── test_voltage_source.py
│   └── test_current_source.py
├── level2_linear/           # Nível 2: Circuitos lineares simples
│   ├── test_rc_circuits.py
│   ├── test_rl_circuits.py
│   ├── test_rlc_circuits.py
│   └── test_voltage_divider.py
├── level3_nonlinear/        # Nível 3: Componentes não-lineares
│   ├── test_diode.py
│   ├── test_switch.py
│   ├── test_mosfet.py
│   └── test_igbt.py
├── level4_converters/       # Nível 4: Conversores de potência
│   ├── test_buck_converter.py
│   ├── test_boost_converter.py
│   ├── test_half_bridge.py
│   └── test_full_bridge.py
└── level5_complex/          # Nível 5: Circuitos complexos
    ├── test_transformer.py
    ├── test_coupled_inductors.py
    └── test_multi_stage.py
```

## Framework Base (framework/base.py)

```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
from enum import Enum
import numpy as np

class ValidationLevel(Enum):
    COMPONENT = 1      # Componente isolado
    LINEAR = 2         # Circuitos lineares
    NONLINEAR = 3      # Componentes não-lineares
    CONVERTER = 4      # Conversores de potência
    COMPLEX = 5        # Circuitos complexos

@dataclass
class ValidationResult:
    """Resultado de um teste de validação"""
    test_name: str
    passed: bool
    pulsim_result: np.ndarray
    reference_result: np.ndarray  # Analítico ou NgSpice
    max_error: float
    rms_error: float
    max_error_threshold: float
    rms_error_threshold: float
    execution_time_pulsim: float
    execution_time_reference: float
    notes: str = ""

@dataclass
class CircuitDefinition:
    """Definição de um circuito de teste"""
    name: str
    description: str
    level: ValidationLevel
    # Definição do circuito para Pulsim
    pulsim_circuit: Callable  # Função que retorna Circuit
    # Netlist SPICE equivalente
    spice_netlist: str
    # Parâmetros de simulação
    tstart: float = 0.0
    tstop: float = 1e-3
    dt: float = 1e-6
    # Nós para comparação
    compare_nodes: List[str] = None
    # Tolerâncias
    max_error_tolerance: float = 1e-3  # 0.1%
    rms_error_tolerance: float = 1e-4  # 0.01%
    # Solução analítica (se disponível)
    analytical_solution: Optional[Callable] = None

class ValidationTest:
    """Classe base para testes de validação"""

    def __init__(self, circuit_def: CircuitDefinition):
        self.circuit_def = circuit_def

    def run_pulsim(self) -> tuple:
        """Executa simulação no Pulsim"""
        import pulsim as sl
        circuit = self.circuit_def.pulsim_circuit()
        opts = sl.SimulationOptions()
        opts.tstart = self.circuit_def.tstart
        opts.tstop = self.circuit_def.tstop
        opts.dt = self.circuit_def.dt
        # ... configurações

    def run_ngspice(self) -> tuple:
        """Executa simulação no NgSpice via PySpice"""
        # Usa PySpice para rodar NgSpice

    def run_analytical(self, time: np.ndarray) -> np.ndarray:
        """Calcula solução analítica"""
        if self.circuit_def.analytical_solution:
            return self.circuit_def.analytical_solution(time)
        return None

    def compare(self, result1, result2) -> ValidationResult:
        """Compara dois resultados"""
        # Interpolação para mesmos pontos de tempo
        # Cálculo de erro max e RMS

    def validate(self) -> ValidationResult:
        """Executa validação completa"""
        # 1. Roda Pulsim
        # 2. Roda NgSpice (ou analítico)
        # 3. Compara resultados
        # 4. Retorna ValidationResult
```

## Casos de Teste por Nível

### Nível 1: Componentes Básicos

| Teste | Descrição | Validação |
|-------|-----------|-----------|
| R_ohms_law | V=IR para vários valores | Analítico |
| R_series | Resistores em série | Analítico |
| R_parallel | Resistores em paralelo | Analítico |
| C_charging | Capacitor carregando via R | Analítico: V(t)=V0(1-e^(-t/RC)) |
| C_initial_condition | Capacitor com IC | Analítico |
| L_step_response | Indutor com step | Analítico: I(t)=V/R(1-e^(-Rt/L)) |
| L_initial_condition | Indutor com IC | Analítico |
| V_dc | Fonte DC | Direto |
| V_pulse | Fonte Pulse | Verificação de bordas |
| V_sine | Fonte Senoidal | Analítico |
| I_dc | Fonte de corrente DC | Lei de Ohm |

### Nível 2: Circuitos Lineares

| Teste | Descrição | Validação |
|-------|-----------|-----------|
| RC_lowpass | Filtro passa-baixa RC | Analítico + NgSpice |
| RC_highpass | Filtro passa-alta RC | Analítico + NgSpice |
| RL_lowpass | Filtro passa-baixa RL | Analítico + NgSpice |
| RLC_series | RLC série (sub/sobre/crítico) | Analítico + NgSpice |
| RLC_parallel | RLC paralelo | Analítico + NgSpice |
| Voltage_divider | Divisor de tensão | Analítico |
| RC_ladder_5 | Ladder 5 estágios | NgSpice |
| RC_ladder_10 | Ladder 10 estágios | NgSpice |

### Nível 3: Componentes Não-Lineares

| Teste | Descrição | Validação |
|-------|-----------|-----------|
| Diode_forward | Diodo em polarização direta | NgSpice |
| Diode_reverse | Diodo em reverso | NgSpice |
| Diode_rectifier | Retificador meia-onda | NgSpice |
| Diode_fullwave | Retificador onda completa | NgSpice |
| Switch_basic | Chave básica ON/OFF | NgSpice |
| Switch_pwm | Chave com PWM | NgSpice |
| MOSFET_dc | MOSFET ponto de operação DC | NgSpice |
| MOSFET_switching | MOSFET chaveando | NgSpice |
| MOSFET_with_Cgs | MOSFET com capacitâncias | NgSpice |

### Nível 4: Conversores de Potência

| Teste | Descrição | Validação |
|-------|-----------|-----------|
| Buck_ideal | Buck com switch ideal | NgSpice + Analítico Vout=D*Vin |
| Buck_real | Buck com Ron/diode | NgSpice |
| Buck_efficiency | Verificar eficiência | Cálculo de perdas |
| Boost_ideal | Boost ideal | NgSpice |
| Boost_real | Boost real | NgSpice |
| HalfBridge_RL | Meia-ponte com carga RL | NgSpice |
| FullBridge_RL | Ponte completa | NgSpice |

### Nível 5: Circuitos Complexos

| Teste | Descrição | Validação |
|-------|-----------|-----------|
| Transformer_ideal | Transformador ideal | Analítico: V2=V1*N2/N1 |
| Transformer_leakage | Com indutância de dispersão | NgSpice |
| CoupledInductors | Indutores acoplados | NgSpice |
| Flyback | Conversor flyback | NgSpice |
| LLC_resonant | Conversor LLC | NgSpice |

## Métricas de Validação

Para cada teste, calcular:

1. **Erro Máximo Absoluto**: max|Pulsim - Referência|
2. **Erro RMS**: sqrt(mean((Pulsim - Referência)²))
3. **Erro Máximo Relativo**: max|Pulsim - Referência| / max|Referência|
4. **Correlação**: Coeficiente de correlação de Pearson
5. **Tempo de Execução**: Comparação Pulsim vs NgSpice

## Critérios de Aprovação

| Nível | Max Error | RMS Error | Notas |
|-------|-----------|-----------|-------|
| 1 | < 0.01% | < 0.001% | Componentes são simples |
| 2 | < 0.1% | < 0.01% | Circuitos lineares |
| 3 | < 1% | < 0.1% | Não-linearidades |
| 4 | < 5% | < 1% | Transientes complexos |
| 5 | < 10% | < 2% | Circuitos muito complexos |

## Implementação em Fases (PRIORIDADE ATUALIZADA)

### Fase 1: Framework Base + Circuitos Lineares Básicos
**Prioridade: ALTA - Começar aqui**

1. **Estrutura do Framework**
   - [ ] Criar diretórios `python/tests/validation/`
   - [ ] Implementar `framework/base.py` (ValidationTest, ValidationResult)
   - [ ] Implementar `framework/spice_runner.py` (PySpice + subprocess fallback)
   - [ ] Implementar `framework/comparator.py` (métricas de erro)
   - [ ] Implementar `framework/analytical.py` (soluções RC/RL/RLC)

2. **Testes RC (com solução analítica)**
   - [ ] RC step response: V(t) = V0(1 - e^(-t/RC))
   - [ ] RC discharge: V(t) = V0 * e^(-t/RC)
   - [ ] RC lowpass filter frequency response
   - [ ] RC highpass filter

3. **Testes RL (com solução analítica)**
   - [ ] RL step response: I(t) = (V0/R)(1 - e^(-Rt/L))
   - [ ] RL current decay
   - [ ] Inductor voltage spike at switch-off

4. **Testes RLC (com solução analítica)**
   - [ ] RLC underdamped (oscilação)
   - [ ] RLC critically damped
   - [ ] RLC overdamped
   - [ ] RLC resonance

### Fase 2: Componentes Básicos + Fontes
- [ ] Resistor: Lei de Ohm, série, paralelo
- [ ] Capacitor: IC, charging curves
- [ ] Inductor: IC, current continuity
- [ ] Fontes: DC, Pulse (bordas), Sine, PWL

### Fase 3: Componentes Não-Lineares
- [ ] Diodo ideal vs Shockley model
- [ ] Switch ON/OFF transitions
- [ ] MOSFET DC operating point
- [ ] MOSFET switching with gate capacitance

### Fase 4: Conversores de Potência
- [ ] Buck converter (ideal)
- [ ] Buck converter (com perdas)
- [ ] Boost converter
- [ ] Half-bridge, Full-bridge

### Fase 5: Circuitos Complexos
- [ ] Transformer
- [ ] Flyback converter
- [ ] Multi-stage circuits

## Execução dos Testes

```bash
# Rodar todos os testes de validação
pytest python/tests/validation/ -v --tb=short

# Rodar apenas um nível
pytest python/tests/validation/level1_components/ -v

# Rodar com relatório detalhado
pytest python/tests/validation/ --validation-report=report.html

# Rodar comparação completa com NgSpice
pytest python/tests/validation/ --with-ngspice -v
```

## Relatório de Validação

Gerar relatório HTML/Markdown com:
- Lista de todos os testes executados
- Status PASS/FAIL
- Métricas de erro
- Gráficos comparativos (Pulsim vs NgSpice vs Analítico)
- Tempo de execução
- Resumo por nível

## Dependências

```
# requirements-validation.txt
pyspice>=1.5
numpy>=1.20
scipy>=1.7
matplotlib>=3.5
pytest>=7.0
pytest-html>=3.1
```

## Notas de Implementação

1. **PySpice requer NgSpice instalado**: `brew install ngspice` ou `apt install ngspice`
2. **Interpolação**: Para comparar resultados com timesteps diferentes, usar interpolação linear
3. **Warm-up**: Ignorar primeiros pontos onde há transitórios de inicialização
4. **Steady-state**: Para conversores, comparar apenas região de steady-state
5. **Tolerâncias ajustáveis**: Permitir override de tolerâncias por teste específico
