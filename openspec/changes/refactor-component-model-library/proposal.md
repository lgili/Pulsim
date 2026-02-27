## Why
A biblioteca de componentes ainda está concentrada em um header monolítico (`device_base.hpp`), o que aumenta acoplamento, dificulta evolução de modelos e amplia risco de regressão em mudanças de convergência/performance.
Também precisamos de um caminho sistemático para melhorar convergência de circuitos com comutação dura sem exigir tuning manual do usuário final.

## What Changes
- Refatorar a biblioteca de componentes para layout modular: um modelo por arquivo, mantendo compatibilidade com o include legado agregador.
- Introduzir revisão e evolução de modelos por família (passivos, fontes, semicondutores, chaves), com foco em robustez numérica sem perder rastreabilidade física.
- Definir envelope de parasíticos numéricos pequenos e controlados para ajudar convergência em casos patológicos de comutação.
- Definir contrato YAML para configuração de regularização/parasitismos (manual e automático) com defaults seguros.
- Amarrar rollout a gates de não-regressão (accuracy + runtime + completion ratio) no benchmark/local-limit/KPI.

## Impact
- Affected specs: `device-models`, `netlist-yaml`
- Affected code:
  - `core/include/pulsim/v1/device_base.hpp`
  - `core/include/pulsim/v1/components/*`
  - `core/include/pulsim/v1/runtime_circuit.hpp`
  - `core/src/v1/simulation*.cpp`
  - `core/src/v1/transient_services.cpp`
  - parser YAML e testes de bindings/benchmarks
