# latent-rl

**Plataforma experimental para agentes financieros basados en representaciones latentes y Aprendizaje por Refuerzo.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Licencia](https://img.shields.io/badge/licencia-MIT-green)

`latent-rl` es una plataforma modular y reproducible para comparar, bajo las mismas
condiciones, agentes financieros de Aprendizaje por Refuerzo (RL) que operan sobre
observaciones directas del mercado frente a agentes que operan sobre **representaciones
latentes** del estado. Sobre datos reales descargados de Yahoo Finance, integra la carga
de datos, los entornos de simulación, los agentes de decisión, los mecanismos de
representación latente, las métricas de evaluación y un dashboard interactivo de análisis.

## Contexto académico

Este repositorio contiene el software desarrollado para el **Trabajo Fin de Máster (TFM)**
titulado *«Plataforma experimental para agentes financieros basados en representaciones
latentes y Aprendizaje por Refuerzo»*, del **Máster Universitario en Inteligencia
Artificial** de la **Universidad Internacional de La Rioja (UNIR)**.

- **Autores:** Carlos Cañas García-Moreno, Lucas Fuentes Galán y Manuel Díaz Bermúdez
- **Director:** Javier Cubo Villalba
- **Escuela:** Escuela Superior de Ingeniería y Tecnología (UNIR)

El objetivo del trabajo es proporcionar un marco común, trazable y extensible para evaluar
de forma homogénea ambos enfoques en un dominio caracterizado por el ruido, la
observabilidad parcial y los cambios de régimen. La validación se realiza mediante un
experimento piloto orientado a comprobar el funcionamiento de la plataforma, no a demostrar
de forma concluyente la superioridad de las representaciones latentes.

## Descripción

Librería Python que compara agentes RL con y sin representaciones latentes
sobre datos reales de mercado descargados de Yahoo Finance.

| Módulo | Descripción |
|---|---|
| `envs/` | `FinancialEnv` — entorno Gymnasium de trading (acciones: hold / buy / sell) |
| `agents/` | `RandomAgent`, `BuyAndHoldAgent`, `DQNAgent`, `LatentDQNAgent` |
| `representations/` | `MLPLatentEncoder` — encoder MLP del espacio latente |
| `pretraining/` | `AutoencoderTrainer` — preentrenamiento del encoder por reconstrucción |
| `evaluation/` | `FinancialMetrics` (Sharpe, MDD…) y `LatentAdvantageIndex` (IVL) |
| `experiments/` | `ExperimentConfig`, `run_experiment` — pipeline completo configurable |

La métrica principal es el **IVL (Índice de Ventaja Latente)**:

```
IVL = w1·ΔSharpe - w2·ΔMDD - w3·ΔSeedStd - w4·ΔIS_OOS_Gap
```

Un IVL positivo indica que el agente latente supera al agente directo.

---

## Instalación desde cero

### 1. Clonar el repositorio

```powershell
git clone https://github.com/madibe/latent-rl.git
cd latent-rl
```

### 2. Crear y activar entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Instalar el paquete

```powershell
pip install -e .
```

### 4. Verificar instalación

```powershell
python -m pytest
```

Deberías ver todos los tests en verde. Si falta `pytest`:

```powershell
pip install -e ".[dev]"
```

---

## Ejecutar el experimento

> **Los datos ya vienen incluidos.** El repositorio trae la campaña de referencia
> `results/abcd_robust_h64/` (SPY, TSLA y BTC-USD) y el encoder preentrenado
> `models/encoders/tcn_heavy.pt`. Si solo quieres explorar los resultados, puedes
> saltar directamente al **[Paso 4 — Dashboard](#paso-4--dashboard)** sin ejecutar
> nada más. Los pasos 1-3 son para regenerar los resultados desde cero.

### Paso 1 — Análisis auxiliar del espacio latente (opcional)

```powershell
python examples\latent_index_example.py
```

Genera `results/latent_index_data.csv` con proyecciones del encoder para visualización.

### Paso 2 — Experimento principal (IS/OOS multi-semilla)

```powershell
python examples\compare_agents_multiseed_experiment.py
```

Descarga datos reales de Yahoo Finance, entrena los 5 agentes sobre datos
In-Sample y los evalúa en IS y OOS. Genera:

```
results/
└── SPY/
    ├── agent_summary.csv       # métricas IS/OOS agregadas por agente
    ├── agent_seed_metrics.csv  # métricas por (agente, semilla, split)
    └── ivl_results.csv         # IVL por par (directo, latente)
```

### Paso 3 — Recalcular el IVL (opcional)

```powershell
python -m scripts.utilities.compute_ivl
```

Lee los CSVs existentes y recalcula el IVL. Para una campaña concreta, pasa
`--results-dir results/<campaña>` y los nombres de sus brazos mediante
`--direct-agent` y `--latent-agents`.

### Paso 4 — Dashboard

```powershell
python dashboard\app.py --results-dir results/abcd_robust_h64
```

Abre **http://127.0.0.1:8050** en el navegador. Como la campaña
`results/abcd_robust_h64/` viene incluida en el repositorio, el dashboard funciona
nada más clonar, sin necesidad de ejecutar los pasos anteriores. Detecta
automáticamente los tickers disponibles (SPY, TSLA, BTC-USD) y muestra la sección
cross-ticker.

### Paso 5 — Figuras reproducibles desde resultados existentes (opcional)

```powershell
python scripts\export_memory_figures.py `
  --results-dir results/abcd_robust_h64 `
  --out-dir artifacts/memory_figures/ABCD_robust_h64
```

Este comando no entrena agentes ni modifica los CSV. Exporta las figuras
agregadas y su `figure_manifest.json`; usa `--include-optional` para incluir
el scatter riesgo-retorno y la estabilidad entre semillas.

> **Nota Windows:** si ves `UnicodeEncodeError`, añade `$env:PYTHONUTF8=1` antes de cada comando.

---

## Personalizar el experimento

Edita `examples/compare_agents_multiseed_experiment.py`. Todos los parámetros
se configuran en `ExperimentConfig`; el resto del código no necesita tocarse:

```python
from latent_rl.experiments import ExperimentConfig, run_experiment

config = ExperimentConfig(
    # ── Datos ────────────────────────────────────────────────
    tickers=["SPY"],            # un ticker o varios: ["SPY", "AAPL", "BTC-USD"]
    start_date="2020-01-01",
    end_date="2023-12-31",
    train_ratio=0.7,            # 70% In-Sample, 30% Out-of-Sample

    # ── Experimento ──────────────────────────────────────────
    seeds=[0, 1, 2, 3, 4],
    n_training_episodes=10,
    n_eval_episodes=3,

    # ── Entorno ──────────────────────────────────────────────
    lookback_window=10,
    initial_balance=10_000.0,
    transaction_cost=0.001,

    # ── LatentDQN ────────────────────────────────────────────
    latent_dim=16,
    encoder_hidden_dims=[64, 32],

    # ── IVL ──────────────────────────────────────────────────
    ivl_weights={"sharpe": 0.25, "mdd": 0.25, "seed_std": 0.25, "is_oos_gap": 0.25},

    results_dir="results",
)

if __name__ == "__main__":
    run_experiment(config)
```

### Experimento multi-ticker

Con más de un ticker el pipeline ejecuta un experimento independiente por activo
y genera además un resumen cross-ticker:

```python
config = ExperimentConfig(
    tickers=["SPY", "AAPL", "BTC-USD"],
    ...
)
```

Estructura de resultados:

```
results/
├── SPY/
│   ├── agent_summary.csv
│   ├── agent_seed_metrics.csv
│   └── ivl_results.csv
├── AAPL/
│   └── ...
├── BTC-USD/
│   └── ...
└── ticker_comparison.csv     # IVL comparado entre activos
```

El dashboard muestra un selector de ticker y una sección cross-ticker
automáticamente cuando hay más de un ticker disponible.

---

## Estructura del proyecto

```
latent-rl/
├── src/latent_rl/
│   ├── agents/          # RandomAgent, BuyAndHoldAgent, DQNAgent, LatentDQNAgent
│   ├── envs/            # FinancialEnv (Gymnasium)
│   ├── representations/ # MLPLatentEncoder
│   ├── pretraining/     # AutoencoderTrainer
│   ├── evaluation/      # FinancialMetrics, LatentAdvantageIndex (IVL)
│   ├── experiments/     # ExperimentConfig, run_experiment, pipeline completo
│   ├── reporting/       # Loader agregado y figuras reproducibles
│   ├── analysis/        # LatentIndexAnalyzer (PCA, norm, first-component)
│   └── data/            # CSVDataLoader, DataPreprocessor, YahooFinanceLoader
├── scripts/
│   ├── experiments/     # Campañas reproducibles del TFM
│   └── utilities/       # Generación y análisis de resultados
├── examples/            # Ejemplos didácticos de la API
├── dashboard/           # Dashboard Dash + Plotly (app.py)
├── tests/               # Pruebas unitarias y de integración
├── results/             # Salidas agrupadas por campaña — excluido de Git
└── pyproject.toml
```

---

## Problemas frecuentes

### Dashboard vacío al arrancar

Genera los datos primero (pasos 1-3) o usa el botón **"Generar datos"** en el dashboard.

### UnicodeEncodeError en Windows

```powershell
$env:PYTHONUTF8=1
python examples\compare_agents_multiseed_experiment.py
```

### Los resultados no aparecen en el dashboard

Comprueba que exista `results/SPY/agent_summary.csv` (o la carpeta del ticker que hayas usado).
El dashboard busca en `results/{ticker}/`, no en `results/` directamente.

### Reinstalar tras actualizar el código

```powershell
git pull
pip install -e .
python -m pytest
```

---

## Notas metodológicas

- El split IS/OOS es **temporal**: IS precede siempre a OOS; los datos nunca se mezclan.
- El preentrenamiento del encoder usa **solo datos IS** para evitar data leakage.
- Las semillas controlan únicamente la inicialización de los agentes; los datos son idénticos entre semillas.
- El IVL es una métrica agregada; se recomienda analizar cada componente (ΔSharpe, ΔMDD, ΔSeedStd, ΔIS_OOS_Gap) por separado.
- Los resultados del dashboard son exploratorios y no implican superioridad estadística formal.

## Cómo citar

Si utilizas esta plataforma en un trabajo académico, puedes citarla así:

```bibtex
@mastersthesis{canas_fuentes_diaz_2026_latentrl,
  title        = {Plataforma experimental para agentes financieros basados en
                  representaciones latentes y Aprendizaje por Refuerzo},
  author       = {Cañas García-Moreno, Carlos and Fuentes Galán, Lucas and
                  Díaz Bermúdez, Manuel},
  school       = {Universidad Internacional de La Rioja (UNIR)},
  type         = {Trabajo Fin de Máster},
  year         = {2026},
  address      = {Valencia, España},
  note         = {Máster Universitario en Inteligencia Artificial. Director: Javier Cubo Villalba}
}
```

## Licencia

Distribuido bajo licencia **MIT**. Consulta el archivo [`LICENSE`](LICENSE) para más detalles.
