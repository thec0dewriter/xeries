# API Architecture - Mermaid Diagrams

## 1. Complete Inheritance Hierarchy



```mermaid
graph TD
    ABC[ABC<br/>Python Abstract Base]
    
    ABC -->|inherits| BasePartitioner["<b>BasePartitioner</b><br/>─────────────<br/>+ fit(X, feature)<br/>+ get_groups(X)<br/>+ fit_get_groups(X, feature)"]
    ABC -->|inherits| BaseExplainer["<b>BaseExplainer</b><br/>─────────────<br/>+ explain(X, **kwargs)<br/>"]
    ABC -->|inherits| BaseAdapter["<b>BaseAdapter</b><br/>─────────────<br/>+ get_training_data()<br/>+ predict(X)<br/>+ get_feature_names()<br/>+ get_series_column()"]
    
    BasePartitioner -->|implements| ManualPartitioner["<b>ManualPartitioner</b><br/>─────────────<br/>- mapping: dict<br/>- series_col: str<br/>- _group_encoder: dict<br/>─────────────<br/>Manual mapping strategy"]
    BasePartitioner -->|implements| TreePartitioner["<b>TreePartitioner</b><br/>─────────────<br/>- _tree: DecisionTreeRegressor<br/>- _encoder: OneHotEncoder<br/>- max_depth, min_samples_leaf<br/>- series_col: str<br/>─────────────<br/>Tree-based auto-discovery"]
    
    BaseExplainer -->|inherits| MetricBasedExplainer["<b>MetricBasedExplainer</b><br/>─────────────<br/>+ __init__(model, metric, random_state)<br/>+ _resolve_metric(metric)"]
    BaseExplainer -->|inherits| AttributionExplainer["<b>AttributionExplainer</b><br/>─────────────<br/>+ __init__(model, background_data, random_state)"]
    BaseExplainer -->|inherits| CausalExplainer["<b>CausalExplainer</b><br/>─────────────<br/>(Future extension)"]
    
    MetricBasedExplainer -->|implements| ConditionalPermutationImportance["<b>ConditionalPermutationImportance</b><br/>─────────────<br/>- strategy: str<br/>- partitioner: BasePartitioner<br/>- n_repeats: int<br/>- n_jobs: int<br/>─────────────<br/>Conditional permutation algorithm"]
    AttributionExplainer -->|implements| ConditionalSHAP["<b>ConditionalSHAP</b><br/>─────────────<br/>- series_col: str<br/>- _series_backgrounds: dict<br/>─────────────<br/>Series-aware SHAP"]
    
    BaseAdapter -->|implements| SkforecastAdapter["<b>SkforecastAdapter</b><br/>─────────────<br/>- forecaster: Any<br/>- _series, _exog: Optional<br/>- _X, _y: Cached data<br/>─────────────<br/>Skforecast integration"]
    
    PlotsModule["<b>Module: plots</b><br/>Standalone Functions<br/>─────────────<br/>• plot_importance_bar()<br/>• plot_importance_heatmap()"]
    
    TypeSystem["<b>Types (core.types)</b><br/>─────────────<br/>• BaseResult<br/>• FeatureImportanceResult<br/>• SHAPResult<br/>• ArrayLike, ModelProtocol, etc."]
    
    ConditionalPermutationImportance -->|uses| BasePartitioner
    ConditionalPermutationImportance -->|returns| TypeSystem
    ConditionalSHAP -->|returns| TypeSystem
    PlotsModule -->|accepts| TypeSystem
    
    style BasePartitioner fill:#e1f5ff
    style BaseExplainer fill:#e1f5ff
    style BaseAdapter fill:#e1f5ff
    style MetricBasedExplainer fill:#bbdefb
    style AttributionExplainer fill:#bbdefb
    style CausalExplainer fill:#bbdefb
    style ManualPartitioner fill:#fff3e0
    style TreePartitioner fill:#fff3e0
    style ConditionalPermutationImportance fill:#f3e5f5
    style SkforecastAdapter fill:#fff3e0
    style ConditionalSHAP fill:#f3e5f5
    style PlotsModule fill:#e8f5e9
    style TypeSystem fill:#fce4ec
```

---

## 2. Partitioner Inheritance & Usage

```mermaid
graph LR
    BasePartitioner["<b>BasePartitioner</b> ABC<br/>─────────────<br/>Defines partitioning contract"]
    
    BasePartitioner -->|implements| Manual["<b>ManualPartitioner</b><br/>─────────────<br/>Input: User mapping<br/>series_id → group_label<br/>─────────────<br/>Use: Domain knowledge"]
    BasePartitioner -->|implements| Tree["<b>TreePartitioner</b><br/>─────────────<br/>Auto: DecisionTree leaves<br/>→ Subgroups<br/>─────────────<br/>Use: Automatic discovery"]
    
    Manual -->|used by| CPFIAuto["ConditionalPermutation<br/>Importance<br/>strategy='manual'"]
    Tree -->|used by| CPFIManual["ConditionalPermutation<br/>Importance<br/>strategy='auto'"]
    
    style BasePartitioner fill:#bbdefb
    style Manual fill:#fff9c4
    style Tree fill:#fff9c4
    style CPFIAuto fill:#f0f4c3
    style CPFIManual fill:#f0f4c3
```

---

## 3. Explainer Hierarchy & Data Flow

```mermaid
graph LR
    subgraph Explainers["<b>Explainers / Importance Calculators</b>"]
        BaseExplainer["<b>BaseExplainer</b> ABC<br/>─────────────<br/>Common initialization<br/>Metric resolution<br/>model, metric, random_state"]
        
        BaseExplainer -->|implements| CPFI["<b>Conditional Permutation<br/>Importance</b><br/>─────────────<br/>Algorithm:<br/>1. Fit partitioner<br/>2. Shuffle within groups<br/>3. Measure metric increase"]
        
        SHAP["<b>Conditional SHAP</b><br/>─────────────<br/>Standalone class<br/>Algorithm:<br/>KernelSHAP with<br/>series-aware backgrounds"]
    end
    
    subgraph Data["<b>Data & Results</b>"]
        Input["Input: X, y, features"]
        Input1["Input: X, feature_names"]
        Result1["Output:<br/>FeatureImportance<br/>Result"]
        Result2["Output:<br/>SHAPResult"]
    end
    
    Input --> CPFI
    CPFI --> Result1
    Input1 --> SHAP
    SHAP --> Result2
    
    style BaseExplainer fill:#e1bee7
    style CPFI fill:#f3e5f5
    style SHAP fill:#f3e5f5
    style Result1 fill:#fce4ec
    style Result2 fill:#fce4ec
```

---

## 4. Adapter Pattern - Framework Integration

```mermaid
graph TB
    subgraph Frameworks["External Frameworks"]
        Skforecast["Skforecast<br/>ForecasterRecursiveMultiSeries"]
        SKLModelPool["Other Models<br/>(Future adapters)"]
    end
    
    subgraph AdapterLayer["<b>Adapter Layer</b>"]
        BaseAdapter["<b>BaseAdapter</b> ABC<br/>─────────────<br/>get_training_data()<br/>predict(X)<br/>get_feature_names()<br/>get_series_column()"]
        
        BaseAdapter -->|implements| SkfAdapter["<b>SkforecastAdapter</b><br/>─────────────<br/>Handles:<br/>• Wide & stacked formats<br/>• Series encoding detection<br/>• Training data caching<br/>• Exogenous variables"]
    end
    
    subgraph TCPFI["<b>TCPFI Library</b>"]
        Explainer["Explainers<br/>• ConditionalPermutation<br/>Importance<br/>• ConditionalSHAP"]
    end
    
    Skforecast -->|adapted via| SkfAdapter
    SkfAdapter -->|unified interface| Explainer
    SKLModelPool -.->|future adapters| BaseAdapter
    
    style BaseAdapter fill:#bbdefb
    style SkfAdapter fill:#ffccbc
    style Skforecast fill:#fff9c4
    style Explainer fill:#f3e5f5
```

---

## 5. Complete Data Flow - From Model to Visualization

```mermaid
graph LR
    subgraph Step1["Step 1: Model Training"]
        A["Forecaster Model<br/>(e.g., Skforecast)"] -->|fit| B["Trained Model"]
    end
    
    subgraph Step2["Step 2: Data Extraction"]
        B -->|SkforecastAdapter| C["Training Data<br/>X: DataFrame<br/>y: Series"]
    end
    
    subgraph Step3["Partitioning"]
        C -->|TreePartitioner<br/>Auto-discovery| D["Data Subgroups<br/>Group Labels"]
        C -->|ManualPartitioner<br/>User-defined| D
    end
    
    subgraph Step4["Importance Calculation"]
        D -->|ConditionalPermutation<br/>Importance.compute| E["FeatureImportance<br/>Result"]
        C -->|ConditionalSHAP<br/>explain| F["SHAPResult"]
    end
    
    subgraph Step5["Visualization"]
        E -->|plot_importance_bar| G["Feature Importance<br/>Bar Chart"]
        E -->|plot_importance<br/>_heatmap| H["Multi-Condition<br/>Heatmap"]
        F -->|plot_importance_bar| I["SHAP Summary<br/>Chart"]
    end
    
    style A fill:#fff9c4
    style B fill:#fff9c4
    style C fill:#bbdefb
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#fce4ec
    style G fill:#e8f5e9
    style H fill:#e8f5e9
    style I fill:#e8f5e9
```

---

## 6. Module Organization

```mermaid
graph TD
    TCPFI["<b>timelens</b> Package"]
    
    TCPFI --> Core["<b>core/</b><br/>─────────────<br/>base.py<br/>├─ BasePartitioner<br/>├─ BaseExplainer<br/>types.py<br/>├─ Types & protocols"]
    
    TCPFI --> Partitioners["<b>partitioners/</b><br/>─────────────<br/>manual.py<br/>├─ ManualPartitioner<br/>tree.py<br/>├─ TreePartitioner"]
    
    TCPFI --> Importance["<b>importance/</b><br/>─────────────<br/>permutation.py<br/>├─ ConditionalPermutation<br/>├─ Importance<br/>shap.py<br/>├─ ConditionalSHAP"]
    
    TCPFI --> Adapters["<b>adapters/</b><br/>─────────────<br/>base.py<br/>├─ BaseAdapter<br/>skforecast.py<br/>├─ SkforecastAdapter"]
    
    TCPFI --> Viz["<b>visualization/</b><br/>─────────────<br/>plots.py<br/>├─ plot_importance_bar()<br/>├─ plot_importance_heatmap()"]
    
    style TCPFI fill:#e3f2fd
    style Core fill:#bbdefb
    style Partitioners fill:#fff3e0
    style Importance fill:#f3e5f5
    style Adapters fill:#ffccbc
    style Viz fill:#e8f5e9
```

---

## 7. Composition Relationships

```mermaid
graph LR
    subgraph "Composition Pattern"
        CPFI["ConditionalPermutation<br/>Importance"]
        Part["Partitioner<br/>(Strategy)"]
        
        CPFI -->|composes| Part
        CPFI -->|uses| Model["Model"]
    end
    
    subgraph "Strategy Implementations"
        Manual["ManualPartitioner"]
        Tree["TreePartitioner"]
        Part -->|strategy| Manual
        Part -->|strategy| Tree
    end
    
    subgraph "Inheritance Pattern"
        Base["BaseExplainer"]
        CPFI -->|inherits| Base
    end
    
    style CPFI fill:#f3e5f5
    style Part fill:#fff3e0
    style Manual fill:#ffd54f
    style Tree fill:#ffd54f
    style Model fill:#fff9c4
    style Base fill:#bbdefb
```

---

## Color Legend

| Color | Meaning |
|-------|---------|
| 🔵 Light Blue | Abstract Base Classes |
| 🟨 Orange/Yellow | Concrete Implementations |
| 🟣 Purple | Explainers/Importance |
| 🟠 Orange | Adapters |
| 🟢 Green | Visualization/Utilities |
| 🩷 Pink | Result Types |

