# Dataset Card: Adult Census Dataset

**Author:** Selman Burak Uygurtürk (ID: 72922)  
**Dataset:** Adult Census Income (OpenML ID: adult, v2)  

### 1. Origin & Licensing
Extracted by Ronny Kohavi and Barry Becker from the 1994 Current Population Survey (US Census Bureau). Distributed under Creative Commons Attribution 4.0 International (CC BY 4.0).

### 2. Dimensionality
- Total Instances: 48,842 observations.
- Attributes: 15 columns (14 predictors, 1 binary target).
- Stratification: 70% Train, 30% Test.

### 3. Ethical Implications & Demographics
- **Temporal Skew:** Reflects 1994 socioeconomic participation where high income among women was underrepresented (~10.9% vs ~30.6% for men).
- **Geographic Representation:** 89.7% of surveyed individuals originate from the United States.
- **Privacy Controls:** Census bureau eliminated PII and aggregated records via sampling weight `fnlwgt`.
