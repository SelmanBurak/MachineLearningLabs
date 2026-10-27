# Data Validation & Integrity Audit Report
**Student:** Selman Burak Uygurtürk (ID: 72922)  
**Standard:** Scikit-Learn Pipeline Integration Requirements  

## 1. Schema Range Bounds Validation
| Feature | Defined Bounds | Empirical Min | Empirical Max | Violations | Status |
|---|---|---|---|---|---|
| `age` | [17, 90] | 17.0 | 90.0 | 0 | VERIFIED |
| `hours-per-week` | [1, 99] | 1.0 | 99.0 | 0 | VERIFIED |
| `capital-gain` | [0, 99999] | 0.0 | 99999.0 | 0 | VERIFIED |
| `capital-loss` | [0, 4356] | 0.0 | 4356.0 | 0 | VERIFIED |

## 2. Missing Value Manifest ('?' Token Handling)
| Attribute | '?' Placeholders | NaN Counts | Total Missing | % Missing | Imputation Strategy |
|---|---|---|---|---|---|
| `workclass` | 0 | 2799 | 2799 | 5.73% | Scikit-Learn SimpleImputer (most_frequent) |
| `occupation` | 0 | 2809 | 2809 | 5.75% | Scikit-Learn SimpleImputer (most_frequent) |
| `native-country` | 0 | 857 | 857 | 1.75% | Scikit-Learn SimpleImputer (most_frequent) |

## 3. Duplicate Conflict Audit (Bayes Error)
- **Total Instances with Duplicate Features:** 111 rows
- **Unique Contradictory Label Profiles:** 5 distinct feature vectors
- **Conclusion:** All 48,842 rows preserved without silent deletion to guarantee unbiased stratified partitioning.
