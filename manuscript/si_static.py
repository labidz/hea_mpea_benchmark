PHYS_DEFS = [
    ('Yang delta (*δ*)', 'Atomic-size mismatch, *δ* = [Σ*c*_{i}(1 − *r*_{i}/*r̄*)^{2}]^{1/2} [@yang2012]'),
    ('Yang omega (*Ω*)', '*T*_{m}Δ*S*_{mix}/|Δ*H*_{mix}| [@yang2012]'),
    ('Mixing enthalpy (Δ*H*_{mix})', 'Σ_{i<j} 4Δ*H*_{ij}*c*_{i}*c*_{j} with Miedema binary enthalpies [@takeuchi2005]'),
    ('Configuration entropy', '−*R*Σ*c*_{i} ln *c*_{i} (sign convention of matminer)'),
    ('Lambda entropy (*Λ*)', 'Δ*S*_{mix}/*δ*^{2}'),
    ('VEC mean', 'Fraction-weighted valence-electron concentration [@guo2011]'),
    ('APE mean', 'Atomic packing efficiency mean'),
    ('Radii local mismatch / gamma', 'Local atomic-radius mismatch and the solid-angle parameter *γ*'),
    ('Electronegativity delta / local mismatch', 'Pauling electronegativity mismatch measures'),
    ('Shear modulus mean / delta / local mismatch', 'Fraction-weighted elemental shear modulus and its mismatch measures'),
    ('Shear modulus strength model', 'Solid-solution strengthening descriptor based on shear-modulus and size mismatch [@wen2019]'),
    ('Mean cohesive energy', 'Fraction-weighted elemental cohesive energy'),
    ('Interant electrons (total, s, p, d, f)', 'Fraction-weighted valence electrons interacting with neighbours'),
    ('Atomic weight mean', 'Fraction-weighted atomic weight'),
]

HP_SPACES = [
    ('kNN', 'k ∈ {1, 3, 5, 10, 15}; distance weighting', 'all 5'),
    ('Ridge', 'α ∈ 10^{−3}–10^{3} (13 log-spaced values; standardized target)', 'all 13'),
    ('SVR', 'C ∈ {1, 10, 100}; γ ∈ {0.002, 0.01, 0.05}; ε ∈ {0.05, 0.2} (standardized target)', '10 of 18'),
    ('GPR', 'Constant × Matérn-5/2 + white noise; ML-II with one restart', '–'),
    ('RF', '300 trees; max_features ∈ {0.2, 0.4, 0.7}; min_samples_leaf ∈ {1, 3}', '6 of 6'),
    ('XGBoost', '500 trees, subsample 0.8, λ = 1; depth ∈ {3, 5, 7}; η ∈ {0.03, 0.08}; colsample ∈ {0.3, 0.7}; min_child_weight ∈ {1, 4}', '10 of 24'),
    ('MLP', 'Hidden layers ∈ {(128), (256, 128)}; L2 α ∈ {10^{−3}, 10^{−1}, 1}; Adam, early stopping (15% validation, patience 30)', '6 of 6'),
    ('TabPFN v2', 'Open v2 regression checkpoint, default inference settings, no tuning (Table S2b)', '–'),
]
