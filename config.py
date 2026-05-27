
# ===== GLOBAL LABELS =====

ATTR_LABELS = {
  "n":              r"$n$",
  "c":              r"$c$",
  "p_in":           r"$p_{in}$",
  "p_out":          r"$p_{out}$",
  "means":          r"$\mu_{s,c}$",
  "std":            r"$std$",
  "m":              r"$m$",
  "p_in_min":       r"$p_{in}^{min}$",
  "p_in_max":       r"$p_{in}^{max}$",
  "p_out_min":      r"$p_{out}^{min}$",
  "p_out_max":      r"$p_{out}^{max}$",
  "means_min":      r"$\mu_{s,c}^{min}$",
  "means_max":      r"$\mu_{s,c}^{max}$",
  "means_max_min":  r"$\Delta\mu_{s,c}$"
}

PARAMS_LABELS = {
  "mu": r"$\hat{\mu}$",
  "k":  r"$\hat{k}$",
  "dC": r"$\delta\hat{c}$",
  "dT": r"$\delta\hat{t}$"
}

MET_LABELS_G = {
  "P_G": r"$\mathcal{P}_{z^*}$",
  "D_G": r"$\mathcal{D}_{z^*}$",
  "I_G": r"$\mathcal{I}_{z^*}$"
}

MET_LABELS = {
  "P":    r"$\mathcal{P}_{z^*}$",
  "D":    r"$\mathcal{D}_{z^*}$",
  "I":    r"$\mathcal{I}_{z^*}$",
  "D_st": r"$\mathcal{D}_{z^*\mathrm{same-topic}}$",
  "D_ct": r"$\mathcal{D}_{z^*\mathrm{cross-topic}}$",
  "dD":   r"$\Delta\mathcal{D}_{z^*}$(\%)",
  "dP":   r"$\Delta\mathcal{P}_{z^*}$(\%)"
}

COND_LABELS = {
  "NR": "Non-Regularized",
  "R":  "Regularized"
}

# ===== GLOBAL COLORS =====

VIOLETS  = ["violet", "darkviolet"]
COLORS_4 = ["#471CA8", "#D1105A", "#F27F04", "#9BB460"]
COLORS_2 = ["#471CA8", "#F27F04"]

# ===== GLOBAL CONSTANTS =====

# og 2 L_ARGS = {"linewidth": 1, "linestyle": ":", "color": "0.5", "alpha": 0.5, "zorder": 0} #***

#og L_ARGS = {"color": "gray", "linestyle": ":", "linewidth": 0.7, "zorder": 1}

L_ARGS = {
  "color": "gray", 
  "linestyle": ":", 
  "linewidth": 0.7, 
  "zorder": 0
}

#maybe??
#CONDS    = ["NR", "RR"]
#COLORS   = {"NR": "#471CA8",
 #          "R": "#F27F04"}