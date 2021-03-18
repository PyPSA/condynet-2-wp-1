import numpy as np

configfile: "config.yaml"

 # TODO: include pypsa-eur as subworkflow
rule preprocess_network:
    input: config["network"]
    output: "networks/prenetwork.nc"
    script: "scripts/preprocess_network.py"

rules compute_robust_c
  input: "networks/prenetwork.nc"
  output: "results/new/c_r.csv"
  script: scripts/find_c_r.py
  
rules compute_approximate_c
  input: "networks/prenetwork.nc"
  output: "results/new/c_a.csv"
  script: scripts/find_c_a.py
  
rules compute_linespecific_c
  input: "networks/prenetwork.nc"
  output: "results/new/c_l.csv"
  script: scripts/find_c_l.py


rule solve_full_contingency:
    input: "networks/prenetwork.nc"
    log:
        memory="logs/memory_full_contingency.log",
        stats="logs/stats_full_contingency.csv"
    output: "results/new/postnetwork_full.nc"
    threads: 4
    resources: mem=50000
    script: "scripts/full_contingency.py"

rule solve_heuristic_contingency:
    input: "networks/prenetwork.nc"
    log:
        memory="logs/memory_heur{s_max_pu}.log",
        stats="logs/stats_heur{s_max_pu}.csv"
    output: "results/new/postnetwork_heur{s_max_pu}.nc"
    threads: 4
    resources: mem=50000
    script: "scripts/heuristic_contingency.py"

rule check_outage_flow:
    input: "results/new/postnetwork_heur{s_max_pu}.nc"
    output: "results/new/outage_line_loading_heur{s_max_pu}.csv"
    script: "scripts/check_outage_flow.py"

def s_max_pus():
    cf = config["s_max_pu"]
    return np.round(np.arange(cf["start"], cf["stop"] + cf["step"], cf["step"]),2)

rule all:
    input:
      heur=expand("results/new/outage_line_loading_heur{s_max_pu}.csv", s_max_pu=s_max_pus()),
      full="results/new/postnetwork_full.nc",
      robust="results/new/c_r.csv",
      approximate="results/new/c_a.csv",
      line_specific="results/new/c_l.csv"
