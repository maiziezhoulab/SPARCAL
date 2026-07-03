import pickle
import os
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class GenotypeMetrics:
    baf: float
    depth: int
    ref: str
    alt: str

def load_transition_data(section_id="1", quality_filter="baseQ13mapQ20", study="P6_tumor"):
    # /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/metrics/beagle/baseQ0mapQ0/P4_TUMOR_1_shifted_results.pkl
    base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
    metrics_dir = os.path.join(base_dir, "data", study, section_id, "metrics/beagle", quality_filter)

    # Load shifted and stable results
    if study == "P4_tumor":
        shifted_file = os.path.join(metrics_dir, f"P4_TUMOR_{section_id}_shifted_results.pkl")
        stable_file = os.path.join(metrics_dir, f"P4_TUMOR_{section_id}_stable_results.pkl")
    elif study == "P6_tumor":
        shifted_file = os.path.join(metrics_dir, f"P6_TUMOR_{section_id}_shifted_results.pkl")
        stable_file = os.path.join(metrics_dir, f"P6_TUMOR_{section_id}_stable_results.pkl")
    
    transitions = defaultdict(int)
    
    # Process shifted transitions
    with open(shifted_file, 'rb') as f:
        shifted_data = pickle.load(f)
        for key, variants in shifted_data['metrics_by_transition'].items():
            orig_gt, new_gt = key.split('_')[0].split('->')
            transitions[f"{orig_gt},{new_gt}"] += len(variants)
            
    # Process stable transitions
    with open(stable_file, 'rb') as f:
        stable_data = pickle.load(f)
        for key, variants in stable_data['metrics_by_transition'].items():
            orig_gt, new_gt = key.split('_')[0].split('->')
            transitions[f"{orig_gt},{new_gt}"] += len(variants)
    
    # Print summary
    print("\nTransition Summary:")
    print("-" * 40)
    total_shifted = sum(count for key, count in transitions.items() if key.split(',')[0] != key.split(',')[1])
    total_stable = sum(count for key, count in transitions.items() if key.split(',')[0] == key.split(',')[1])
    print(f"Total shifted SNVs: {total_shifted:,}")
    print(f"Total stable SNVs: {total_stable:,}")
    
    # Generate Mermaid diagram
    mermaid = ["--- config: sankey: showValues: True ---", "sankey-beta"]
    for key, count in transitions.items():
        orig, new = key.split(',')
        mermaid.append(f"{orig}original,{new}corrected,{count}")
    
    return '\n'.join(mermaid)

if __name__ == "__main__":
    mermaid_diagram = load_transition_data()
    print("\nMermaid Diagram:")
    print(mermaid_diagram)