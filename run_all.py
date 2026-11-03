"""Run all project stages in sequence. Use this after installing requirements."""
from src.parta_datavalidation import audit_data,clean_data,validate_data,pandas_crosscheck_eda
from src.partb_custsegmentsprediction import rfm_and_segments,build_lapse_model
from src.partc_tickettriage import run_pipeline,FEW
from src.partd_refundagent import DOC_CHUNKS, evaluate
def main_call():
    print("\n=== PART A: AUDIT / CLEAN / VALIDATE ===")
    audit_data(); 
    clean_data();  
    validate_data();
    pandas_crosscheck_eda()

    print("\n=== PART B: SEGMENTS AND LAPSE ===")
    rfm,evaluation,profile,pca=rfm_and_segments()
    
    print("K-means diagnostics:\n",evaluation.to_string(index=False))
    print("Segment profile:\n",profile.to_string(index=False)); print("PCA explained variance:",pca)
    print("Lapse model summary:",build_lapse_model())

    print("\n=== PART C: TICKET TRIAGE ===")
    triage,summary=run_pipeline(prompt=FEW,token_budget=100000)
    triage,summary=run_pipeline(prompt=FEW,token_budget=60000)
    print(summary); 
    print(triage.head().to_string(index=False))

    print("\n=== PART D: POLICY RAG ===")
    evaluate()
    print("chunks:", len(DOC_CHUNKS))
    print("\nProject pipeline finished. Review outputs/ and data/instamart.db.")
if __name__=="__main__": 
    print("Start")
    main_call()