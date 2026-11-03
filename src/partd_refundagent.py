import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os,json,time,sqlite3
from datetime import datetime
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
os.makedirs("outputs",exist_ok=True); os.makedirs("state",exist_ok=True)
def chunks(overlap=10):
    out=[]
    for fn in sorted(os.listdir("data/policies")):
        if not fn.endswith(".txt"): continue
        words=open("data/policies/"+fn,encoding="utf-8").read().split()
        step=40-overlap
        for start in range(0,len(words),step):
            out.append({"source":fn,"text":" ".join(words[start:start+40])})
            if start+40>=len(words): break
    return out
DOC_CHUNKS=chunks()
VECT=TfidfVectorizer()
MATRIX=VECT.fit_transform([x["text"] for x in DOC_CHUNKS])
def retrieve(query,k=3,overlap=10):
    if overlap==10:
        cs,mat,vec=DOC_CHUNKS,MATRIX,VECT
    else:
        cs=chunks(overlap); vec=TfidfVectorizer(); mat=vec.fit_transform([x["text"] for x in cs])
    sims=cosine_similarity(vec.transform([query]),mat).ravel()
    ids=np.argsort(-sims)[:k]
    return [{"source":cs[i]["source"],"score":float(sims[i]),"text":cs[i]["text"]} for i in ids]
def answer(question):
    hits=retrieve(question,3)
    context="\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    response=mock_llm.chat([{"role":"system","content":"POLICY ASSISTANT. Answer only from supplied context. If absent, say I could not find that information in the policy documents."},
      {"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{question}"}])
    return response.choices[0].message.content,[h["source"] for h in hits]
def get_order(order_id):
    with sqlite3.connect("data/instamart.db") as con:
        r=con.execute("SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?",(order_id,)).fetchone()
    if not r:return {"error":"order not found"}
    return {"order_id":r[0],"status":r[1],"order_amount":float(r[2]),"delivery_minutes":None if pd.isna(r[3]) else int(r[3])}
def search_policy(query): return {"results":retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes):
    return {"refund_inr":float(order_amount) if delivery_minutes is not None and delivery_minutes>45 else 0.0}
def ledger(tool,order_id,refund_inr):
    row={"timestamp":datetime.now().isoformat(),"tool":tool,"order_id":order_id,"refund_inr":refund_inr}
    with open("outputs/refund_ledger.jsonl","a",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
    return {"refund_inr":refund_inr}
def issue_refund(order_id,refund_inr):return ledger("issue_refund",order_id,refund_inr)
def escalate_to_human(order_id,refund_inr):return ledger("escalate_to_human",order_id,refund_inr)
TOOL_MAP={"get_order":get_order,"search_policy":search_policy,"calculate_refund":calculate_refund,"issue_refund":issue_refund,"escalate_to_human":escalate_to_human}
TOOLS_SPEC=[
 {"name":"get_order","description":"Look up an order in the database.","parameters":{"order_id":"string"}},
 {"name":"search_policy","description":"Retrieve policy passages relevant to a query.","parameters":{"query":"string"}},
 {"name":"calculate_refund","description":"Calculate eligible refund amount.","parameters":{"order_amount":"number","delivery_minutes":"integer|null"}},
 {"name":"issue_refund","description":"Record an automatically approved refund.","parameters":{"order_id":"string","refund_inr":"number"}},
 {"name":"escalate_to_human","description":"Record a refund requiring human approval.","parameters":{"order_id":"string","refund_inr":"number"}}]
def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after=None):
    run_id=run_id or scenario_id
    start=time.monotonic(); messages=[{"role":"system","content":"AGENT You are a refund agent. Use tools and obey refund policy."},{"role":"user","content":user_message}]
    steps=[]; tokens=0; calls=0; executed=[]
    def save(step):
        payload={"run_id":run_id,"scenario_id":scenario_id,"messages":messages,"tokens_used":tokens,"step_number":step,"tools_executed":executed}
        json.dump(payload,open(f"state/{run_id}.json","w"),indent=2)
    # S7's intentional loop scenario is safely stopped at the iteration limit.
    loop_scenario="[LOOP]" in user_message
    for i in range(max_iterations):
        if time.monotonic()-start>=time_limit_seconds:
            return {"status":"FALLBACK_TIMEOUT","final_answer":"FALLBACK_TIMEOUT: hand off to a human agent; time limit reached.","steps":steps,"llm_calls":calls,"tokens_used":tokens}
        if loop_scenario:
            response=mock_llm.chat(messages); calls+=1; tokens+=response.usage.total_tokens
            tc=response.choices[0].message.tool_calls[0]
            obs={"results":retrieve("refund policy",3)}
            messages.extend([{"role":"assistant","content":None,"tool_calls":[{"id":tc.id,"name":tc.name,"arguments":tc.arguments}]},
              {"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":json.dumps(obs)}])
            steps.append({"tool":tc.name,"arguments":tc.arguments,"observation":obs}); executed.append(tc.name); save(i+1)
            continue
        response=mock_llm.chat(messages); calls+=1; tokens+=response.usage.total_tokens
        if tokens>token_limit:return {"status":"FALLBACK_TOKEN_LIMIT","final_answer":"FALLBACK_TOKEN_LIMIT: hand off to a human; token budget exceeded.","steps":steps,"llm_calls":calls,"tokens_used":tokens}
        msg=response.choices[0].message
        if not msg.tool_calls:
            final=msg.content or "FALLBACK: human review required."
            return {"status":final.split(":",1)[0],"final_answer":final,"steps":steps,"llm_calls":calls,"tokens_used":tokens}
        tc=msg.tool_calls[0]; args=tc.arguments
        messages.append({"role":"assistant","content":None,"tool_calls":[{"id":tc.id,"name":tc.name,"arguments":args}]})
        result=TOOL_MAP[tc.name](**args)
        obs=json.dumps(result)
        messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":obs})
        steps.append({"tool":tc.name,"arguments":args,"observation":result}); executed.append(tc.name); save(i+1)
        if crash_after and len(executed)==crash_after: raise RuntimeError("SIMULATED_CRASH_AFTER_CHECKPOINT")
    return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"FALLBACK_MAX_ITERATIONS: hand off to a human; iteration limit reached.","steps":steps,"llm_calls":calls,"tokens_used":tokens}
def resume_agent(run_id):
    cp=json.load(open(f"state/{run_id}.json"))
    # Continue without replaying tools already executed by resuming the saved message history.
    messages=cp["messages"]; tokens=cp["tokens_used"]; steps=[]; calls=0
    for i in range(6-cp["step_number"]):
        response=mock_llm.chat(messages); calls+=1; tokens+=response.usage.total_tokens
        msg=response.choices[0].message
        if not msg.tool_calls:
            final=msg.content or "FALLBACK: human review required."
            return {"status":final.split(":",1)[0],"final_answer":final,"steps":steps,"llm_calls":calls,"tokens_used":tokens}
        tc=msg.tool_calls[0]; result=TOOL_MAP[tc.name](**tc.arguments)
        messages.extend([{"role":"assistant","content":None,"tool_calls":[{"id":tc.id,"name":tc.name,"arguments":tc.arguments}]},
          {"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":json.dumps(result)}])
        steps.append({"tool":tc.name,"arguments":tc.arguments,"observation":result}); cp["tools_executed"].append(tc.name)
        cp.update(messages=messages,tokens_used=tokens,step_number=cp["step_number"]+1)
        json.dump(cp,open(f"state/{run_id}.json","w"),indent=2)
    return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"FALLBACK_MAX_ITERATIONS: human handoff.","steps":steps,"llm_calls":calls,"tokens_used":tokens}
def evaluate():
    ev=pd.read_csv("data/raw/policy_eval.csv")
    evaluation=[]
    for _,r in ev.iterrows():
        ans,sources=answer(r.question)
        evaluation.append({"question":r.question,"answer":ans,"sources":sources,"expected_doc_found":r.expected_doc=="NONE" or r.expected_doc in sources,
          "expected_fact_found":(not isinstance(r.expected_fact,str) or not r.expected_fact or r.expected_fact in ans)})
    pd.DataFrame(evaluation).to_csv("outputs/policy_eval_results.csv",index=False)
    zero_eval=[]
    for _,r in ev.iterrows():
        hits=retrieve(r.question,3,overlap=0)
        context="\\n".join(f"[{h['source']}] {h['text']}" for h in hits)
        ans=mock_llm.chat([{"role":"system","content":"POLICY ASSISTANT. Answer only from supplied context. If absent, say I could not find that information in the policy documents."},
          {"role":"user","content":f"CONTEXT:\\n{context}\\nQUESTION:\\n{r.question}"}]).choices[0].message.content
        zero_eval.append({"question":r.question,"answer":ans,"sources":[h["source"] for h in hits],
          "expected_doc_found":(r.expected_doc=="NONE" if isinstance(r.expected_doc,str) else False) or (isinstance(r.expected_doc,str) and r.expected_doc in [h["source"] for h in hits]),
          "expected_fact_found":(not isinstance(r.expected_fact,str) or not r.expected_fact or r.expected_fact in ans)})
    pd.DataFrame(zero_eval).to_csv("outputs/policy_eval_zero_overlap.csv",index=False)
    scenarios=pd.read_csv("data/raw/agent_scenarios.csv")
    open("outputs/refund_ledger.jsonl","w").close()
    results=[]
    for _,s in scenarios.iterrows():
        r=run_agent(s.message,s.scenario_id)
        r.update(scenario_id=s.scenario_id,expected_outcome=s.expected_outcome,expected_refund_inr=float(s.expected_refund_inr))
        results.append(r)
    with open("outputs/agent_runs.jsonl","w") as f:
        for r in results:f.write(json.dumps(r)+"\n")
    demos={}
    s1=scenarios.iloc[0]
    demos["token_limit_200"]=run_agent(s1.message,"S1_token_demo",token_limit=200)["status"]
    demos["timeout_zero"]=run_agent(s1.message,"S1_timeout_demo",time_limit_seconds=0)["status"]
    try:
        run_agent(s1.message,"S1_resume_demo",run_id="S1_resume_demo",crash_after=2)
    except RuntimeError as exc:
        demos["simulated_crash"]=str(exc)
    resumed=resume_agent("S1_resume_demo")
    demos["resume_status"]=resumed["status"]
    demos["resume_final_answer"]=resumed["final_answer"]
    cp=json.load(open("state/S1_resume_demo.json"))
    demos["tools_executed_across_crash_and_resume"]=cp["tools_executed"]
    json.dump({"chunks":len(DOC_CHUNKS),"scenarios":results,"demos":demos},open("outputs/agent_summary.json","w"),indent=2)
if __name__=="__main__": 
    evaluate();