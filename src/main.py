"""Demo runner for evidence verification pipeline."""

import os
import sys
import json

# For direct execution, add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import build_graph

# Windows consoles default to cp1252, which cannot print the emoji below
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_demo():
    print("=== Evidence Retrieval and Verification Pipeline ===")
    
    app = build_graph()
    state = {
        "post_id": "p1",
        "post_text": "I read that mrna vaccines cause cancer !",
        "lang_hint": "en"
    }
    
    print(f"📝 Input: {state['post_text']}")
    
    result = app.invoke(state)
    
    # Claim Detection
    detection = result.get("claim_detection", {})
    if detection:
        has_claim = detection.get("has_claim", True)
        status = "Yes" if has_claim else "No"
        print(f"\n📌 Contains claim: {status}\n")
    
    # Claims
    claims = result.get("claims", [])
    print(f"🎯 Claims Extracted: {len(claims)}")
    for i, claim in enumerate(claims, 1):
        c = claim.model_dump() if hasattr(claim, 'model_dump') else claim
        print(f"  {i}. Claim: '{c.get('normalized_claim', 'N/A')}' (confidence: {c.get('confidence', 0):.2f})")
    print()
    
    # Evidence
    evidence = result.get("claim_evidence", [])
    print(f"🔍 Evidence Retrieved:")
    for ev in evidence:
        e = ev.model_dump() if hasattr(ev, 'model_dump') else ev
        print(f"  Claim {e.get('claim_id')}: {len(e.get('evidence', []))} items from authoritative sources")
    print()
    
    # Verification
    verifier = result.get("verifier_results", [])
    print(f"✅ Verification Results:")
    for v in verifier:
        vr = v.model_dump() if hasattr(v, 'model_dump') else v
        print(f"  Claim {vr.get('claim_id')}: Score {vr.get('claim_score')}")
        print(f"  Support confidence: {vr.get('support_confidence', 0):.2f}, Refute confidence: {vr.get('refute_confidence', 0):.2f}")
    print()
    
    # Manipulation
    manip = result.get("manipulation")
    if manip:
        m = manip.model_dump() if hasattr(manip, 'model_dump') else manip
        print(f"⚠️  Manipulation Score: {m.get('manipulation_score', 0):.2f}")
        print(f"   Explanation: {m.get('explanation', 'N/A')}")
    print()
    
    # Final Decision
    final = result.get("final_decision")
    if final:
        f = final.model_dump() if hasattr(final, 'model_dump') else final
        print(f"🏁 Final Decision: {f.get('label', 'N/A').upper()}")
        print(f"   Reason: {f.get('why', 'N/A')}")
        print(f"   Post Score: {f.get('post_verification_score')}")
        if f.get('top_refuting_urls'):
            print(f"   Refuting URLs:")
            for url in f.get('top_refuting_urls', [])[:3]:  # Show first 3
                print(f"     - {url}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    run_demo()
