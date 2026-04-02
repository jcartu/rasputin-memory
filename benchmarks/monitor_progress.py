#!/usr/bin/env python3
"""Monitor LoCoMo benchmark progress — prints when a conversation completes."""
import json
import time

CP = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/results/locomo-leaderboard-checkpoint.json"
EXPECTED = {'conv-26':199,'conv-30':105,'conv-41':193,'conv-42':260,'conv-43':242,'conv-44':158,'conv-47':190,'conv-48':239,'conv-49':196,'conv-50':204}
TOTAL_QA = sum(EXPECTED.values())

reported_done = set()

while True:
    try:
        cp = json.load(open(CP))
        convs = {}
        for k in cp['completed_keys']:
            c = k.rsplit('_',1)[0]
            convs[c] = convs.get(c, 0) + 1
        
        total = len(cp['completed_keys'])
        correct = sum(1 for r in cp['results'] if r.get('correct'))
        
        for c, exp in sorted(EXPECTED.items()):
            done = convs.get(c, 0)
            if done >= exp and c not in reported_done:
                reported_done.add(c)
                conv_results = [r for r in cp['results'] if r.get('conv_id') == c]
                conv_correct = sum(1 for r in conv_results if r.get('correct'))
                conv_nonadv = [r for r in conv_results if r.get('category') != 5]
                conv_nonadv_correct = sum(1 for r in conv_nonadv if r.get('correct'))
                
                remaining = sum(1 for cc, ee in EXPECTED.items() if convs.get(cc, 0) < ee)
                
                print(f"CONV_DONE|{c}|{conv_correct}/{len(conv_results)}|{conv_correct/len(conv_results)*100:.1f}%|nonadv:{conv_nonadv_correct}/{len(conv_nonadv)}|{conv_nonadv_correct/len(conv_nonadv)*100:.1f}%|total:{total}/{TOTAL_QA}|overall:{correct/total*100:.1f}%|remaining:{remaining}", flush=True)
        
        if total >= TOTAL_QA:
            print(f"ALL_DONE|{total}|{correct}|{correct/total*100:.1f}%", flush=True)
            break
            
    except Exception:
        pass
    
    time.sleep(30)
