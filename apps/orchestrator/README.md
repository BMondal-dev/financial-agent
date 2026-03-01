# Orchestration

`apps/orchestrator/server/api/run-experiment.post.ts`

Pseudo flow:
```
1. Load metadata.json
2. Pick target
3. Run baseline → neighbors: []
4. Run correlated → metadata[target].top_correlated.slice(0,3)
5. Run sector → metadata[target].same_sector.slice(0,3)
6. Compare MAE
7. Return result
```