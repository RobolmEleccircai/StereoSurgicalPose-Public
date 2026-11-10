"""Run isolated alternating baseline/candidate stereo benchmarks and summarize."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--baseline-config',type=Path,required=True)
    p.add_argument('--candidate-config',type=Path,required=True)
    p.add_argument('--candidate-overlay',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--rounds',type=int,default=10)
    a=p.parse_args()
    if a.repeats<3 or a.rounds<10:raise ValueError('Final comparison requires at least three repeats of 560 measured pairs')
    a.output.mkdir(parents=True,exist_ok=False)
    reports={'baseline':[],'candidate':[]};frames={'baseline':[],'candidate':[]}
    tool=Path(__file__).with_name('benchmark_realtime.py')
    for repeat in range(a.repeats):
        for mode,config in [('baseline',a.baseline_config),('candidate',a.candidate_config)]:
            output=a.output/f'{mode}_{repeat+1}'
            command=[sys.executable,str(tool),'--inputs',str(a.inputs),'--inference-backend','libtorch',
                     '--modes','fp32','--predictor-config',str(config),'--rounds',str(a.rounds),'--output',str(output)]
            if mode=='candidate':command+=['--overlay-library',str(a.candidate_overlay)]
            with (a.output/f'{mode}_{repeat+1}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads((output/'report.json').read_text())
            reports[mode].append(report)
            frames[mode]+=list(map(json.loads,(output/'frames.jsonl').read_text().splitlines()))
            r=report['results']['fp32']
            print(json.dumps({'mode':mode,'repeat':repeat+1,'mean_ms':r['end_to_end_ms']['mean_ms'],'p95_ms':r['end_to_end_ms']['p95_ms'],'hz':r['hz']}),flush=True)
    def stats(values):
        x=np.asarray(values);return {'mean_ms':float(x.mean()),'p50_ms':float(np.percentile(x,50)),
            'p95_ms':float(np.percentile(x,95)),'p99_ms':float(np.percentile(x,99)),
            'max_ms':float(x.max()),'hz':float(1000/x.mean()),'over_33_33_ms_fraction':float((x>1000/30).mean())}
    summary={}
    for mode in reports:
        summary[mode]={'measured_pairs':len(frames[mode]),
          'timings':{field:stats([r[field] for r in frames[mode]]) for field in
             ('end_to_end_ms','predictor_including_preprocess_transfer_export_ms','cpp_projection_and_draw_ms','network_cuda_interval_ms')},
          'runs':[{'mean_ms':r['results']['fp32']['end_to_end_ms']['mean_ms'],
                   'p95_ms':r['results']['fp32']['end_to_end_ms']['p95_ms'],
                   'cpu_percent':r['results']['fp32']['process_cpu_percent'],
                   'peak_rss_mib':r['results']['fp32']['process_peak_rss_mib'],
                   'peak_torch_cuda_allocated_mib':r['results']['fp32']['peak_cuda_allocated_mib'],
                   'actual_execution':r['actual_execution'],'graph_error':r['graph_error']} for r in reports[mode]]}
    improvement=1-summary['candidate']['timings']['end_to_end_ms']['mean_ms']/summary['baseline']['timings']['end_to_end_ms']['mean_ms']
    repeatable=max(r['mean_ms'] for r in summary['candidate']['runs'])<min(r['mean_ms'] for r in summary['baseline']['runs'])
    p95_pass=max(r['p95_ms'] for r in summary['candidate']['runs'])<=min(r['p95_ms'] for r in summary['baseline']['runs'])
    graph_pass=all(r['actual_execution']=='optimized_graph' and r['graph_error'] is None for r in summary['candidate']['runs'])
    result={'completed':True,'summary':summary,'latency_reduction':improvement,'repeatable_improvement':repeatable,
            'p95_pass':p95_pass,'graph_pass':graph_pass,'speed_goal_passed':improvement>=.1 and repeatable and p95_pass and graph_pass,
            'scope':reports['candidate'][0]['timing_scope'],'telemetry_scope':'GPU telemetry includes warmup and measurement; CPU percentage is measured-loop process CPU time / wall time.'}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['speed_goal_passed']:raise SystemExit('Final speed acceptance failed')
if __name__=='__main__':main()
