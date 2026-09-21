#!/usr/bin/env python3
"""One US eight-GPU run, bootstrapped by container CMD; results travel via logs."""
import argparse,base64,hashlib,importlib.util,io,json,os,shlex,subprocess,sys,tarfile,time,uuid,zlib
from pathlib import Path
MODULE=Path(__file__).with_name('runpod-rtx-topology.py')
spec=importlib.util.spec_from_file_location('supervisor',MODULE)
supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
# Pinned linux/amd64 manifest of nvidia/cuda:12.8.1-devel-ubuntu24.04.
IMAGE = "nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43"
TOTAL_SECONDS = 1560
BOOT_SECONDS = 420
MAX_HOURLY_RATE = 16.8
PRIOR_COMPUTE_USD = 9.2944
CPU_RESERVE_USD = 0.10
INCIDENTAL_RESERVE_USD = 0.60
REQUIRED_BUDGET_USD = 20.0


def container_command():
 bootstrap = "import os,base64,zlib,json;bundle=json.loads(zlib.decompress(base64.b64decode(os.environ['RTX_PAYLOAD'])));exec(bundle['runner'])"
 # CUDA devel provides nvcc, but does not promise Python or pip. A venv avoids
 # Ubuntu's externally-managed Python restriction without replacing system pip.
 setup = "\n".join([
  "set -eu", "mkdir -p /workspace", "echo RTX_CONTAINER_START",
  "export DEBIAN_FRONTEND=noninteractive",
  "timeout 150 bash -c 'apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=20 update && apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=20 install -y --no-install-recommends python3-venv ca-certificates'",
  "timeout 30 python3 -m venv /workspace/rtx-python",
  "export PATH=/workspace/rtx-python/bin:$PATH",
  "exec /workspace/rtx-python/bin/python3 -u -c " + shlex.quote(bootstrap),
 ])
 return shlex.join(["bash", "-c", setup])


REMOTE=r'''
import base64,hashlib,io,json,os,subprocess,tarfile,time
from pathlib import Path
p=Path('/workspace/rtx-log-suite');p.mkdir(parents=True,exist_ok=True)
marker=p/'attempt'
try: marker.open('x').close()
except FileExistsError:
 print('RTX_DUPLICATE_START',flush=True);time.sleep(1500);raise SystemExit(2)
print('RTX_BOOT',flush=True)
for name,source in bundle['files'].items():(p/name).write_text(source)
job=subprocess.Popen(['bash',str(p/'direct.sh'),'--output',str(p/'results'),'--suite',str(p/'suite.py'),'--base-image',bundle['image'],'--prepare-seconds','240','--work-seconds','660','--total-seconds','960',*(['--collectives-only','--container-compat'] if bundle.get('collectives_only') else []),*(['--transport-diagnosis'] if bundle.get('transport_diagnosis') else []),*(['--same-node-comparison','--container-compat'] if bundle.get('same_node_comparison') else [])])
while job.poll() is None:
 time.sleep(5)
 f=p/'results/direct-metadata.json'
 if f.exists():
  try:
   d=json.loads(f.read_text());print('RTX_PROGRESS '+json.dumps({'phase':d.get('phase'),'elapsed':d.get('elapsed_seconds'),'step':d.get('steps',[{}])[-1].get('name'),'suite_step':json.loads((p/'results/results/metadata.json').read_text()).get('steps',[{}])[-1].get('name') if (p/'results/results/metadata.json').exists() else None}),flush=True)
  except (ValueError,IndexError):pass
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as archive:
 if (p/'results').exists():archive.add(p/'results',arcname='results')
data=buf.getvalue();encoded=base64.b64encode(data).decode();chunks=[encoded[i:i+4096] for i in range(0,len(encoded),4096)]
print('RTX_RESULT '+json.dumps({'count':len(chunks),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'returncode':job.returncode}),flush=True)
for i,part in enumerate(chunks):print('RTX_CHUNK '+str(i)+' '+part,flush=True)
print('RTX_DONE',flush=True)
time.sleep(180)
'''

def payload(files,image,collectives_only=False,transport_diagnosis=False,same_node_comparison=False):
 bundle={'files':files,'image':image,'runner':REMOTE,'collectives_only':collectives_only,'transport_diagnosis':transport_diagnosis,'same_node_comparison':same_node_comparison}
 return base64.b64encode(zlib.compress(json.dumps(bundle).encode())).decode()

def unpack(chunks,header,target):
 data=base64.b64decode(''.join(chunks[i] for i in range(header['count'])),validate=True)
 if len(data)!=header['bytes'] or hashlib.sha256(data).hexdigest()!=header['sha256']:raise ValueError('Archive checksum mismatch')
 (target/'results.tar.gz').write_bytes(data)
 with tarfile.open(fileobj=io.BytesIO(data)) as archive:
  for m in archive:
   if Path(m.name).is_absolute() or '..' in Path(m.name).parts or not (m.isfile() or m.isdir()):raise ValueError('Unsafe archive')
  archive.extractall(target)
 return data

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output');ap.add_argument('--collectives-only',action='store_true');ap.add_argument('--transport-diagnosis',action='store_true');ap.add_argument('--data-center-ids');ap.add_argument('--same-node-comparison',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--approved-cumulative-budget',type=float,default=10.0);args=ap.parse_args()
 files={'suite.py':Path(__file__).with_name('rtx-topology-suite.py').read_text(),'direct.sh':Path(__file__).with_name('runpod-rtx-topology-direct.sh').read_text(),'numa-first-touch.py':Path(__file__).with_name('rtx-numa-first-touch.py').read_text()}
 encoded=payload(files,IMAGE,args.collectives_only,args.transport_diagnosis,args.same_node_comparison)
 if args.self_test:
  import tempfile
  decoded=json.loads(zlib.decompress(base64.b64decode(encoded)));assert decoded['files']==files;compile(decoded['runner'],'remote','exec')
  command=shlex.split(container_command());assert command[:2]==['bash','-c'];subprocess.run(['bash','-n','-c',command[2]],check=True)
  assert TOTAL_SECONDS - BOOT_SECONDS >= 960 + 90
  assert PRIOR_COMPUTE_USD + CPU_RESERVE_USD + INCIDENTAL_RESERVE_USD + TOTAL_SECONDS / 3600 * MAX_HOURLY_RATE <= REQUIRED_BUDGET_USD
  with tempfile.TemporaryDirectory() as folder:
   b=io.BytesIO()
   with tarfile.open(fileobj=b,mode='w:gz') as tar:
    raw=b'{"status":"complete"}';info=tarfile.TarInfo('results/metadata.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
   data=b.getvalue();parts={0:base64.b64encode(data).decode()};header={'count':1,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
   unpack(parts,header,Path(folder));assert json.loads((Path(folder)/'results/metadata.json').read_text())['status']=='complete'
   header['sha256']='0'*64
   try:unpack(parts,header,Path(folder))
   except ValueError:pass
   else:raise AssertionError('bad checksum accepted')
  print('Payload, remote syntax, archive roundtrip, checksum rejection passed; no cloud calls');return
 if args.data_center_ids and ',' in args.data_center_ids:raise ValueError('This CLI honors only one data center; select one verified location')
 if not args.execute:print(json.dumps({'country':'US','gpu_count':8,'create_calls':1,'total_seconds':TOTAL_SECONDS,'boot_seconds':BOOT_SECONDS,'max_hourly_rate':MAX_HOURLY_RATE,'image':IMAGE,'required_cumulative_budget_usd':REQUIRED_BUDGET_USD,'payload_bytes':len(encoded),'uses_ssh':False}));return
 if args.approved_cumulative_budget < REQUIRED_BUDGET_USD:raise ValueError('Updated plan requires explicit cumulative budget authorization of USD 20.00 before creating resources')
 if not args.output:raise ValueError('--output is required for execution')
 directory=Path(args.output).resolve();repo=Path(__file__).resolve().parents[1]
 if directory==repo or repo in directory.parents:raise ValueError('Private output required')
 os.umask(0o077);directory.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();run_spec={'name':'blog-rtx-log-'+uuid.uuid4().hex[:18],'parent_pid':os.getpid(),'created_unix':time.time(),'created_mono':start,'deadline_mono':start+TOTAL_SECONDS,'ready_deadline_mono':start+BOOT_SECONDS,'max_hourly_rate':MAX_HOURLY_RATE,'cli':str(Path.home()/'.local/bin/runpodctl'),'country_code':'US','image':IMAGE}
 supervisor.atomic_json(directory/'run.json',run_spec);run=supervisor.Run(directory)
 if run.list_owned():raise RuntimeError('Name exists')
 with (directory/'watchdog.log').open('ab') as log:subprocess.Popen([sys.executable,str(MODULE),'--watchdog',str(directory)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 for _ in range(50):
  if (directory/'watchdog-ready.json').exists():break
  time.sleep(.1)
 else:raise RuntimeError('Watchdog unavailable')
 header=None;chunks={};done=False;boot=False;failure=None;seen=set()
 try:
  creation=['pod','create','--name',run_spec['name'],'--gpu-id',supervisor.GPU,'--gpu-count','8','--cloud-type','SECURE','--country-code','US','--image',IMAGE,'--container-disk-in-gb','40','--volume-in-gb','0','--ssh=false','--min-cuda-version','12.8','--docker-args',container_command(),'--env',json.dumps({'RTX_PAYLOAD':encoded})]
  if args.data_center_ids:creation.extend(['--data-center-ids',args.data_center_ids])
  (directory/'create-attempted').touch();run.event('create_requested');code,created=run.cli(*creation,timeout=45)
  if code==0 and created.get('id'):run.claim({**created,'name':run_spec['name']})
  while not run.pod_id():
   matches=run.list_owned()
   if matches:run.claim(matches[0]);break
   rejection=supervisor.create_rejection_code(created)
   if rejection:
    supervisor.atomic_json(directory/'create-rejected.json',{'code':rejection});raise RuntimeError('Create rejected; no retry')
   if time.monotonic()>run_spec['ready_deadline_mono']:raise TimeoutError('Create timeout')
   time.sleep(5)
  while not done:
   run.remaining(reserve=30)
   code,pod=run.cli('pod','get',run.pod_id(),'--include-machine',timeout=15)
   if code==0:
    run.validate_pod(pod)
    location=(pod.get('machine') or {}).get('location')
    if location and location!='US':raise RuntimeError('Wrong country')
   log_path=directory/f'logs-{time.time_ns()}.jsonl'
   with log_path.open('wb') as output:
    try:r=subprocess.run([run_spec['cli'],'pod','logs',run.pod_id(),'--tail','5000','--max-wait','3s'],stdout=output,stderr=subprocess.PIPE,timeout=12)
    except subprocess.TimeoutExpired:r=None
   for line in log_path.read_text(errors='replace').splitlines():
    try:entry=json.loads(line);message=entry.get('line','')
    except ValueError:continue
    if 'error creating container' in message.lower() or 'failed to pull image' in message.lower():raise RuntimeError('Container startup failure; logs preserved')
    if message=='RTX_DUPLICATE_START':raise RuntimeError('Container restarted; refusing duplicate suite')
    if message=='RTX_BOOT' and not boot:
     boot=True;supervisor.atomic_json(directory/'ready.json',{'time':time.time()});run.event('container_boot');print('Container started; automatic suite running',flush=True)
    if message.startswith('RTX_PROGRESS ') and message not in seen:
     seen.add(message);(directory/'progress.json').write_text(message[len('RTX_PROGRESS '):])
    if message.startswith('RTX_RESULT '):header=json.loads(message[len('RTX_RESULT '):])
    if message.startswith('RTX_CHUNK '):_,i,data=message.split(' ',2);chunks[int(i)]=data
    if message=='RTX_DONE':done=True
   if header and len(chunks)==header['count']:
    unpack(chunks,header,directory);supervisor.atomic_json(directory/'archive-verified.json',header);done=True
   if done and not (directory/'archive-verified.json').exists():done=False
   if not boot and time.monotonic()>run_spec['ready_deadline_mono']:raise TimeoutError(f'No boot event in {BOOT_SECONDS} seconds')
   if not done:time.sleep(10)
  if header['returncode']!=0:raise RuntimeError('Remote preparation/suite failed; complete diagnostic archive recovered')
  meta=json.loads((directory/'results/direct-metadata.json').read_text())
  suite=json.loads((directory/'results/results/metadata.json').read_text())
  if meta.get('status')!='complete' or suite.get('status')!='complete':raise RuntimeError('Incomplete metadata')
 except BaseException as error:failure=error;run.event('run_failed',detail=str(error))
 finally:
  (directory/'cleanup-requested').touch();run.cleanup()
  for _ in range(20):
   if (directory/'cleaned.json').exists():break
   time.sleep(1)
  cleaned=(directory/'cleaned.json').exists();supervisor.atomic_json(directory/'local-run.json',{'state':'failed' if failure else 'finished','error':str(failure) if failure else None,'elapsed_seconds':time.monotonic()-start,'cleanup_verified':cleaned})
 print(json.dumps({'state':'failed' if failure else 'finished','error':str(failure) if failure else None,'cleanup_verified':cleaned,'archive_verified':(directory/'archive-verified.json').exists()}))
 if failure or not cleaned:raise SystemExit(1)
if __name__=='__main__':main()
