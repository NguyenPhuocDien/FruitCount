import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
const root=path.resolve('analysis_outputs/qa_fixes')
fs.mkdirSync(root, { recursive: true })
const fixture=path.join(root,'slow-worker.py')
const pidFile=path.join(root,'worker-pids.json')
fs.writeFileSync(fixture,`import json, os, subprocess, sys, time\nchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\nwith open(sys.argv[1], 'w') as f: json.dump([os.getpid(), child.pid], f)\ntime.sleep(60)\n`)
const code=fs.readFileSync('lib/python-runner.ts','utf8').replace("const SCRIPT = path.join(/*turbopackIgnore: true*/ process.cwd(), 'scripts', 'run_inference.py')",`const SCRIPT = ${JSON.stringify(fixture)}`)
const load=async source=>import(`data:text/javascript;base64,${Buffer.from(ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2020,module:ts.ModuleKind.ESNext}}).outputText).toString('base64')}`)
const {runModelScript}=await load(code)
const wait=ms=>new Promise(r=>setTimeout(r,ms))
const alive=pid=>{try{process.kill(pid,0);return true}catch{return false}}
const results=[]
for(const mode of ['abort','timeout']) {
 fs.rmSync(pidFile,{force:true})
 const controller=new AbortController()
 const promise=runModelScript([pidFile],mode==='timeout'?2000:10000,controller.signal)
 const deadline=Date.now()+5000
 while(!fs.existsSync(pidFile)&&Date.now()<deadline)await wait(30)
 assert.ok(fs.existsSync(pidFile),'worker must start')
 const pids=JSON.parse(fs.readFileSync(pidFile))
 assert.ok(pids.every(alive),'parent and nested child are alive before cancellation')
 if(mode==='abort')controller.abort()
 const result=await promise
 assert.equal(result.ok,false)
 assert.ok(pids.every(pid=>!alive(pid)),'entire worker tree must be gone before runModelScript settles')
 results.push({mode,pids,result,treeStopped:true})
}
const cancelled=new AbortController();cancelled.abort()
assert.equal((await runModelScript([],1000,cancelled.signal)).ok,false)
const {validateUploadForm,InvalidUpload}=await load(fs.readFileSync('lib/upload-validation.ts','utf8'))
for(const key of ['modelFile','imageFile','labelsFile']){
 const form=new FormData();form.append(key,'bad');assert.throws(()=>validateUploadForm(form),InvalidUpload)
}
const valid=new FormData();valid.append('modelFile',new File(['x'],'x.pt'));validateUploadForm(valid)
const duplicate=new FormData();duplicate.append('imageFile',new File(['x'],'x.png'));duplicate.append('imageFile',new File(['y'],'y.png'));assert.throws(()=>validateUploadForm(duplicate),InvalidUpload)
fs.writeFileSync(path.join(root,'runner-regression.json'),JSON.stringify(results,null,2))
console.log('PASS: cancellation, timeout stop nested worker; pre-abort; file and duplicate-field validation')
