const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync(process.argv[2],'utf8');
const source=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join('\n');
function setup(){
 let now=1000000,nextId=0;const timers=new Map(),elements={},docEvents={},windowEvents={};
 for(const m of html.matchAll(/<[^>]*\bid="([^"]+)"[^>]*>/g)){
  const attrs={},events={};const e={hidden:/\bhidden\b/.test(m[0]),disabled:/\bdisabled\b/.test(m[0]),value:'2',style:{},dataset:{},textContent:'',classList:{toggle(){}},
   addEventListener:(k,fn)=>events[k]=fn,removeAttribute:k=>delete attrs[k],hasAttribute:k=>Object.hasOwn(attrs,k),setAttribute:(k,v)=>attrs[k]=v,focus(){},events};
  Object.defineProperty(e,'src',{get:()=>attrs.src,set:v=>attrs.src=v});elements[m[1]]=e;
 }
 const data=()=>({generation:0,age_ms:0,active:false,ready:true,reason:'TEST',left:0,right:0,readiness:{arm_start_light_ready:true,direct_start_ready:true,laps:{enabled:false,total_laps:2},start_light:{enabled:true,armed:false,connected:true,phase:'DISARMED',peer_url:'http://10.0.0.2/',search_roi:[96,8,128,112],target_locked:false,required_frames:4}}});
 const env={elements,fetcher:async()=>data()};
 const document={hidden:false,getElementById:id=>{assert.ok(elements[id],id);return elements[id]},querySelectorAll:()=>[],addEventListener:(k,fn)=>docEvents[k]=fn};
 const window={addEventListener:(k,fn)=>windowEvents[k]=fn};
 const context=vm.createContext({document,window,navigator:{onLine:true},Date:{now:()=>now},URL,Number,Array,Math,Error,Promise,console:{warn(){}},AbortSignal:{timeout:()=>({})},localStorage:{getItem:()=>null,setItem(){}},
  fetch:async()=>({ok:true,json:async()=>env.fetcher()}),setTimeout:(fn,delay)=>{const id=++nextId;timers.set(id,{fn,at:now+delay});return id},clearTimeout:id=>timers.delete(id),setInterval:()=>++nextId});
 vm.runInContext(source,context);
 env.context=context;env.docEvents=docEvents;env.windowEvents=windowEvents;env.defaultData=data;
 env.advance=ms=>{const until=now+ms;for(let i=0;i<100;i++){let chosen;for(const [id,t] of timers)if(t.at<=until&&(!chosen||t.at<chosen[1].at))chosen=[id,t];if(!chosen)break;now=chosen[1].at;timers.delete(chosen[0]);chosen[1].fn();}now=until};
 env.delay=ms=>now+=ms;return env;
}
const flush=async()=>{for(let i=0;i<8;i++)await Promise.resolve()};
async function main(){
 {
  const e=setup();await flush();const image=e.elements['esp-video'];assert.ok(image.hasAttribute('src'));image.onload();
  e.fetcher=async()=>{throw Error('network unavailable')};await e.context.updateStatus();
  assert.equal(image.hidden,false);assert.ok(image.hasAttribute('src'));assert.match(e.elements['esp-state'].textContent,/信号状态未知/);
  assert.equal(e.elements['direct-start'].disabled,true);assert.equal(e.elements.start.disabled,true);
 }
 {
  const e=setup();await flush();const image=e.elements['esp-video'],oldLoad=image.onload,oldURL=image.src;
  e.advance(1500);assert.equal(image.hasAttribute('src'),false);assert.equal(e.elements['esp-state'].textContent,'预览重连');
  e.advance(250);assert.ok(image.hasAttribute('src'));assert.notEqual(image.src,oldURL);oldLoad();assert.equal(image.hidden,true);
  image.onload();assert.equal(image.hidden,false);
 }
 {
  const e=setup();await flush();e.context.document.hidden=true;e.docEvents.visibilitychange();
  assert.equal(e.elements.video.hasAttribute('src'),false);assert.equal(e.elements['esp-video'].hasAttribute('src'),false);
  e.advance(5000);assert.equal(e.elements['esp-video'].hasAttribute('src'),false);
  e.context.document.hidden=false;e.docEvents.visibilitychange();await flush();
  assert.ok(e.elements.video.hasAttribute('src'));assert.ok(e.elements['esp-video'].hasAttribute('src'));
 }
 {
  const e=setup();await flush();e.context.navigator.onLine=false;e.windowEvents.offline();
  assert.equal(e.elements['esp-state'].textContent,'网络中断');assert.equal(e.elements.start.disabled,true);
  e.advance(5000);assert.equal(e.elements['esp-video'].hasAttribute('src'),false);
  e.context.navigator.onLine=true;e.windowEvents.online();await flush();assert.ok(e.elements['esp-video'].hasAttribute('src'));
 }
 {
  const e=setup();await flush();e.fetcher=async()=>{e.delay(1100);return e.defaultData()};await e.context.updateStatus();
  assert.equal(e.elements['direct-start'].disabled,true);assert.match(e.elements.status.textContent,/状态更新中断/);
 }
 {
  const e=setup();await flush();e.fetcher=async()=>{const s=e.defaultData();s.readiness.crosswalk={phase:'HOLD',hold_elapsed_s:null};return s};await e.context.updateStatus();
  assert.match(e.elements.status.textContent,/显示异常/);assert.notEqual(e.elements['esp-state'].textContent,'离线');
 }
 {
  const e=setup();await flush();const old=e.elements.video.src;
  e.fetcher=async()=>{throw Error('network unavailable')};await e.context.updateStatus();
  e.fetcher=async()=>e.defaultData();await e.context.updateStatus();
  assert.notEqual(e.elements.video.src,old);
  const oldError=e.elements.video.onerror;e.elements.video.onerror();
  e.advance(250);const resumed=e.elements.video.src;oldError();assert.equal(e.elements.video.src,resumed);
 }
 {
  const e=setup();await flush();e.fetcher=async()=>{const s=e.defaultData();s.readiness.start_light={enabled:true,phase:'RUNNING',permission:true,armed:false,connected:false,error:'ESP32_OFFLINE',event_confirmed:true,confirmation_frames:4,required_frames:3};s.active=true;s.reason='FOLLOW';s.left=120;s.right=120;return s};await e.context.updateStatus();
  assert.match(e.elements['start-light-status'].textContent,/许可已锁存.*不影响本轮行驶/);
  assert.match(e.elements['start-light-status'].textContent,/3\/3帧/);
  assert.doesNotMatch(e.elements['start-light-status'].textContent,/信号中断|重新 ARM|许可已过期/);
  assert.match(e.elements.status.textContent,/运行中.*L=120 R=120/);
 }
 {
  const e=setup();await flush();e.fetcher=async()=>{const s=e.defaultData();Object.assign(s.readiness.start_light,{phase:'FAULT',connected:true,fault_reason:'RESPONSE_TOO_SLOW'});return s};await e.context.updateStatus();
  assert.match(e.elements['start-light-status'].textContent,/响应超时.*通信已恢复/);
  assert.doesNotMatch(e.elements['start-light-status'].textContent,/信号中断/);
 }
 {
  const e=setup();await flush();e.fetcher=async()=>{const s=e.defaultData();s.ready=false;s.readiness.direct_start_ready=false;s.readiness.blocked_by=['UNO_EMERGENCY_STOP'];return s};await e.context.updateStatus();
  assert.equal(e.elements['direct-start'].disabled,true);
  assert.match(e.elements.status.textContent,/UNO急停尚未解除/);
  assert.match(e.elements.diagnostics.textContent,/UNO急停尚未解除/);
 }
 {
  const e=setup();await flush();e.fetcher=async()=>{const s=e.defaultData();s.readiness.crosswalk={mode:'yield',phase:'PASSING',distance_to_body_m:.24,hold_elapsed_s:3.2,yield_completed:true,yield_verified:false};return s};await e.context.updateStatus();
  assert.match(e.elements['crosswalk-status'].textContent,/异常通行：停位未验证/);
  assert.doesNotMatch(e.elements['crosswalk-status'].textContent,/礼让已完成|合法停位与停留已验证/);
 }
 {
  const e=setup();await flush();e.fetcher=async()=>{const s=e.defaultData();Object.assign(s.readiness.start_light,{search_roi:[0,0,320,240],target_locked:true,target_roi:[64,60,60,52]});return s};await e.context.updateStatus();
  e.elements['esp-video'].onload();
  assert.equal(e.elements['esp-roi'].hidden,true);
  assert.equal(e.elements['esp-target'].hidden,false);
 }
 {
  const e=setup();await flush();e.fetcher=async()=>{const s=e.defaultData();s.readiness.crosswalk={mode:'yield',phase:'SEARCH',observation:null,hold_elapsed_s:3.2,yield_completed:true,yield_verified:true};return s};await e.context.updateStatus();
  assert.match(e.elements['crosswalk-status'].textContent,/上次横道：合法停位与停留已验证/);
  e.fetcher=async()=>{const s=e.defaultData();s.readiness.crosswalk={mode:'yield',phase:'APPROACH',observation:{near_m:.3},hold_elapsed_s:0,yield_completed:false,yield_verified:false};return s};await e.context.updateStatus();
  assert.doesNotMatch(e.elements['crosswalk-status'].textContent,/上次横道|合法停位与停留已验证/);
 }
 console.log('13 preview connection scenarios passed');
}
main().catch(e=>{console.error(e);process.exitCode=1});
