const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync(process.argv[2],'utf8');
const source=[...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join('\n');
function makeDOM(markup){
 const elements={},all=[],roots=[];let stack=[];
 function simpleMatch(element,selector){
  const tag=selector.match(/^[a-z][\w-]*/i);
  if(tag&&element.tagName!==tag[0].toUpperCase())return false;
  for(const id of selector.matchAll(/#([\w-]+)/g))if(element.id!==id[1])return false;
  for(const cls of selector.matchAll(/\.([\w-]+)/g))if(!element.classList.contains(cls[1]))return false;
  for(const attr of selector.matchAll(/\[([^\]=\s]+)(?:\s*=\s*["']?([^\]"']*)["']?)?\]/g)){
   if(!element.hasAttribute(attr[1]))return false;
   if(attr[2]!==undefined&&element.getAttribute(attr[1])!==attr[2])return false;
  }
  return true;
 }
 function matches(element,selector){
  return selector.split(',').some(choice=>{
   const parts=choice.trim().split(/\s+/);let current=element;
   if(!simpleMatch(current,parts.pop()))return false;
   while(parts.length){const wanted=parts.pop();current=current.parentElement;while(current&&!simpleMatch(current,wanted))current=current.parentElement;if(!current)return false;}
   return true;
  });
 }
 function node(tag,attrs){
  const events={},e={tagName:tag.toUpperCase(),attributes:attrs,children:[],parentElement:null,style:{},dataset:{},events,_text:null,_ownText:'',naturalWidth:320,naturalHeight:240};
  const reflected={hidden:'hidden',disabled:'disabled',open:'open',checked:'checked'};
  for(const [property,attribute] of Object.entries(reflected))Object.defineProperty(e,property,{get:()=>Object.hasOwn(attrs,attribute),set:value=>{if(value)attrs[attribute]='';else delete attrs[attribute]}});
  for(const [property,attribute] of Object.entries({id:'id',className:'class',src:'src',value:'value',type:'type',min:'min',max:'max'}))Object.defineProperty(e,property,{get:()=>attrs[attribute]??'',set:value=>attrs[attribute]=String(value)});
  Object.defineProperty(e,'valueAsNumber',{get:()=>Number(e.value),set:value=>e.value=String(value)});
  Object.defineProperty(e,'textContent',{get:()=>e._text??(e._ownText+e.children.map(c=>c.textContent).join('')),set:value=>e._text=String(value)});
  for(const [key,value] of Object.entries(attrs))if(key.startsWith('data-'))e.dataset[key.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=value;
  const classes=()=>new Set((attrs.class||'').split(/\s+/).filter(Boolean));
  e.classList={contains:name=>classes().has(name),add(...names){const c=classes();names.forEach(n=>c.add(n));attrs.class=[...c].join(' ')},remove(...names){const c=classes();names.forEach(n=>c.delete(n));attrs.class=[...c].join(' ')},toggle(name,force){const selected=force===undefined?!classes().has(name):!!force;selected?this.add(name):this.remove(name);return selected}};
  e.addEventListener=(event,fn)=>{const prior=events[event];events[event]=prior?payload=>{prior(payload);fn(payload)}:fn};
  e.removeEventListener=event=>delete events[event];
  e.setAttribute=(key,value)=>attrs[key]=String(value);e.getAttribute=key=>attrs[key]??null;e.removeAttribute=key=>delete attrs[key];e.hasAttribute=key=>Object.hasOwn(attrs,key);
  e.focus=()=>{};e.setPointerCapture=()=>{};e.releasePointerCapture=()=>{};
  e.getBoundingClientRect=()=>({left:0,top:0,right:320,bottom:240,width:320,height:240});
  e.matches=selector=>matches(e,selector);
  e.closest=selector=>{for(let current=e;current;current=current.parentElement)if(matches(current,selector))return current;return null};
  e.contains=other=>{for(let current=other;current;current=current.parentElement)if(current===e)return true;return false};
  e.querySelectorAll=selector=>all.filter(other=>other!==e&&e.contains(other)&&matches(other,selector));
  e.querySelector=selector=>e.querySelectorAll(selector)[0]??null;
  all.push(e);if(e.id)elements[e.id]=e;return e;
 }
 const markupWithoutScripts=markup.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi,'');
 const voidTags=new Set(['area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr']);
 let previous=0;
 for(const token of markupWithoutScripts.matchAll(/<\/?([a-z][\w-]*)([^<>]*?)\/?\s*>/gi)){
  if(stack.length)stack.at(-1)._ownText+=markupWithoutScripts.slice(previous,token.index).replace(/\s+/g,' ').trim();
  previous=token.index+token[0].length;const tag=token[1].toLowerCase();
  if(token[0].startsWith('</')){while(stack.length){if(stack.pop().tagName===tag.toUpperCase())break;}continue;}
  const attrs={};for(const attribute of token[2].matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g))attrs[attribute[1]]=attribute[2]??attribute[3]??attribute[4]??'';
  const element=node(tag,attrs);if(stack.length){element.parentElement=stack.at(-1);element.parentElement.children.push(element)}else roots.push(element);
  if(!voidTags.has(tag)&&!token[0].endsWith('/>'))stack.push(element);
 }
 return {elements,all,roots,node,matches};
}
function setup(options={}){
 let now=1000000,nextId=0;const timers=new Map(),docEvents={},windowEvents={};
 const dom=makeDOM(html),elements=dom.elements;
 const data=()=>({generation:0,age_ms:0,active:false,ready:true,reason:'TEST',left:0,right:0,readiness:{arm_start_light_ready:true,direct_start_ready:true,laps:{enabled:false,total_laps:2},start_light:{enabled:true,armed:false,connected:true,phase:'DISARMED',peer_url:'http://10.0.0.2/',search_roi:[96,8,128,112],target_locked:false,required_frames:4,lamp_configuration:{manual:true,roi:[139,51,63,61]},configuration_update:{pending:false,state:'IDLE',error:'',request_id:0}}}});
 const env={elements,dom,requests:[],fetcher:async()=>data(),storage:new Map(Object.entries(options.storage||{}))};
 const document={hidden:false,getElementById:id=>{assert.ok(elements[id],id);return elements[id]},querySelectorAll:selector=>dom.all.filter(e=>dom.matches(e,selector)),querySelector:selector=>dom.all.find(e=>dom.matches(e,selector))??null,createElement:tag=>dom.node(tag,{}),addEventListener:(k,fn)=>docEvents[k]=fn};
 const window={addEventListener:(k,fn)=>windowEvents[k]=fn};
 const context=vm.createContext({document,window,navigator:{onLine:true},Date:{now:()=>now},URL,Number,Array,Math,Error,Promise,console:{warn(){}},AbortSignal:{timeout:()=>({})},localStorage:{getItem:key=>env.storage.get(key)??null,setItem:(key,value)=>env.storage.set(key,String(value))},
  fetch:async(url,options={})=>{env.requests.push({url:String(url),method:options.method||'GET',body:options.body?JSON.parse(options.body):undefined,headers:options.headers});const result=await env.fetcher(String(url),options);return result&&result.__response?{ok:result.status>=200&&result.status<300,status:result.status,json:async()=>result.body}:{ok:true,status:200,json:async()=>result}},
  setTimeout:(fn,delay)=>{const id=++nextId;timers.set(id,{fn,at:now+delay});return id},clearTimeout:id=>timers.delete(id),setInterval:()=>++nextId});
 vm.runInContext(source,context);
 env.context=context;env.docEvents=docEvents;env.windowEvents=windowEvents;env.defaultData=data;
 env.response=(body,status=200)=>({__response:true,body,status});
 env.posts=path=>env.requests.filter(request=>request.method==='POST'&&(!path||request.url===path));
 env.evaluate=expression=>vm.runInContext(expression,context);
 env.inline=(id,event)=>{const element=elements[id],handler=element.getAttribute('on'+event);if(!handler)return;context.__inlineTarget=element;return vm.runInContext('(function(){return '+handler+'}).call(__inlineTarget)',context)};
 env.click=id=>elements[id].disabled?undefined:env.inline(id,'click');
 env.input=(id,value)=>{elements[id].value=String(value);return env.inline(id,'input')};
 env.advance=ms=>{const until=now+ms;for(let i=0;i<100;i++){let chosen;for(const [id,t] of timers)if(t.at<=until&&(!chosen||t.at<chosen[1].at))chosen=[id,t];if(!chosen)break;now=chosen[1].at;timers.delete(chosen[0]);chosen[1].fn();}now=until};
 env.delay=ms=>now+=ms;return env;
}

const flush=async()=>{for(let i=0;i<8;i++)await Promise.resolve()};
const roiEdges=['left','right','top','bottom'];
const bounds=e=>roiEdges.map(edge=>Number(e.elements['roi-'+edge].value));
const loadedESP=e=>{assert.equal(typeof e.elements['esp-video'].onload,'function');e.elements['esp-video'].onload()};
function assertStopUsable(e){
 const stop=e.elements.stop;assert.equal(stop.hidden,false);assert.equal(stop.disabled,false);
 assert.equal(stop.closest('details'),null);assert.ok(stop.closest('.mission-bar'));
}
async function replaceStatus(e,mutate){
 e.fetcher=async()=>{const status=e.defaultData();mutate(status);return status};
 await e.context.updateStatus();
}
async function fieldScenarios(){
 {
  const e=setup();await flush();
  assert.equal(e.elements['video-window'].dataset.view,'annotated');
  assert.equal(e.elements['camera-grid'].dataset.view,'annotated');
  assert.equal(e.elements.labels.hidden,true);
  const selected=e.context.document.querySelectorAll('.view-controls button.selected');
  assert.equal(selected.length,1);assert.equal(selected[0].dataset.view,'annotated');
  for(const name of ['diagnostics-panel','advanced-panel']){assert.equal(e.elements[name].tagName,'DETAILS');assert.equal(e.elements[name].open,false)}
  assert.equal(e.elements['direct-start'].closest('details').id,'advanced-panel');
  assert.equal(e.elements.start.closest('details'),null);assert.equal(e.elements['target-laps'].closest('details'),null);
  assert.equal(e.dom.all.filter(element=>element.tagName==='BUTTON'&&element.getAttribute('onclick')==="control('stop')").length,1);
  assertStopUsable(e);
  const legacy=setup({storage:{'curve-camera-view':'all'}});await flush();
  assert.equal(legacy.elements['video-window'].dataset.view,'annotated');
  const remembered=setup({storage:{'race-console-view-v1':'ground'}});await flush();
  assert.equal(remembered.elements['video-window'].dataset.view,'ground');
 }
 {
  const e=setup();await flush();
  // Status freshness cannot stand in for a loaded ESP frame when editing.
  assert.equal(e.elements['roi-edit'].disabled,true);e.click('roi-edit');
  assert.equal(e.elements['roi-tools'].hidden,true);assert.equal(e.posts('/lamp-config').length,0);
  loadedESP(e);assert.equal(e.elements['roi-edit'].disabled,false);
  assert.equal(e.elements['esp-target'].hidden,false);
  assert.equal(e.elements['esp-target'].style.left,139/320*100+'%');
  e.click('roi-edit');assert.deepEqual(bounds(e),[139,202,51,112]);
  assert.equal(e.elements['roi-layer'].hidden,false);assert.equal(e.elements['roi-tools'].hidden,false);
  assert.equal(e.elements['esp-target'].hidden,true);
  e.input('roi-left',140);assert.deepEqual(bounds(e),[140,202,51,112]);
  assert.equal(e.elements['roi-line-left'].getAttribute('x1'),'140');
  assert.equal(e.elements.start.disabled,true);assert.equal(e.elements['direct-start'].disabled,true);
  await e.context.control('start');assert.equal(e.posts('/control').length,0);
  await e.context.updateStatus();assert.deepEqual(bounds(e),[140,202,51,112]);
  assert.match(e.elements['roi-status'].textContent,/尚未保存/);assert.equal(e.posts('/lamp-config').length,0);
  await replaceStatus(e,s=>{s.readiness.start_light.lamp_configuration={manual:true,roi:[150,60,50,50]}});
  assert.deepEqual(bounds(e),[140,202,51,112],'a newer saved status must preserve unsaved local edits');
  e.click('roi-cancel');assert.equal(e.elements['roi-tools'].hidden,true);
  assert.equal(e.elements.start.disabled,false);assert.equal(e.elements['esp-target'].hidden,false);
  assert.deepEqual(bounds(e),[150,200,60,110]);assert.equal(e.posts('/lamp-config').length,0);
 }
 {
  const e=setup();await flush();
  await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{lamp_configuration:{manual:false,roi:[139,51,63,61]},target_locked:true,target_roi:[64,60,60,52]}));
  loadedESP(e);e.context.beginROIEdit();assert.deepEqual(bounds(e),[64,124,60,112]);
  e.context.setROIBound('left',Infinity);assert.deepEqual(bounds(e),[64,124,60,112]);
  e.context.setROIBound('right',30);assert.equal(bounds(e)[1]-bounds(e)[0],16);
  e.context.setROIBound('bottom',999);assert.equal(bounds(e)[3],240);
  e.context.setROIBound('top',999);assert.equal(bounds(e)[3]-bounds(e)[2],16);
  e.context.setROIBound('left',-999);assert.equal(bounds(e)[0],0);
  assert.equal(e.posts('/lamp-config').length,0);assertStopUsable(e);
 }
 {
  const e=setup();await flush();loadedESP(e);e.click('roi-edit');
  const layer=e.elements['roi-layer'];layer.getBoundingClientRect=()=>({left:10,top:20,width:640,height:480});
  layer.events.pointerdown({target:e.elements['roi-hit-left'],pointerId:1});
  layer.events.pointermove({clientX:260,clientY:20,pointerId:1});
  assert.equal(bounds(e)[0],125,'drag coordinates scale to the sensor rectangle');
  layer.events.pointerup();layer.events.pointermove({clientX:300,clientY:20,pointerId:1});
  assert.equal(bounds(e)[0],125);assert.equal(e.posts('/lamp-config').length,0);
 }
 for(const age of [100,2500]){
  const e=setup();await flush();
  await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},target_locked:false,target_roi:[100,40,100,80],
   candidate_hint:{valid:true,roi:[80,48,48,40],age_ms:age}}));
  loadedESP(e);assert.equal(e.elements['esp-target'].hidden,false);
  assert.equal(e.elements['esp-target'].style.left,80/320*100+'%');
  e.click('roi-edit');assert.deepEqual(bounds(e),[80,128,48,88]);
  assert.equal(e.posts('/lamp-config').length,0,'candidate hints never save a manual box');
 }
 {
  const e=setup();await flush();
  await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{
   target_locked:true,target_roi:[64,60,60,52],candidate_hint:{valid:true,roi:[80,48,48,40],age_ms:100}}));
  loadedESP(e);assert.equal(e.elements['esp-target'].style.left,139/320*100+'%');
  e.click('roi-edit');assert.deepEqual(bounds(e),[139,202,51,112],'saved manual ROI takes priority over model output');
 }
 {
  const e=setup();await flush();
  await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},target_locked:true,target_roi:[64,60,60,52],
   candidate_hint:{valid:true,roi:[80,48,48,40],age_ms:100}}));
  loadedESP(e);assert.equal(e.elements['esp-target'].style.left,64/320*100+'%');
  e.click('roi-edit');assert.deepEqual(bounds(e),[64,124,60,112],'confirmed face takes priority over a provisional candidate');
 }
 const rejectedHints=[
  {valid:false,roi:[80,48,48,40],age_ms:100},
  {valid:true,roi:[80,48,48,40],age_ms:2501},
  {valid:true,roi:[80,48,48,40],age_ms:-1},
  {valid:true,roi:[80,48,48,40],age_ms:'100'},
  {valid:true,roi:[80,48,48,40],age_ms:100.5},
  {valid:true,roi:[300,48,48,40],age_ms:100},
  {valid:true,roi:[80,48,0,40],age_ms:100},
  {valid:true,roi:[80,48,48],age_ms:100},
  {valid:true,roi:['80',48,48,40],age_ms:100},
 ];
 for(const candidate of rejectedHints){
  const e=setup();await flush();
  await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},target_locked:false,candidate_hint:candidate}));
  loadedESP(e);assert.equal(e.elements['esp-target'].hidden,true,JSON.stringify(candidate));
  assert.equal(e.elements.start.disabled,false,'a rejected optional hint is not a start-signal failure');
  await replaceStatus(e,s=>{s.active=true;s.reason='FOLLOW';Object.assign(s.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},permission:true,phase:'RUNNING',connected:false,target_locked:false,candidate_hint:candidate})});
  assert.match(e.elements['mission-state'].textContent,/行驶中/);
  assert.match(e.elements['start-light-status'].textContent,/许可已锁存/);
  assert.doesNotMatch(e.elements.status.textContent,/显示异常/);assertStopUsable(e);
 }
 {
  const e=setup();await flush();loadedESP(e);e.elements['target-laps'].value='4';
  await e.context.updateStatus();assert.equal(e.elements['target-laps'].value,'4');
  await e.click('start');
  assert.deepEqual(e.posts('/control').map(request=>request.body),[{action:'start',generation:0,laps:4}]);
  assert.equal(e.posts('/lamp-config').length,0,'ARM must not silently save a lamp draft');
 }
 const lockedStates={
  active:s=>{s.active=true;s.reason='FOLLOW';s.readiness.arm_start_light_ready=false},
  armed:s=>{s.readiness.start_light.armed=true;s.readiness.start_light.phase='WAIT_ON';s.readiness.arm_start_light_ready=false},
  stale:s=>{s.age_ms=1000},
  pending:s=>{s.readiness.start_light.configuration_update={pending:true,state:'SENDING',error:'',request_id:4}},
 };
 for(const [name,mutate] of Object.entries(lockedStates)){
  const e=setup();await flush();loadedESP(e);e.context.beginROIEdit();e.context.setROIBound('left',140);
  await replaceStatus(e,mutate);
  for(const id of ['roi-save','roi-auto','roi-edit',...roiEdges.map(edge=>'roi-'+edge)])assert.equal(e.elements[id].disabled,true,name+': '+id);
  assert.equal(e.elements.start.disabled,true,name+': ARM');assertStopUsable(e);
  const before=e.posts().length;
  e.context.setROIBound('left',141);assert.equal(bounds(e)[0],140,name+': locked draft');
  await e.context.saveLampConfiguration(true);await e.context.saveLampConfiguration(false);await e.context.control('start');
  assert.equal(e.posts().length,before,name+': no forbidden mutation');
 }
 {
  const e=setup();await flush();loadedESP(e);e.delay(2001);await e.context.updateStatus();
  assert.equal(e.elements['roi-edit'].disabled,true);e.context.beginROIEdit();assert.equal(e.elements['roi-tools'].hidden,true);
  loadedESP(e);assert.equal(e.elements['roi-edit'].disabled,false);
  await replaceStatus(e,s=>{s.readiness.start_light.connected=false});
  assert.equal(e.elements['roi-edit'].disabled,true);assertStopUsable(e);
 }
 {
  const e=setup();await flush();loadedESP(e);e.context.beginROIEdit();e.context.setROIBound('left',140);
  const server=e.defaultData();let release;
  e.fetcher=async url=>url==='/lamp-config'?new Promise(resolve=>release=resolve):server;
  const saving=e.click('roi-save');await flush();
  assert.deepEqual(e.posts('/lamp-config').map(request=>request.body),[{generation:0,configuration:{manual:true,roi:[140,51,62,61]}}]);
  for(const id of ['start','direct-start','roi-save','roi-auto','roi-edit','roi-cancel'])assert.equal(e.elements[id].disabled,true,id+': save in flight');
  assertStopUsable(e);await e.context.control('start');assert.equal(e.posts('/control').length,0);
  release(e.response({queued:true,request_id:1},202));await saving;
  assert.equal(e.elements.start.disabled,true,'HTTP acceptance alone must not acknowledge the ESP setting');
  server.readiness.start_light.configuration_update={pending:true,state:'SENDING',error:'',request_id:1};await e.context.updateStatus();
  assert.equal(e.elements.start.disabled,true);
  server.readiness.start_light.lamp_configuration={manual:true,roi:[140,51,62,61]};
  server.readiness.start_light.configuration_update={pending:false,state:'SAVED',error:'',request_id:1};await e.context.updateStatus();
  assert.equal(e.elements.start.disabled,false);assert.equal(e.elements['roi-tools'].hidden,true);
  assert.equal(e.elements['esp-target'].hidden,false);assert.equal(e.elements['esp-target'].style.left,140/320*100+'%');
 }
 {
  const e=setup();await flush();loadedESP(e);e.context.beginROIEdit();e.context.setROIBound('left',140);
  e.fetcher=async url=>url==='/lamp-config'?e.response({message:'先停车'},409):e.defaultData();
  await e.context.saveLampConfiguration(true);
  assert.match(e.elements['roi-status'].textContent,/未保存.*先停车/);
  assert.equal(e.elements.start.disabled,true);assert.deepEqual(bounds(e),[140,202,51,112]);
  assert.equal(e.elements['roi-save'].disabled,false,'rejected save remains retryable');assertStopUsable(e);
 }
 for(const requestId of [undefined,0,'1']){
  const e=setup();await flush();loadedESP(e);e.context.beginROIEdit();e.context.setROIBound('left',140);
  e.fetcher=async url=>url==='/lamp-config'?e.response({queued:true,request_id:requestId},202):e.defaultData();
  await e.context.saveLampConfiguration(true);
  assert.equal(e.elements.start.disabled,true);assert.equal(e.elements['roi-save'].disabled,false);
  assert.match(e.elements['roi-status'].textContent,/未保存|无效/);assert.deepEqual(bounds(e),[140,202,51,112]);
 }
 {
  const e=setup();await flush();loadedESP(e);e.context.beginROIEdit();
  const server=e.defaultData();
  e.fetcher=async url=>url==='/lamp-config'?e.response({queued:true,request_id:5},202):server;
  await e.context.saveLampConfiguration(true);
  assert.equal(e.elements.start.disabled,true,'identical old configuration is not acknowledgement of a new save');
  server.readiness.start_light.configuration_update={pending:false,state:'SAVED',error:'',request_id:4};await e.context.updateStatus();
  assert.equal(e.elements.start.disabled,true,'an older completed request cannot unlock the new save');
  server.readiness.start_light.configuration_update={pending:false,state:'ERROR',error:'old_save_failed',request_id:4};await e.context.updateStatus();
  assert.equal(e.elements.start.disabled,true);
  assert.doesNotMatch(e.elements['roi-status'].textContent,/old_save_failed/,'an old failure is not the outcome of the pending request');
  server.readiness.start_light.configuration_update={pending:false,state:'SAVED',error:'',request_id:5};await e.context.updateStatus();
  assert.equal(e.elements.start.disabled,false);assert.equal(e.elements['roi-tools'].hidden,true);
 }
 {
  const e=setup();await flush();loadedESP(e);e.context.beginROIEdit();
  const server=e.defaultData();server.readiness.start_light.configuration_update={pending:true,state:'SENDING',error:'',request_id:2};
  e.fetcher=async url=>url==='/lamp-config'?e.response({queued:true,request_id:2},202):server;
  await e.click('roi-auto');
  assert.deepEqual(e.posts('/lamp-config').map(request=>request.body),[{generation:0,configuration:{manual:false}}]);
  assert.equal(e.elements.start.disabled,true);
  server.readiness.start_light.configuration_update={pending:false,state:'SAVED',error:'',request_id:2};
  server.readiness.start_light.lamp_configuration={manual:false,roi:[139,51,63,61]};await e.context.updateStatus();
  assert.equal(e.elements.start.disabled,false);assert.equal(e.elements['roi-tools'].hidden,true);
  assert.match(e.elements['lamp-mode'].textContent,/自动/);assert.equal(e.elements['esp-target'].hidden,true);
 }
 {
  const e=setup();await flush();loadedESP(e);e.context.beginROIEdit();e.context.setROIBound('left',140);
  e.context.navigator.onLine=false;e.windowEvents.offline();assertStopUsable(e);
  await e.context.control('stop');assert.equal(e.posts('/control').at(-1).body.action,'stop');
 }
 {
  const e=setup();await flush();
  await replaceStatus(e,s=>{s.active=true;s.reason='FOLLOW';Object.assign(s.readiness.start_light,{permission:true,phase:'RUNNING',connected:false})});
  assert.match(e.elements['mission-state'].textContent,/行驶中/);
  e.fetcher=async()=>{throw Error('network unavailable')};await e.context.updateStatus();
  assert.match(e.elements['mission-state'].textContent,/未知|中断|待确认/);
  assert.match(e.elements['road-state'].textContent,/未知/);assert.match(e.elements['motor-state'].textContent,/未知/);assertStopUsable(e);
 }
 console.log('field console layout, lamp drafts, lock matrix, async save and STOP scenarios passed');
}

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
 await fieldScenarios();
 console.log('13 preview connection scenarios passed');
}
main().catch(e=>{console.error(e);process.exitCode=1});
