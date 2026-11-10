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
 function node(tag,attrs,namespaceURI='http://www.w3.org/1999/xhtml'){
  const events={},e={tagName:tag.toUpperCase(),namespaceURI,attributes:attrs,children:[],parentElement:null,style:{},dataset:{},events,_text:null,_ownText:'',naturalWidth:320,naturalHeight:240};
  // hidden is an HTMLElement reflection, not an SVGElement reflection.
  // Mirroring it for SVG used to hide the real-page [hidden] regression.
  const reflected=namespaceURI==='http://www.w3.org/1999/xhtml'?{hidden:'hidden',disabled:'disabled',open:'open',checked:'checked'}:{};
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
  e.toggleAttribute=(key,force)=>{const present=force===undefined?!e.hasAttribute(key):!!force;present?e.setAttribute(key,''):e.removeAttribute(key);return present};
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
  const namespaceURI=tag==='svg'||stack.at(-1)?.namespaceURI==='http://www.w3.org/2000/svg'?'http://www.w3.org/2000/svg':'http://www.w3.org/1999/xhtml';
  const element=node(tag,attrs,namespaceURI);if(stack.length){element.parentElement=stack.at(-1);element.parentElement.children.push(element)}else roots.push(element);
  if(!voidTags.has(tag)&&!token[0].endsWith('/>'))stack.push(element);
 }
 return {elements,all,roots,node,matches};
}
function setup(options={}){
 let now=1000000,nextId=0;const timers=new Map(),docEvents={},windowEvents={};
 const dom=makeDOM(html),elements=dom.elements;
 const data=()=>({generation:0,age_ms:0,active:false,ready:true,reason:'TEST',left:0,right:0,preparation:{pending:false,phase:'IDLE',error:null,request_id:0},readiness:{arm_start_light_ready:true,direct_start_ready:true,laps:{enabled:false,total_laps:2},start_light:{enabled:true,armed:false,connected:true,phase:'DISARMED',initial_lamp_state:'OFF',peer_url:'http://10.0.0.2/',search_roi:[96,8,128,112],target_locked:false,required_frames:4,lamp_configuration:{manual:true,roi:[139,51,63,61]},configuration_update:{pending:false,state:'IDLE',error:'',request_id:0}}}});
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
 env.change=(id,value)=>{elements[id].value=String(value);return elements[id].events.change?.({target:elements[id]})};
 env.advance=ms=>{const until=now+ms;for(let i=0;i<100;i++){let chosen;for(const [id,t] of timers)if(t.at<=until&&(!chosen||t.at<chosen[1].at))chosen=[id,t];if(!chosen)break;now=chosen[1].at;timers.delete(chosen[0]);chosen[1].fn();}now=until};
 env.delay=ms=>now+=ms;return env;
}

const flush=async()=>{for(let i=0;i<8;i++)await Promise.resolve()};
const roiEdges=['left','right','top','bottom'];
const bounds=e=>roiEdges.map(edge=>Number(e.elements['roi-line-'+edge].getAttribute(edge==='left'||edge==='right'?'x1':'y1')));
function visible(element){
 for(let current=element;current;current=current.parentElement)if(current.hasAttribute('hidden')||current.style.display==='none'||current.style.visibility==='hidden')return false;
 return true;
}
function cssRule(selector){
 const rules=[...html.matchAll(/([^{}]+)\{([^{}]*)\}/g)];
 const declarations={};
 for(const [,selectors,body] of rules)if(selectors.split(',').some(s=>s.trim()===selector))for(const part of body.split(';')){const i=part.indexOf(':');if(i>0)declarations[part.slice(0,i).trim()]=part.slice(i+1).trim()}
 return declarations;
}
function assertVisibleGuides(e){
 const layer=e.elements['roi-layer'];assert.equal(layer.namespaceURI,'http://www.w3.org/2000/svg');
 assert.equal(visible(layer),true,'SVG hidden attribute must actually be removed; assigning .hidden does not reflect it');
 for(const edge of roiEdges){
  const line=e.elements['roi-line-'+edge],hit=e.elements['roi-hit-'+edge];
  assert.ok(layer.contains(line)&&layer.contains(hit));assert.equal(visible(line),true);
  for(const attr of ['x1','x2','y1','y2']){assert.notEqual(line.getAttribute(attr),null,edge+': missing '+attr);assert.ok(Number.isFinite(Number(line.getAttribute(attr))),edge+': '+attr)}
  assert.notDeepEqual([line.getAttribute('x1'),line.getAttribute('y1')],[line.getAttribute('x2'),line.getAttribute('y2')],edge+': line has nonzero length');
 }
}
function pointer(e,kind,edge,coords={}){
 const event={target:edge?e.elements['roi-hit-'+edge]:e.elements['roi-shade'],pointerId:1,isPrimary:true,buttons:kind==='pointerup'?0:1,clientX:0,clientY:0,preventDefault(){},stopPropagation(){},...coords};
 e.elements['roi-layer'].events[kind](event);
}
function keyBound(e,edge,key,shiftKey=false){
 let prevented=false;e.elements['roi-layer'].events.keydown({target:e.elements['roi-hit-'+edge],key,shiftKey,preventDefault(){prevented=true}});return prevented;
}
function dragBound(e,edge,value){
 pointer(e,'pointerdown',edge);
 pointer(e,'pointermove',edge,{clientX:edge==='left'||edge==='right'?value:0,clientY:edge==='top'||edge==='bottom'?value:0});
 pointer(e,'pointerup',edge);
}
const loadedESP=e=>{assert.equal(typeof e.elements['esp-video'].onload,'function');e.elements['esp-video'].onload()};
function assertStopUsable(e){
 const stop=e.elements.stop;assert.equal(stop.hidden,false);assert.equal(stop.disabled,false);
 assert.equal(stop.closest('details'),null);assert.ok(stop.closest('.mission-bar'));
}
async function replaceStatus(e,mutate){
 e.fetcher=async()=>{const status=e.defaultData();mutate(status);return status};
 await e.context.updateStatus();
}
function assertFrozenROI(e,expected){
 assert.deepEqual(bounds(e),expected,'the displayed ARM snapshot must remain fixed');
 for(const id of ['roi-auto','roi-cancel'])assert.equal(e.elements[id].disabled,true,id+': no edit while the ARM snapshot is frozen');
 assert.equal(e.elements['roi-layer'].getAttribute('data-editable'),'false');
 dragBound(e,'left',expected[0]+1);keyBound(e,'left','ArrowRight');
 e.context.setROIBound('left',expected[0]+1);e.context.selectAutomaticROI();e.context.cancelROIEdit();
 assert.deepEqual(bounds(e),expected,'pointer, keyboard, and direct handlers must all enforce the frozen snapshot');
}
async function fieldScenarios(){
 {
  const e=setup();await flush();
  assert.equal(e.elements['video-window'].dataset.view,'annotated');
  assert.equal(e.elements['camera-grid'].dataset.view,'annotated');assert.equal(e.elements.labels.hidden,true);
  const selected=e.context.document.querySelectorAll('.view-controls button.selected');
  assert.equal(selected.length,1);assert.equal(selected[0].dataset.view,'annotated');
  for(const name of ['diagnostics-panel','advanced-panel']){assert.equal(e.elements[name].tagName,'DETAILS');assert.equal(e.elements[name].open,false)}
  assert.equal(e.elements['direct-start'].closest('details').id,'advanced-panel');
  assert.equal(e.elements.start.closest('details'),null);assert.equal(e.elements['target-laps'].closest('details'),null);
  assert.equal(e.dom.all.filter(element=>element.tagName==='BUTTON'&&element.getAttribute('onclick')==="control('stop')").length,1);
  assert.equal(e.context.document.querySelectorAll('input[type="range"]').length,0,'drag on the image; no duplicate slider controls');
  assert.equal(e.elements['roi-save'],undefined,'ARM is the single save-and-prepare action');
  assert.equal(e.elements['roi-edit'],undefined,'visible image guides do not need a separate edit mode');assertStopUsable(e);
  const legacy=setup({storage:{'curve-camera-view':'all'}});await flush();
  assert.equal(legacy.elements['video-window'].dataset.view,'annotated');
  const remembered=setup({storage:{'race-console-view-v1':'ground'}});await flush();
  assert.equal(remembered.elements['video-window'].dataset.view,'ground');
 }
 {
  // Assert actual SVG attributes/CSS and overlay placement, not just synthetic properties.
  const e=setup();await flush();const layer=e.elements['roi-layer'],image=e.elements['esp-video'];
  assert.equal(layer.closest('.esp-window'),image.closest('.esp-window'));
  assert.ok(layer.parentElement.children.indexOf(layer)>layer.parentElement.children.indexOf(image),'guides paint above the image');
  assert.equal(cssRule('.esp-window').position,'relative');
  assert.equal(cssRule('.roi-layer').position,'absolute');assert.equal(cssRule('.roi-layer')['pointer-events'],'none');
  assert.notEqual(cssRule('.roi-layer')['touch-action'],'none','blank image space must still permit page scrolling');
  const line=cssRule('.roi-edge'),hit=cssRule('.roi-hit');
  assert.ok(line.stroke&&!['none','transparent'].includes(line.stroke),'visible line stroke');
  assert.ok(Number.parseFloat(line['stroke-width'])>0,'visible line width');
  assert.equal(hit['pointer-events'],'stroke');assert.equal(hit['touch-action'],'none');assert.equal(cssRule('.roi-handle')['touch-action'],'none');assert.ok(Number.parseFloat(hit['stroke-width'])>=15,'touch hit target wider than painted line');
  for(const edge of roiEdges){const handle=e.elements['roi-handle-'+edge];assert.equal(handle.tagName,'CIRCLE');assert.equal(handle.dataset.edge,edge);assert.ok(layer.contains(handle));const hit=e.elements['roi-hit-'+edge];assert.equal(hit.getAttribute('role'),'slider');assert.equal(hit.getAttribute('tabindex'),'-1')}
  assert.equal(visible(layer),false,'no stale picture guides before an ESP image is loaded');
  pointer(e,'pointerdown','left');pointer(e,'pointermove','left',{clientX:140});assert.equal(e.posts().length,0);
  loadedESP(e);assertVisibleGuides(e);assert.deepEqual(bounds(e),[139,202,51,112]);
  assert.equal(e.elements.start.disabled,false,'showing the guides is not an unsaved edit');
 }
 {
  const e=setup();await flush();loadedESP(e);assertVisibleGuides(e);
  pointer(e,'pointerdown',null);pointer(e,'pointermove',null,{clientX:140});assert.deepEqual(bounds(e),[139,202,51,112]);
  dragBound(e,'left',140);assert.deepEqual(bounds(e),[140,202,51,112]);
  assert.match(e.elements['roi-status'].textContent,/未保存|准备|调整/);
  assert.equal(e.elements.start.disabled,false,'ARM saves a dirty draft atomically');
  assert.equal(e.posts('/lamp-config').length,0);assert.equal(e.posts('/control').length,0);
  await e.context.updateStatus();assert.deepEqual(bounds(e),[140,202,51,112]);
  await replaceStatus(e,s=>{s.readiness.start_light.lamp_configuration={manual:true,roi:[150,60,50,50]}});
  assert.deepEqual(bounds(e),[140,202,51,112],'polls preserve the unsaved drag');
  await e.click('roi-cancel');assert.deepEqual(bounds(e),[150,200,60,110]);
  assert.equal(e.posts().length,0,'undo restores the latest confirmed box without a request');
 }
 {
  const e=setup();await flush();loadedESP(e);const layer=e.elements['roi-layer'];
  layer.getBoundingClientRect=()=>({left:10,top:20,width:640,height:480});
  pointer(e,'pointerdown','left');pointer(e,'pointermove','left',{clientX:260,clientY:20});
  assert.equal(bounds(e)[0],125,'drag coordinates scale to the sensor rectangle');
  pointer(e,'pointerup','left');pointer(e,'pointermove','left',{clientX:300,clientY:20});assert.equal(bounds(e)[0],125);
  pointer(e,'pointerdown','left');pointer(e,'pointermove','left',{clientX:270,clientY:20});pointer(e,'pointercancel','left');
  pointer(e,'pointermove','left',{clientX:310,clientY:20});assert.equal(bounds(e)[0],130);
  pointer(e,'pointerdown','left');pointer(e,'lostpointercapture','left');pointer(e,'pointermove','left',{clientX:310,clientY:20});assert.equal(bounds(e)[0],130);
  assert.equal(e.posts().length,0);
 }
 {
  const e=setup();await flush();loadedESP(e);
  for(const edge of roiEdges){assert.equal(e.elements['roi-hit-'+edge].getAttribute('tabindex'),'0');assert.equal(e.elements['roi-hit-'+edge].getAttribute('aria-disabled'),'false')}
  assert.equal(keyBound(e,'left','ArrowRight'),true);assert.equal(bounds(e)[0],140);
  assert.equal(keyBound(e,'left','ArrowLeft',true),true);assert.equal(bounds(e)[0],135);
  assert.equal(keyBound(e,'right','ArrowLeft'),true);assert.equal(bounds(e)[1],201);
  assert.equal(keyBound(e,'right','ArrowRight',true),true);assert.equal(bounds(e)[1],206);
  assert.equal(keyBound(e,'top','ArrowDown'),true);assert.equal(bounds(e)[2],52);
  assert.equal(keyBound(e,'top','ArrowUp',true),true);assert.equal(bounds(e)[2],47);
  assert.equal(keyBound(e,'bottom','ArrowUp'),true);assert.equal(bounds(e)[3],111);
  assert.equal(keyBound(e,'bottom','ArrowDown',true),true);assert.equal(bounds(e)[3],116);
  const before=bounds(e);for(const key of ['ArrowUp','ArrowDown','Enter','Escape'])assert.equal(keyBound(e,'left',key),false);
  for(const key of ['ArrowLeft','ArrowRight','Enter','Escape'])assert.equal(keyBound(e,'top',key),false);
  assert.deepEqual(bounds(e),before);assert.equal(e.elements['roi-hit-left'].getAttribute('aria-valuenow'),'135');
  pointer(e,'pointerdown','left',{target:e.elements['roi-handle-left']});pointer(e,'pointermove','left',{clientX:136});pointer(e,'pointerup','left');
  assert.equal(bounds(e)[0],136,'visible grab handles use the same direct drag behavior');assert.equal(e.posts().length,0);
 }
 {
  const e=setup();await flush();loadedESP(e);
  dragBound(e,'left',999);assert.equal(bounds(e)[1]-bounds(e)[0],16);
  dragBound(e,'right',-999);assert.equal(bounds(e)[1]-bounds(e)[0],16);
  dragBound(e,'top',999);assert.equal(bounds(e)[3]-bounds(e)[2],16);
  dragBound(e,'bottom',999);assert.equal(bounds(e)[3],240);
  dragBound(e,'left',-999);assert.equal(bounds(e)[0],0);
  const before=bounds(e);e.context.setROIBound('left',Infinity);assert.deepEqual(bounds(e),before);
  const layer=e.elements['roi-layer'];layer.getBoundingClientRect=()=>({left:0,top:0,width:0,height:0});
  dragBound(e,'left',100);assert.deepEqual(bounds(e),before);assert.equal(e.posts().length,0);
 }
 for(const age of [100,2500]){
  const e=setup();await flush();
  await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},target_locked:false,target_roi:[100,40,100,80],
   candidate_hint:{valid:true,roi:[80,48,48,40],age_ms:age}}));
  loadedESP(e);assertVisibleGuides(e);assert.deepEqual(bounds(e),[80,128,48,88]);
  dragBound(e,'left',82);assert.deepEqual(bounds(e),[82,128,48,88]);assert.equal(e.posts().length,0);
 }
 {
  const e=setup();await flush();
  await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{
   target_locked:true,target_roi:[64,60,60,52],candidate_hint:{valid:true,roi:[80,48,48,40],age_ms:100}}));
  loadedESP(e);assert.deepEqual(bounds(e),[139,202,51,112],'saved manual wins over all automatic hints');
 }
 {
  const e=setup();await flush();
  await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},target_locked:true,target_roi:[64,60,60,52],
   candidate_hint:{valid:true,roi:[80,48,48,40],age_ms:100}}));
  loadedESP(e);assert.deepEqual(bounds(e),[64,124,60,112],'confirmed face wins over candidate');
 }
 const rejectedHints=[
  {valid:false,roi:[80,48,48,40],age_ms:100},{valid:true,roi:[80,48,48,40],age_ms:2501},
  {valid:true,roi:[80,48,48,40],age_ms:-1},{valid:true,roi:[80,48,48,40],age_ms:'100'},
  {valid:true,roi:[80,48,48,40],age_ms:100.5},{valid:true,roi:[300,48,48,40],age_ms:100},
  {valid:true,roi:[80,48,0,40],age_ms:100},{valid:true,roi:[80,48,48],age_ms:100},
  {valid:true,roi:['80',48,48,40],age_ms:100},
 ];
 for(const candidate of rejectedHints){
  const e=setup();await flush();
  await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},target_locked:false,candidate_hint:candidate}));
  loadedESP(e);assertVisibleGuides(e);assert.equal(e.elements.start.disabled,true,'automatic mode requires a fresh valid candidate before ARM');
  for(const value of bounds(e))assert.ok(Number.isFinite(value),'an invalid candidate never creates invalid guide coordinates');
  await replaceStatus(e,s=>{s.active=true;s.reason='FOLLOW';Object.assign(s.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},permission:true,phase:'RUNNING',connected:false,candidate_hint:candidate})});
  assert.match(e.elements['mission-state'].textContent,/行驶中/);assert.match(e.elements['start-light-status'].textContent,/许可已锁存/);
  assert.doesNotMatch(e.elements.status.textContent,/显示异常/);assertStopUsable(e);
 }
 {
  const e=setup();await flush();loadedESP(e);e.elements['target-laps'].value='4';dragBound(e,'left',140);
  const server=e.defaultData();server.preparation={pending:true,phase:'SAVING',error:null,request_id:8};
  e.fetcher=async url=>url==='/control'?e.response({message:'准备中',request_id:8},202):server;
  await e.click('start');
  assert.deepEqual(e.posts('/control').map(request=>request.body),[{action:'start',generation:0,laps:4,initial_lamp_state:'OFF',lamp_configuration:{manual:true,roi:[140,51,62,61]}}]);
  assert.equal(e.posts('/lamp-config').length,0,'ARM has no separate or implicit second save request');
  assert.equal(e.elements.start.disabled,true);assertStopUsable(e);
  // An unrelated old configuration acknowledgement cannot complete preparation.
  server.readiness.start_light.configuration_update={pending:false,state:'SAVED',error:null,request_id:7};await e.context.updateStatus();
  assert.equal(e.elements.start.disabled,true);assert.equal(e.posts('/control').length,1);
 }
 {
  const e=setup();await flush();loadedESP(e);dragBound(e,'left',140);
  const server=e.defaultData();server.preparation={pending:true,phase:'SAVING',error:null,request_id:8};
  e.fetcher=async url=>url==='/control'?e.response({message:'准备中',request_id:8},202):server;await e.click('start');
  server.preparation={pending:false,phase:'ARMED',error:null,request_id:8};server.readiness.arm_start_light_ready=false;server.readiness.direct_start_ready=false;
  Object.assign(server.readiness.start_light,{armed:true,phase:'WAIT_ON'});await e.context.updateStatus();
  assert.equal(e.evaluate('roiDirty'),true,'armed without the selected configuration cannot confirm the new draft');
  server.readiness.start_light.lamp_configuration={manual:true,roi:[140,51,62,61]};server.generation=1;await e.context.updateStatus();
  assert.equal(e.evaluate('roiDirty'),true,'a different control generation cannot acknowledge a submitted draft');
  server.generation=0;await e.context.updateStatus();
  assert.equal(e.evaluate('roiDirty'),false,'same-generation armed status with matching config confirms preparation');
  assertVisibleGuides(e);assert.equal(e.elements['roi-layer'].getAttribute('data-editable'),'false');assert.equal(e.posts('/control').length,1);assertStopUsable(e);
 }
 {
  // The first successful poll can arrive after ARM has already become driving.
  const e=setup();await flush();loadedESP(e);dragBound(e,'left',140);
  const server=e.defaultData();server.active=true;server.reason='FOLLOW';
  server.readiness.arm_start_light_ready=false;server.readiness.direct_start_ready=false;
  Object.assign(server.readiness.start_light,{armed:false,permission:true,phase:'RUNNING',lamp_configuration:{manual:true,roi:[140,51,62,61]}});
  e.fetcher=async url=>url==='/control'?e.response({message:'准备请求已接收'},202):server;
  await e.click('start');
  assert.equal(e.evaluate('roiDirty'),false,'first poll already driving confirms the same-generation applied draft');
  assert.match(e.elements['mission-state'].textContent,/行驶中/);assert.equal(e.posts('/control').length,1);assert.equal(e.posts('/lamp-config').length,0);
 }
 {
  // A backgrounded tab can miss the entire armed/running interval.
  const e=setup();await flush();loadedESP(e);dragBound(e,'left',140);
  const server=e.defaultData();server.generation=1;server.reason='LAPS COMPLETE';
  server.preparation={pending:false,phase:'ARMED',error:null,request_id:8,generation:0};
  server.readiness.laps={enabled:true,total_laps:2,completed_laps:2,complete:true,phase:'COMPLETE'};
  server.readiness.start_light.lamp_configuration={manual:true,roi:[140,51,62,61]};
  e.fetcher=async url=>url==='/control'?e.response({message:'准备请求已接收'},202):server;
  await e.click('start');
  assert.equal(e.evaluate('roiDirty'),false,'captured preparation generation proves application after completion advanced the live generation');
  assert.match(e.elements['mission-state'].textContent,/完赛/);assert.equal(e.posts('/control').length,1);assert.equal(e.posts('/lamp-config').length,0);
 }
 {
  const e=setup();await flush();loadedESP(e);dragBound(e,'left',140);
  const server=e.defaultData();server.generation=1;server.reason='LAPS COMPLETE';
  server.preparation={pending:false,phase:'ARMED',error:null,request_id:8,generation:99};
  server.readiness.start_light.lamp_configuration={manual:true,roi:[140,51,62,61]};
  e.fetcher=async url=>url==='/control'?e.response({message:'准备请求已接收'},202):server;
  await e.click('start');
  assert.equal(e.evaluate('roiDirty'),true,'old or unrelated preparation generation must never retire the draft');
  server.preparation.generation=0;server.readiness.start_light.lamp_configuration={manual:true,roi:[139,51,63,61]};await e.context.updateStatus();
  assert.equal(e.evaluate('roiDirty'),true,'matching preparation proof still requires the selected configuration');
 }
 {
  const e=setup();await flush();loadedESP(e);await e.click('start');
  assert.deepEqual(e.posts('/control').map(request=>request.body),[{action:'start',generation:0,laps:2,initial_lamp_state:'OFF',lamp_configuration:{manual:true,roi:[139,51,63,61]}}]);
  assert.equal(e.posts('/lamp-config').length,0,'confirmed manual face also uses the atomic preparation request');
 }
 {
  const e=setup();await flush();loadedESP(e);dragBound(e,'left',140);const original=bounds(e),server=e.defaultData();let release;
  e.fetcher=async url=>url==='/lamp-config'?new Promise(resolve=>release=resolve):server;
  const scanning=e.click('roi-auto');await flush();
  assert.deepEqual(e.posts('/lamp-config').map(r=>r.body),[{generation:0,configuration:{manual:false}}],'automatic assistance must enable the locator when the detector was manual');
  assert.equal(e.posts('/control').length,0,'enabling the locator never ARMs or starts the car');
  for(const id of ['start','direct-start','roi-auto','roi-cancel'])assert.equal(e.elements[id].disabled,true,id+': locator setup in flight');
  dragBound(e,'left',141);keyBound(e,'left','ArrowRight');assert.deepEqual(bounds(e),original);assertStopUsable(e);
  release(e.response({message:'定位模式已排队',request_id:9},202));await scanning;
  assert.equal(e.elements.start.disabled,true,'202 is acceptance, not proof that automatic mode is active');
  server.readiness.start_light.configuration_update={pending:false,state:'SAVED',error:null,request_id:8};await e.context.updateStatus();
  assert.equal(e.elements['roi-auto'].disabled,true,'an older configuration acknowledgement cannot finish locator setup');
  server.readiness.start_light.configuration_update={pending:false,state:'ERROR',error:'old_save_failed',request_id:8};await e.context.updateStatus();
  assert.equal(e.elements['roi-auto'].disabled,true);assert.doesNotMatch(e.elements['roi-status'].textContent,/old_save_failed/);
  server.readiness.start_light.lamp_configuration={manual:false,roi:[139,51,63,61]};
  server.readiness.start_light.candidate_hint={valid:true,roi:[80,48,48,40],age_ms:100};
  server.readiness.start_light.configuration_update={pending:false,state:'SAVED',error:null,request_id:9};await e.context.updateStatus();
  assert.equal(e.elements['roi-auto'].disabled,false);assert.deepEqual(bounds(e),[80,128,48,88]);assert.equal(e.elements.start.disabled,false);
  e.fetcher=async url=>url==='/control'?e.response({message:'准备请求已接收'},202):server;
  await e.click('start');assert.deepEqual(e.posts('/control')[0].body.lamp_configuration,{manual:true,roi:[80,48,48,40]});
  assert.equal(e.posts('/lamp-config').length,1,'ARM does not issue another auxiliary save');
 }
 for(const requestId of [undefined,0,'9']){
  const e=setup();await flush();loadedESP(e);const server=e.defaultData();
  e.fetcher=async url=>url==='/lamp-config'?e.response({message:'配置已排队',request_id:requestId},202):server;
  await e.click('roi-auto');assert.equal(e.elements['roi-auto'].disabled,false,'invalid scan receipt must allow a deliberate retry');
  assert.match(e.elements['roi-status'].textContent,/回执无效/);assert.equal(e.posts('/control').length,0);assertStopUsable(e);
 }
 {
  const e=setup();await flush();loadedESP(e);const server=e.defaultData();
  e.fetcher=async url=>url==='/lamp-config'?e.response({message:'配置已排队',request_id:9},202):server;
  await e.click('roi-auto');assert.equal(e.elements['roi-auto'].disabled,true);
  e.delay(5001);await e.context.updateStatus();loadedESP(e);
  assert.equal(e.elements['roi-auto'].disabled,false,'a missing matching ESP receipt must time out instead of locking the console forever');
  assert.match(e.elements['roi-status'].textContent,/超时.*重试/);assert.equal(e.posts('/control').length,0);assertStopUsable(e);
 }
 {
  const e=setup();await flush();const server=e.defaultData();Object.assign(server.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},candidate_hint:{valid:true,roi:[80,48,48,40],age_ms:100}});
  e.fetcher=async()=>server;await e.context.updateStatus();loadedESP(e);dragBound(e,'left',82);await e.click('roi-auto');
  assert.equal(e.posts().length,0,'an already automatic locator needs no mode-change request');assert.deepEqual(bounds(e),[80,128,48,88]);
  await e.context.updateStatus();assert.deepEqual(bounds(e),[80,128,48,88]);
 }
 {
  const e=setup();await flush();const server=e.defaultData();server.readiness.start_light.lamp_configuration={manual:false,roi:[139,51,63,61]};
  e.fetcher=async()=>server;await e.context.updateStatus();loadedESP(e);assert.equal(e.elements.start.disabled,true);
  await e.context.control('start');assert.equal(e.posts('/control').length,0,'fallback coordinates alone are not evidence of automatic localization');
  dragBound(e,'left',140);assert.equal(e.elements.start.disabled,false,'a manually chosen range can be prepared without a model candidate');
  await e.click('start');assert.deepEqual(e.posts('/control')[0].body.lamp_configuration,{manual:true,roi:[140,51,62,61]});
 }
 {
  const e=setup();await flush();loadedESP(e);const server=e.defaultData();let release;
  e.fetcher=async(url,options)=>url==='/lamp-config'?new Promise(resolve=>release=resolve):server;
  const scanning=e.click('roi-auto');await flush();assert.equal(e.posts('/lamp-config').length,1);
  server.generation=1;await e.click('stop');assert.equal(e.posts('/control').at(-1).body.action,'stop');
  release(e.response({message:'旧辅助定位请求',request_id:9},202));await scanning;await e.context.updateStatus();
  assert.equal(e.elements['roi-auto'].disabled,false,'STOP cancels the local locator request even if its HTTP result arrives late');
  assert.equal(e.posts('/lamp-config').length,1);assert.equal(e.posts('/control').filter(r=>r.body.action==='start').length,0);
 }
 {
  const e=setup();await flush();
  const server=e.defaultData();Object.assign(server.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},candidate_hint:{valid:true,roi:[80,48,48,40],age_ms:100}});
  let release;
  e.fetcher=async(url,options)=>url==='/control'&&JSON.parse(options.body).action==='start'?new Promise(resolve=>release=resolve):server;
  await e.context.updateStatus();loadedESP(e);const original=bounds(e);assert.deepEqual(original,[80,128,48,88]);
  const starting=e.click('start');await flush();assertFrozenROI(e,original);
  assert.deepEqual(e.posts('/control')[0].body.lamp_configuration,{manual:true,roi:[80,48,48,40]},'an automatic suggestion becomes the exact detector box on ARM');
  // Even a fresh automatic suggestion cannot move the snapshot before the POST returns.
  server.readiness.start_light.candidate_hint={valid:true,roi:[170,60,40,40],age_ms:100};await e.context.updateStatus();assertFrozenROI(e,original);
  server.preparation={pending:true,phase:'SAVING',error:null,request_id:8,generation:0};
  release(e.response({message:'准备请求已接收'},202));await starting;assertFrozenROI(e,original);
  server.age_ms=1000;server.generation=1;await e.context.updateStatus();assertFrozenROI(e,original);
  server.age_ms=0;server.generation=0;server.readiness.start_light.candidate_hint={valid:true,roi:[180,70,40,40],age_ms:100};
  await e.context.updateStatus();assertFrozenROI(e,original); // Expired generation evidence cannot unlock the ARM box.
  Object.assign(server.readiness.start_light,{armed:true,phase:'WAIT_ON',lamp_configuration:{manual:true,roi:[80,48,48,40]}});
  server.preparation={pending:false,phase:'ARMED',error:null,request_id:8,generation:0};server.readiness.arm_start_light_ready=false;server.readiness.direct_start_ready=false;
  await e.context.updateStatus();assertFrozenROI(e,original);
  server.active=true;server.reason='FOLLOW';Object.assign(server.readiness.start_light,{armed:false,permission:true,phase:'RUNNING',target_locked:true,target_roi:[190,70,40,40]});
  await e.context.updateStatus();assertFrozenROI(e,original);
  e.context.navigator.onLine=false;e.windowEvents.offline();assertFrozenROI(e,original);assertStopUsable(e);
  e.context.navigator.onLine=true;e.windowEvents.online();await flush();loadedESP(e);assertFrozenROI(e,original);
  assert.equal(e.posts('/control').length,1);assert.equal(e.posts('/lamp-config').length,0);
 }
 {
  const e=setup();await flush();
  const server=e.defaultData();Object.assign(server.readiness.start_light,{
   lamp_configuration:{manual:false,roi:[139,51,63,61]},candidate_hint:{valid:true,roi:[80,48,48,40],age_ms:100}});
  e.fetcher=async url=>url==='/control'?e.response({message:'准备请求已接收'},202):server;
  await e.context.updateStatus();loadedESP(e);const original=bounds(e);
  server.preparation={pending:false,phase:'ERROR',error:'configuration_offline',request_id:8,generation:0};
  await e.click('start');assertFrozenROI(e,original);assert.equal(e.elements.start.disabled,false,'an explicit ARM retry may reuse the frozen box');
  server.readiness.start_light.candidate_hint={valid:true,roi:[170,60,40,40],age_ms:100};await e.context.updateStatus();assertFrozenROI(e,original);
  await e.click('start');assert.equal(e.posts('/control').length,2);
  assert.deepEqual(e.posts('/control').map(r=>r.body.lamp_configuration),[{manual:true,roi:[80,48,48,40]},{manual:true,roi:[80,48,48,40]}]);
  server.generation=1;server.preparation={pending:false,phase:'CANCELLED',error:null,request_id:8,generation:0};
  await e.click('stop');assert.deepEqual(bounds(e),original,'STOP preserves the selected range for further adjustment');
  assert.equal(e.elements['roi-layer'].getAttribute('data-editable'),'true');
  dragBound(e,'left',82);assert.deepEqual(bounds(e),[82,128,48,88],'STOP releases the snapshot for direct adjustment');
  assert.equal(e.posts('/control').at(-1).body.action,'stop');assert.equal(e.posts('/lamp-config').length,0);
 }
 const lockedStates={
  active:s=>{s.active=true;s.reason='FOLLOW';s.readiness.arm_start_light_ready=false},
  armed:s=>{s.readiness.start_light.armed=true;s.readiness.start_light.phase='WAIT_ON';s.readiness.arm_start_light_ready=false},
  stale:s=>{s.age_ms=1000},
  remote_configuration:s=>{s.readiness.start_light.configuration_update={pending:true,state:'SENDING',error:null,request_id:4}},
  preparation:s=>{s.preparation={pending:true,phase:'SAVING',error:null,request_id:8}},
 };
 for(const [name,mutate] of Object.entries(lockedStates)){
  const e=setup();await flush();loadedESP(e);dragBound(e,'left',140);const before=bounds(e);
  await replaceStatus(e,mutate);
  for(const id of ['roi-auto','roi-cancel','start','direct-start','initial-lamp-state'])assert.equal(e.elements[id].disabled,true,name+': '+id);
  dragBound(e,'left',141);keyBound(e,'left','ArrowRight');assert.deepEqual(bounds(e),before,name+': no forbidden image or keyboard edit');
  for(const edge of roiEdges){assert.equal(e.elements['roi-hit-'+edge].getAttribute('tabindex'),'-1');assert.equal(e.elements['roi-hit-'+edge].getAttribute('aria-disabled'),'true')}
  await e.context.control('start');assert.equal(e.posts().length,0,name+': no forbidden request');assertStopUsable(e);
 }
 {
  const e=setup();await flush();loadedESP(e);const before=bounds(e);e.delay(2001);await e.context.updateStatus();
  dragBound(e,'left',140);assert.deepEqual(bounds(e),before,'stale image cannot be adjusted');
  loadedESP(e);dragBound(e,'left',140);assert.equal(bounds(e)[0],140);
  await replaceStatus(e,s=>{s.readiness.start_light.connected=false});dragBound(e,'left',141);assert.equal(bounds(e)[0],140);
  assert.equal(e.posts().length,0);assertStopUsable(e);
 }
 {
  const e=setup();await flush();loadedESP(e);dragBound(e,'left',140);let release;
  const server=e.defaultData();server.preparation={pending:true,phase:'SAVING',error:null,request_id:8};
  e.fetcher=async(url,options)=>url==='/control'&&JSON.parse(options.body).action==='start'?new Promise(resolve=>release=resolve):server;
  const starting=e.click('start');await flush();assert.equal(e.posts('/control').length,1);
  for(const id of ['start','direct-start','roi-auto','roi-cancel'])assert.equal(e.elements[id].disabled,true,id+': preparation HTTP in flight');
  const before=bounds(e);dragBound(e,'left',141);assert.deepEqual(bounds(e),before);assertStopUsable(e);
  server.generation=1;server.preparation={pending:false,phase:'CANCELLED',error:null,request_id:8};
  await e.click('stop');assert.equal(e.posts('/control').at(-1).body.action,'stop');
  release(e.response({message:'旧准备请求',request_id:8},202));await starting;await e.context.updateStatus();
  assert.equal(e.posts('/control').filter(r=>r.body.action==='start').length,1,'STOP must never retry an earlier ARM');
  assert.deepEqual(bounds(e),[140,202,51,112],'STOP and the stale response retain the unsaved selection');assert.match(e.elements['lamp-mode'].textContent,/ARM 时/);
  assert.equal(e.posts('/lamp-config').length,0);
 }
 {
  const e=setup();await flush();loadedESP(e);dragBound(e,'left',140);
  const server=e.defaultData();server.preparation={pending:false,phase:'ERROR',error:'configuration_offline',request_id:8};
  e.fetcher=async url=>url==='/control'?e.response({message:'ESP32 暂不可达'},409):server;
  await e.click('start');assert.deepEqual(bounds(e),[140,202,51,112],'failed preparation preserves the selected face for retry');
  assertFrozenROI(e,[140,202,51,112]);assert.equal(e.elements.start.disabled,false,'HTTP failure retains the snapshot but permits explicit same-box retry');
  assert.match(e.elements.notice.textContent+' '+e.elements['mission-state'].textContent+' '+e.elements['roi-status'].textContent,/不可达|失败|准备|重试/);
  assert.equal(e.posts('/control').length,1);assert.equal(e.posts('/lamp-config').length,0);assertStopUsable(e);
 }
 {
  const e=setup();await flush();loadedESP(e);dragBound(e,'left',140);e.context.navigator.onLine=false;e.windowEvents.offline();
  assertStopUsable(e);await e.context.control('stop');assert.equal(e.posts('/control').at(-1).body.action,'stop');
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>{s.active=true;s.reason='FOLLOW';Object.assign(s.readiness.start_light,{permission:true,phase:'RUNNING',connected:false})});
  assert.match(e.elements['mission-state'].textContent,/行驶中/);
  e.fetcher=async()=>{throw Error('network unavailable')};await e.context.updateStatus();
  assert.match(e.elements['mission-state'].textContent,/未知|中断|待确认/);assert.match(e.elements['road-state'].textContent,/未知/);assert.match(e.elements['motor-state'].textContent,/未知/);assertStopUsable(e);
 }
 console.log('direct image guides, atomic ARM, pending/STOP locks and field console scenarios passed');
}


function assertLampSignal(e,state,text){
 const badge=e.elements['lamp-signal-state'];assert.ok(badge,'signal state has a dedicated badge');
 assert.equal(badge.dataset.state,state);assert.match(badge.textContent,text);
 assert.equal(badge.hidden,false);assert.equal(badge.closest('details'),null);
}
async function lampSignalScenarios(){
 {
  const e=setup();await flush();assertLampSignal(e,'LOCATING',/定位中/);
  assert.equal(e.elements['lamp-signal-state'].closest('.card-header'),e.elements['esp-state'].closest('.card-header'));
  assert.notEqual(e.elements['lamp-signal-state'],e.elements['esp-state'],'signal progression and preview connectivity are separate');
  assert.doesNotMatch(e.elements['lamp-mode'].textContent,/已锁定/,'a saved manual rectangle is not an armed lock');
 }
 for(const phase of ['WAIT_TARGET','LOCK_TARGET']){
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{armed:true,permission:false,phase,target_locked:false}));
  assertLampSignal(e,'LOCATING',/定位中/);
  assert.doesNotMatch(e.elements['lamp-signal-state'].textContent,/灯\s*OFF|灯\s*ON|出发/,'acquisition phases are not proof of the lamp level');
 }
 for(const phase of ['WAIT_ON','ARMING']){
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{armed:true,permission:false,phase,target_locked:false,off_baseline_ready:true}));
  assertLampSignal(e,'LOCKED',/已锁定/);assert.match(e.elements['lamp-mode'].textContent,/已锁定/);
  assert.doesNotMatch(e.elements['lamp-signal-state'].textContent,/灯\s*OFF|灯\s*ON|出发/,'WAIT_ON is entered before the first frame and must not claim a verified OFF');
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>{s.preparation={pending:true,phase:'SAVING',error:null,request_id:9,generation:0}});
  assertLampSignal(e,'LOCKED',/已锁定/);assertStopUsable(e);
 }
 {
  const e=setup();await flush();loadedESP(e);dragBound(e,'left',140);let release;
  e.fetcher=async(url,options)=>url==='/control'&&JSON.parse(options.body).action==='start'?new Promise(resolve=>release=resolve):e.defaultData();
  const starting=e.click('start');await flush();assertLampSignal(e,'UNKNOWN',/未知|更新|确认/);
  release(e.response({message:'准备请求已接收'},202));await starting;
  assertLampSignal(e,'LOCKED',/已锁定/);assert.match(e.elements['lamp-mode'].textContent,/已锁定/);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{armed:true,permission:false,phase:'WAIT_OFF',target_locked:true,target_roi:[139,51,63,61]}));
  assertLampSignal(e,'ON',/灯\s*ON/);assert.match(e.elements['lamp-mode'].textContent,/已锁定/);
  loadedESP(e);assertLampSignal(e,'ON',/灯\s*ON/);
  e.elements['esp-video'].onerror();assertLampSignal(e,'ON',/灯\s*ON/);
  assert.match(e.elements['esp-state'].textContent,/重连/,'image failure only changes the connection badge');
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{armed:true,permission:false,phase:'CONFIRM_OFF',target_locked:true,target_roi:[139,51,63,61],confirmation_frames:1}));
  assertLampSignal(e,'OFF',/灯\s*OFF.*确认中/);assert.doesNotMatch(e.elements['lamp-signal-state'].textContent,/出发/);
  assert.match(e.elements['lamp-mode'].textContent,/已锁定/);
 }
 for(const phase of ['SIGNAL_READY','PERMITTED','RUNNING']){
  const e=setup();await flush();await replaceStatus(e,s=>{s.active=phase==='RUNNING';s.reason=s.active?'FOLLOW':'WAIT START LIGHT';Object.assign(s.readiness.start_light,{phase,permission:true,armed:!s.active,connected:false,error:'ESP32_OFFLINE'})});
  assertLampSignal(e,'START',/出发/);assertStopUsable(e);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>{s.active=true;s.reason='FOLLOW';Object.assign(s.readiness.start_light,{phase:'RUNNING',permission:false,armed:false,connected:true})});
  assertLampSignal(e,'MANUAL',/未等灯/);assert.doesNotMatch(e.elements['lamp-signal-state'].textContent,/出发/);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>{s.active=true;s.reason='FOLLOW';Object.assign(s.readiness.start_light,{
   phase:'RUNNING',permission:false,armed:false,connected:true,target_locked:false,lamp_configuration:{manual:false,roi:[139,51,63,61]},
   candidate_hint:{valid:true,roi:[80,48,48,40],age_ms:100}})});
  assertLampSignal(e,'MANUAL',/未等灯/);assert.doesNotMatch(e.elements['lamp-mode'].textContent,/已锁定/,'manual driving does not freeze an automatic locator');
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{
   phase:'WAIT_ON',permission:false,armed:true,connected:true,target_locked:false,lamp_configuration:{manual:false,roi:[139,51,63,61]}}));
  assertLampSignal(e,'LOCATING',/定位中/);assert.doesNotMatch(e.elements['lamp-mode'].textContent,/已锁定/,'a remote automatic ARM without a located face does not prove the geometry is fixed');
 }
 for(const permission of ['true',1]){
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{phase:'SIGNAL_READY',permission,armed:true,connected:true}));
  assert.notEqual(e.elements['lamp-signal-state'].dataset.state,'START','only a true boolean permission can report the start signal');
 }
 for(const phase of ['SIGNAL_READY','PERMITTED']){
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{phase,permission:false,armed:true,connected:true}));
  assert.notEqual(e.elements['lamp-signal-state'].dataset.state,'START','phase label alone never substitutes for the Maix permission latch');
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{phase:'TARGET_UNCERTAIN',permission:false,armed:true,connected:true}));
  assertLampSignal(e,'UNKNOWN',/待确认/);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{phase:'FAULT',permission:false,armed:false,connected:true,fault_reason:'RESPONSE_TOO_SLOW'}));
  assertLampSignal(e,'FAULT',/等灯中断/);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>{s.readiness.start_light.enabled=false});assertLampSignal(e,'DISABLED',/未启用/);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{phase:'WAIT_OFF',permission:false,armed:true,connected:false}));
  assertLampSignal(e,'UNKNOWN',/未知|中断|离线/);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>{s.active=true;Object.assign(s.readiness.start_light,{phase:'RUNNING',permission:true,connected:false})});
  assertLampSignal(e,'START',/出发/);
  e.delay(1000);e.context.syncControls();assertLampSignal(e,'UNKNOWN',/状态未知/);
  loadedESP(e);assertLampSignal(e,'UNKNOWN',/状态未知/,'a fresh image alone cannot freshen Maix signal telemetry');
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>{s.age_ms=1000;Object.assign(s.readiness.start_light,{phase:'RUNNING',permission:true,connected:true})});
  assertLampSignal(e,'UNKNOWN',/状态未知/);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{phase:'WAIT_OFF',armed:true,target_locked:true,target_roi:[139,51,63,61]}));
  e.fetcher=async()=>{throw Error('network unavailable')};await e.context.updateStatus();assertLampSignal(e,'UNKNOWN',/状态未知/);assertStopUsable(e);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>{s.active=true;Object.assign(s.readiness.start_light,{phase:'RUNNING',permission:true,connected:false})});
  assertLampSignal(e,'START',/出发/);let release;
  e.fetcher=async(url,options)=>url==='/control'&&JSON.parse(options.body).action==='stop'?new Promise(resolve=>release=resolve):e.defaultData();
  const stopping=e.click('stop');await flush();assertLampSignal(e,'UNKNOWN',/状态未知/);
  release(e.response({message:'停车请求已接收'},200));await stopping;assertLampSignal(e,'LOCATING',/定位中/);
 }
 console.log('lamp signal progress, independent lock state, permission and stale-state scenarios passed');
}


async function initialLampStateScenarios(){
 const id='initial-lamp-state';
 {
  const e=setup();await flush();const input=e.elements[id];
  assert.equal(input.tagName,'SELECT');assert.equal(input.value,'OFF');assert.equal(input.disabled,false);
  assert.equal(input.closest('details'),null);assert.equal(input.closest('.run-setting'),e.elements.start.closest('.run-setting'));
  assert.ok(e.dom.all.indexOf(e.elements['target-laps'])<e.dom.all.indexOf(input));assert.ok(e.dom.all.indexOf(input)<e.dom.all.indexOf(e.elements.start));
  assert.deepEqual(input.children.filter(c=>c.tagName==='OPTION').map(c=>c.value),['OFF','ON']);
  assert.match(input.children.find(c=>c.value==='ON').textContent,/已亮/,'ON is an explicit initial condition chosen by the operator');
  e.change(id,'ON');await e.context.updateStatus();assert.equal(input.value,'ON','idle polling preserves an unsubmitted selection');
  e.change(id,'bad');assert.equal(input.value,'ON','the UI only accepts the two declared modes');
  e.change(id,'OFF');loadedESP(e);await e.click('start');assert.equal(e.posts('/control')[0].body.initial_lamp_state,'OFF');
  assert.equal(e.posts('/lamp-config').length,0);
 }
 {
  const e=setup();await flush();loadedESP(e);e.change(id,'ON');e.elements['target-laps'].value='7';dragBound(e,'left',140);
  const server=e.defaultData();server.preparation={pending:true,phase:'SAVING',error:null,request_id:9,generation:0,initial_lamp_state:'ON'};let release;
  e.fetcher=async(url,options)=>url==='/control'&&JSON.parse(options.body).action==='start'?new Promise(resolve=>release=resolve):server;
  const starting=e.click('start');await flush();assert.equal(e.elements[id].disabled,true);assert.equal(e.elements[id].value,'ON');
  e.change(id,'OFF');assert.equal(e.elements[id].value,'ON','mode locks before the preparation HTTP response returns');
  assert.deepEqual(e.posts('/control')[0].body,{action:'start',generation:0,laps:7,lamp_configuration:{manual:true,roi:[140,51,62,61]},initial_lamp_state:'ON'});
  assert.equal(e.posts('/lamp-config').length,0,'mode, laps and frame have one atomic ARM request');
  release(e.response({message:'准备请求已接收'},202));await starting;assert.equal(e.elements[id].value,'ON');assert.equal(e.elements[id].disabled,true);
  Object.assign(server.readiness.start_light,{armed:true,phase:'WAIT_OFF',initial_lamp_state:'ON',lamp_configuration:{manual:true,roi:[140,51,62,61]}});
  server.preparation={...server.preparation,pending:false,phase:'ARMED'};await e.context.updateStatus();assert.equal(e.elements[id].disabled,true);assert.equal(e.elements[id].value,'ON');
  server.active=true;server.reason='FOLLOW';server.readiness.start_light.armed=false;await e.context.updateStatus();assert.equal(e.elements[id].disabled,true);
  e.context.navigator.onLine=false;e.windowEvents.offline();e.change(id,'OFF');assert.equal(e.elements[id].disabled,true);assert.equal(e.elements[id].value,'ON');
  e.context.navigator.onLine=true;e.windowEvents.online();await flush();loadedESP(e);assert.equal(e.elements[id].value,'ON');
  server.active=false;server.reason='STOPPED';server.generation=1;server.preparation={...server.preparation,pending:false,phase:'CANCELLED'};
  await e.click('stop');assert.equal(e.elements[id].disabled,false);e.change(id,'OFF');assert.equal(e.elements[id].value,'OFF');
  assert.equal(e.posts('/control').filter(r=>r.body.action==='start').length,1);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>{s.readiness.start_light.initial_lamp_state='ON'});
  assert.equal(e.elements[id].value,'OFF','a completed or idle peer mode is not a persisted preference');assert.equal(e.elements[id].disabled,false);
 }
 {
  const e=setup();await flush();await replaceStatus(e,s=>Object.assign(s.readiness.start_light,{armed:true,phase:'WAIT_OFF',initial_lamp_state:'ON'}));
  assert.match(e.elements['lamp-signal-state'].getAttribute('title'),/稳定基准.*请确保/);
  assert.doesNotMatch(e.elements['lamp-signal-state'].getAttribute('title'),/亮灯已确认/,'ON is an operator assertion, not automatic evidence of the initial physical lamp state');
  assert.match(e.elements['mission-state'].textContent,/基准稳定/);assert.doesNotMatch(e.elements['mission-state'].textContent,/已见亮灯/);
 }
 for(const phase of ['SAVING','ARM_QUEUED','ERROR']){
  const e=setup();await flush();await replaceStatus(e,s=>{s.preparation={pending:phase!=='ERROR',phase,error:phase==='ERROR'?'configuration_offline':null,request_id:9,generation:0,initial_lamp_state:'ON'}});
  assert.equal(e.elements[id].value,'ON',phase+': restore captured preparation mode');assert.equal(e.elements[id].disabled,true,phase+': captured mode stays locked');
 }
 for(const active of [false,true]){
  const e=setup();await flush();await replaceStatus(e,s=>{s.active=active;Object.assign(s.readiness.start_light,{armed:!active,phase:active?'RUNNING':'WAIT_OFF',initial_lamp_state:'ON'})});
  assert.equal(e.elements[id].value,'ON','a fresh page reflects the actual accepted mission mode');assert.equal(e.elements[id].disabled,true);
 }
 {
  const e=setup();await flush();e.change(id,'ON');await replaceStatus(e,s=>{s.age_ms=1000;s.preparation={pending:true,phase:'SAVING',generation:0,initial_lamp_state:'OFF'}});
  assert.equal(e.elements[id].disabled,true);assert.equal(e.elements[id].value,'ON','stale metadata cannot overwrite the local choice');
 }
 {
  const e=setup();await flush();loadedESP(e);e.change(id,'ON');const server=e.defaultData();
  e.fetcher=async url=>url==='/control'?e.response({message:'暂不可达'},409):server;
  await e.click('start');assert.equal(e.elements[id].disabled,true);assert.equal(e.elements[id].value,'ON','a rejected request does not unlock the operator-confirmed mode');
  await e.click('start');assert.deepEqual(e.posts('/control').map(r=>r.body.initial_lamp_state),['ON','ON']);
  server.generation=1;await e.click('stop');assert.equal(e.elements[id].disabled,false);
 }
 {
  const e=setup();await flush();loadedESP(e);e.change(id,'ON');let release;const server=e.defaultData();
  e.fetcher=async(url,options)=>url==='/control'&&JSON.parse(options.body).action==='start'?new Promise(resolve=>release=resolve):server;
  const starting=e.click('start');await flush();server.generation=1;await e.click('stop');assert.equal(e.elements[id].disabled,false);e.change(id,'OFF');
  release(e.response({message:'旧准备请求'},202));await starting;await e.context.updateStatus();assert.equal(e.elements[id].value,'OFF');assert.equal(e.elements[id].disabled,false,'a late ARM result cannot relock a stopped selection');
 }
 {
  const e=setup();await flush();loadedESP(e);e.change(id,'ON');const server=e.defaultData();
  e.fetcher=async()=>server;await e.click('start');server.generation=1;server.reason='LAPS COMPLETE';
  server.preparation={pending:false,phase:'ARMED',generation:0,initial_lamp_state:'ON'};await e.context.updateStatus();
  assert.equal(e.elements[id].disabled,false,'verified mission completion releases the mode');e.change(id,'OFF');assert.equal(e.elements[id].value,'OFF');
 }
 {
  const e=setup();await flush();loadedESP(e);e.change(id,'ON');await e.click('direct-start');
  assert.deepEqual(e.posts('/control')[0].body,{action:'start_now',generation:0,laps:2},'manual testing does not submit an initial-lamp assertion');
 }
 console.log('initial lamp state default, atomic submission, mode locks and restored mission state passed');
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
  assert.equal(e.elements['esp-target'].hidden,true,'the line overlay replaces the duplicate box');assertVisibleGuides(e);
 }
 {
  const e=setup();await flush();e.fetcher=async()=>{const s=e.defaultData();s.readiness.crosswalk={mode:'yield',phase:'SEARCH',observation:null,hold_elapsed_s:3.2,yield_completed:true,yield_verified:true};return s};await e.context.updateStatus();
  assert.match(e.elements['crosswalk-status'].textContent,/上次横道：合法停位与停留已验证/);
  e.fetcher=async()=>{const s=e.defaultData();s.readiness.crosswalk={mode:'yield',phase:'APPROACH',observation:{near_m:.3},hold_elapsed_s:0,yield_completed:false,yield_verified:false};return s};await e.context.updateStatus();
  assert.doesNotMatch(e.elements['crosswalk-status'].textContent,/上次横道|合法停位与停留已验证/);
 }
 await fieldScenarios();
 await lampSignalScenarios();
 await initialLampStateScenarios();
 console.log('13 preview connection scenarios passed');
}
main().catch(e=>{console.error(e);process.exitCode=1});
