const $=s=>document.querySelector(s);
const all=s=>Array.from(document.querySelectorAll(s));
let token=localStorage.getItem("meridian_token")||"",page="overview";
const esc=v=>String(v===null||v===undefined?"":v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
async function api(path,options){options=options||{};const headers=Object.assign({"Content-Type":"application/json","Authorization":"Bearer "+token},options.headers||{});const response=await fetch(path,Object.assign({},options,{headers:headers}));const data=await response.json().catch(()=>({detail:response.statusText}));if(!response.ok)throw new Error(data.detail||("HTTP "+response.status));return data}
function card(title,body,cls){return '<section class="card '+(cls||'')+'"><div class="row"><strong>'+title+'</strong></div>'+body+'</section>'}
function pill(value){let v=String(value||"unknown");let cls=(["failed","error","offline"].includes(v)?"err":(["waiting_for_model","blocked","pending","review"].includes(v)?"warn":""));return '<span class="pill '+cls+'">'+esc(v)+'</span>'}
function list(items,renderer){return items&&items.length?'<div class="list">'+items.map(renderer).join("")+'</div>':'<div class="empty" style="margin-top:13px">Nothing here yet.</div>'}
function taskItem(t){return '<div class="item"><div class="row"><strong>'+esc(t.title)+'</strong>'+pill(t.status)+'</div><p>'+esc(t.goal)+'</p><div class="label">'+esc(t.priority)+' · '+esc(String(t.id).slice(0,8))+'</div>'+(t.blocker?'<p class="warntext">'+esc(t.blocker)+'</p>':'')+'</div>'}
function runItem(r){return '<div class="item"><div class="row"><strong>'+esc(r.summary||"Agent run")+'</strong>'+pill(r.status)+'</div><p class="label">Started '+esc(r.started_at)+' · Finished '+esc(r.finished_at||"not finished")+'</p><p>'+esc(r.evidence||"No execution evidence recorded.")+'</p></div>'}
async function show(nextPage){
 if(nextPage)page=nextPage;
 all("#nav button").forEach(b=>b.classList.toggle("active",b.dataset.page===page));
 const titles={overview:"Overview",chat:"Talk to Meridian",tasks:"Task queue",runs:"Activity & reports",approvals:"Review & approvals",memory:"Working memory"};
 $("#pageTitle").textContent=titles[page]||"Overview";
 const target=$("#content");target.innerHTML=card("Loading","Fetching current records…");
 try{
  const status=await api("/api/status");
  $("#status").innerHTML=pill(status.worker)+"<br>"+(status.model_ready?"Model ready":"Model not configured")+" · "+(status.github_ready?"GitHub ready":"GitHub not configured")+"<br>Model calls today: "+esc(status.model_calls_today)+"/"+esc(status.daily_model_call_limit);
  $("#readiness").textContent="Model "+(status.model_ready?"connected":"missing")+" · GitHub "+(status.github_ready?"connected":"missing");
  if(page==="overview"){
   const data=await api("/api/overview");
   target.innerHTML='<div class="grid">'+
    card("Worker state",'<div class="metric">'+esc(status.worker)+'</div><p class="muted">Heartbeat: '+esc(status.heartbeat||"not yet recorded")+'</p><div class="row">'+pill(status.model_ready?"model ready":"waiting_for_model")+pill(status.github_ready?"GitHub ready":"GitHub unavailable")+'</div><p class="label">A heartbeat shows the process is alive. A task run appears only after actual execution.</p>',"half")+
    card("Queued tasks",'<div class="metric">'+esc(data.counts.queued)+'</div><p class="muted">'+esc(data.counts.running)+' running · '+esc(data.counts.done)+' done</p>',"third")+
    card("Approvals awaiting you",'<div class="metric">'+esc(data.counts.approvals)+'</div><p class="muted">No code is published before approval.</p>',"third")+
    card("Current task backlog",list(data.tasks,taskItem),"half")+
    card("Latest executions",list(data.runs,runItem),"half")+
    card("Connection diagnostics",'<div class="list"><div class="item row"><span>Database process</span>'+pill(status.database_ready?"ready":"not ready")+'</div><div class="item row"><span>Model API</span>'+pill(status.model_ready?"configured":"not configured")+'</div><div class="item row"><span>GitHub tools</span>'+pill(status.github_ready?"configured":"not configured")+'</div><div class="item row"><span>Worker enabled</span>'+pill(status.worker_enabled?"yes":"no")+'</div></div>'+(status.last_error?'<p class="warntext">'+esc(status.last_error)+'</p>':""),"half")+
   '</div>';
  }else if(page==="chat"){
   const messages=await api("/api/messages");
   target.innerHTML=card("Conversation",'You can ask for analysis, request plans, and discuss engineering tasks. Use Task Queue to run work in the background.<div id="chatlog" class="chatlog">'+messages.map(m=>'<div class="bubble '+(m.role==="user"?"user":"")+'"><div class="label">'+esc(m.role)+" · "+esc(m.created_at)+'</div>'+esc(m.content)+'</div>').join("")+'</div><form id="chatform" class="form"><textarea id="message" placeholder="Describe what you want Meridian to investigate…" required></textarea><button class="primary">Send message</button></form>');
   $("#chatlog").scrollTop=$("#chatlog").scrollHeight;
   $("#chatform").onsubmit=async e=>{e.preventDefault();const btn=e.submitter;const input=$("#message");const message=input.value.trim();if(!message)return;btn.disabled=true;try{await api("/api/chat",{method:"POST",body:JSON.stringify({message:message})});await show("chat")}catch(err){alert(err.message)}finally{btn.disabled=false}};
  }else if(page==="tasks"){
   const data=await api("/api/tasks");
   target.innerHTML='<div class="grid">'+card("Create a task",'<form id="taskform" class="form"><label>Title<input id="tasktitle" maxlength="200" required placeholder="Investigate Rook generation quality"></label><label>Goal and success criteria<textarea id="taskgoal" required maxlength="10000" placeholder="Describe the issue, what to inspect, and what evidence would count as success."></textarea></label><div class="two"><label>Priority<select id="priority"><option>normal</option><option>high</option><option>low</option></select></label><button class="primary">Queue task</button></div></form>',"half")+card("Durable task queue",list(data,taskItem),"half")+'</div>';
   $("#taskform").onsubmit=async e=>{e.preventDefault();const btn=e.submitter;btn.disabled=true;try{await api("/api/tasks",{method:"POST",body:JSON.stringify({title:$("#tasktitle").value,goal:$("#taskgoal").value,priority:$("#priority").value})});await show("tasks")}catch(err){alert(err.message)}finally{btn.disabled=false}};
  }else if(page==="runs"){
   const data=await api("/api/runs");target.innerHTML=card("Execution log",list(data,runItem));
  }else if(page==="approvals"){
   const data=await api("/api/approvals");
   target.innerHTML=card("Change proposals",list(data,a=>{
    let detail={summary:a.details};
    try{detail=JSON.parse(a.details)}catch(e){}
    const paths=Array.isArray(detail.files)?detail.files.map(f=>f.path).join(", "):"";
    return '<div class="item"><div class="row"><strong>'+esc(a.title)+'</strong>'+pill(a.status)+'</div><p>'+esc(detail.summary||a.details)+'</p>'+(paths?'<p class="label">Files: '+esc(paths)+'</p>':'')+(a.status==="pending"?'<div class="row"><button class="danger" data-reject="'+esc(a.id)+'">Reject</button><button class="primary" data-approve="'+esc(a.id)+'">Approve and create draft PR</button></div>':'')+(a.evidence_url?'<p><a target="_blank" rel="noopener noreferrer" href="'+esc(a.evidence_url)+'">Open pull request</a></p>':'')+'</div>'
   }));
   all("[data-approve]").forEach(b=>b.onclick=async()=>{if(!confirm("Approve this proposal and create a draft pull request? No merge will occur."))return;b.disabled=true;try{const result=await api("/api/approvals/"+b.dataset.approve+"/decision",{method:"POST",body:JSON.stringify({approved:true})});alert("Draft pull request: "+result.url);await show("approvals")}catch(err){alert(err.message);b.disabled=false}});
   all("[data-reject]").forEach(b=>b.onclick=async()=>{if(!confirm("Reject this proposal?"))return;try{await api("/api/approvals/"+b.dataset.reject+"/decision",{method:"POST",body:JSON.stringify({approved:false})});await show("approvals")}catch(err){alert(err.message)}});
  }else{
   const data=await api("/api/memories");
   target.innerHTML=card("Durable working memory",'<form id="memoryform" class="form"><label>Memory title<input id="memorytitle" required maxlength="200" placeholder="Rook model architecture decision"></label><label>Memory content<textarea id="memorycontent" required maxlength="12000" placeholder="Decision, verified fact, task handoff, or lesson learned"></textarea></label><button class="primary">Save memory</button></form>'+list(data,m=>'<div class="item"><div class="row"><strong>'+esc(m.title)+'</strong>'+pill(m.verified?"verified":"unverified")+'</div><p>'+esc(m.content)+'</p><div class="label">'+esc(m.category)+' · '+esc(m.created_at)+'</div>'+(m.source_url?'<p><a target="_blank" rel="noopener noreferrer" href="'+esc(m.source_url)+'">Source</a></p>':'')+'</div>'));
   $("#memoryform").onsubmit=async e=>{e.preventDefault();const btn=e.submitter;btn.disabled=true;try{await api("/api/memories",{method:"POST",body:JSON.stringify({title:$("#memorytitle").value,content:$("#memorycontent").value,category:"project",verified:false})});await show("memory")}catch(err){alert(err.message)}finally{btn.disabled=false}};
  }
 }catch(err){target.innerHTML=card("Unable to load workspace",'<p>'+esc(err.message)+'</p><button id="retry">Retry</button>');$("#retry").onclick=()=>show(page)}
}
$("#nav").addEventListener("click",e=>{const btn=e.target.closest("[data-page]");if(btn)show(btn.dataset.page)});
$("#logout").addEventListener("click",()=>{localStorage.removeItem("meridian_token");token="";$("#app").classList.add("hidden");$("#login").classList.remove("hidden");$("#token").value=""});
$("#loginform").addEventListener("submit",async e=>{e.preventDefault();token=$("#token").value;const btn=e.submitter;btn.disabled=true;try{await api("/api/status");localStorage.setItem("meridian_token",token);$("#login").classList.add("hidden");$("#app").classList.remove("hidden");await show("overview")}catch(err){$("#loginerr").textContent=err.message;token=""}finally{btn.disabled=false}});
if(token){api("/api/status").then(()=>{$("#login").classList.add("hidden");$("#app").classList.remove("hidden");show("overview")}).catch(()=>{token="";localStorage.removeItem("meridian_token")})}
