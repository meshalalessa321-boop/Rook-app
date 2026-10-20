/* Rook Tools v1: local, transparent utilities. No network, no external model, no scripted replies. */
(()=> {
  "use strict";
  const KEY="rook-memory-v1";
  const $=id=>document.getElementById(id);
  const safe=(s)=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  function read(){try{const v=JSON.parse(localStorage.getItem(KEY)||"[]");return Array.isArray(v)?v:[]}catch{return[]}}
  function write(items){localStorage.setItem(KEY,JSON.stringify(items))}
  function render(){
    const list=$("rookMemoryList");if(!list)return;
    const items=read();list.innerHTML="";
    if(!items.length){list.innerHTML='<p class="tool-muted">No memories saved on this device yet.</p>';return}
    for(const item of items){
      const row=document.createElement("div");row.className="rook-memory-item";
      row.innerHTML='<label><input type="checkbox" class="rook-memory-use" data-id="'+safe(item.id)+'"> <span>'+safe(item.text)+'</span></label><button type="button" class="smallbtn rook-memory-delete" data-id="'+safe(item.id)+'">Delete</button>';
      list.append(row);
    }
  }
  function addMemory(text){
    const clean=String(text||"").trim().slice(0,1200);if(!clean)throw new Error("Write a note first.");
    const items=read();items.unshift({id:Date.now().toString(36)+Math.random().toString(36).slice(2,7),text:clean,createdAt:new Date().toISOString()});
    write(items.slice(0,100));render();
  }
  function selectedMemories(){
    const ids=[...document.querySelectorAll(".rook-memory-use:checked")].map(x=>x.dataset.id);
    return read().filter(x=>ids.includes(x.id)).map(x=>x.text).slice(0,8);
  }
  function analyze(text){
    const words=(String(text||"").toLowerCase().match(/[\p{L}\p{N}']+/gu)||[]);
    const counts=new Map();for(const w of words)counts.set(w,(counts.get(w)||0)+1);
    const repeated=[...counts].filter(([w,n])=>n>=3&&w.length>2).sort((a,b)=>b[1]-a[1]).slice(0,8);
    const unique=counts.size, ratio=words.length?unique/words.length:0;
    const latin=(String(text||"").match(/[A-Za-z]/g)||[]).length,arabic=(String(text||"").match(/[\u0600-\u06FF]/g)||[]).length;
    const lines=String(text||"").split(/\n+/).filter(x=>x.trim()).length;
    return {characters:String(text||"").length,words:words.length,uniqueWords:unique,uniqueRatio:ratio,repeated,latin,arabic,lines};
  }
  function showAnalysis(text){
    const out=$("rookAnalysis");if(!out)return;
    const a=analyze(text);
    out.innerHTML='<div class="rook-metrics"><span><b>'+a.words+'</b> words</span><span><b>'+a.uniqueWords+'</b> unique</span><span><b>'+Math.round(a.uniqueRatio*100)+'%</b> lexical variety</span><span><b>'+a.lines+'</b> lines</span></div>'+
      '<p class="tool-muted">Script counts: '+a.latin+' Latin letters, '+a.arabic+' Arabic letters. These are diagnostics, not a measure of intelligence or meaning.</p>'+
      (a.repeated.length?'<p><strong>Repeated words (3+ uses):</strong> '+a.repeated.map(([w,n])=>safe(w)+' ×'+n).join(', ')+'</p>':'<p>No obvious high-frequency repeated words.</p>');
  }
  window.RookTools={
    init(){render();
      $("rookMemoryAdd")?.addEventListener("click",()=>{try{addMemory($("rookMemoryInput").value);$("rookMemoryInput").value="";$("rookToolNotice").textContent="Memory saved only in this browser."}catch(e){$("rookToolNotice").textContent=e.message}});
      $("rookMemoryList")?.addEventListener("click",e=>{const b=e.target.closest(".rook-memory-delete");if(!b)return;write(read().filter(x=>x.id!==b.dataset.id));render()});
      $("rookAnalyzeLast")?.addEventListener("click",()=>{const msgs=[...document.querySelectorAll(".msg.rook")];const last=msgs.at(-1);showAnalysis(last?.textContent||"");});
      $("rookAnalyzeInput")?.addEventListener("click",()=>showAnalysis($("rookAnalysisInput").value));
    },
    selectedMemories,
    analyze,
    showAnalysis
  };
})();