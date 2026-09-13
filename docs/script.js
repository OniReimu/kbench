'use strict';
const $=id=>document.getElementById(id);
const isChinese=document.documentElement.lang==='zh-CN';
const tr=(zh,en)=>isChinese?zh:en;
const laneNames={'P':tr('P · 模型权重','P · Model weights'),'C':tr('C · 工作上下文','C · Working context'),'R-text':tr('R-text · 文本检索','R-text · Text retrieval'),'R-struct':tr('R-struct · 结构化检索','R-struct · Structured retrieval')};
const laneCopy={
'P':[
 tr('模型权重中的目标信息','Target information in model weights'),
 tr('通过 LoRA 微调注入目标信息，智能体在推理时从参数中回忆。','Target information is injected through LoRA fine-tuning and accessed through parametric recall.'),
 tr('访问方式：参数回忆','Access: parametric recall'),
 tr('事后摘要','Elicited summary'),
 tr('目标值可在事后摘要中出现，因而仅观察最终回答会低估恢复率。','The target can appear in an elicited summary, increasing recovery beyond the final-answer channel.')],
'C':[
 tr('工作上下文中的目标信息','Target information in working context'),
 tr('目标值直接置于系统提示中，模型通过注意力机制访问输入内容。','The target is placed in the system prompt and accessed through attention over the input.'),
 tr('访问方式：读取上下文','Access: input context'),
 tr('最终回答','Final answer'),
 tr('上下文中的目标信息可直接进入最终回答。在 Qwen 的该配置下，答案通道已覆盖全部观测到的泄露。','Contextual information can be reproduced in the final answer. On Qwen in this configuration, the answer channel covers all observed recovery.')],
'R-text':[
 tr('检索文本中的目标信息','Target information in retrieved text'),
 tr('目标信息作为自由文本段落存储在外部索引中，通过搜索工具取回。','The target is stored as a free-text passage in an external index and retrieved through a search tool.'),
 tr('访问方式：段落检索','Access: passage retrieval'),
 tr('工具相关通道','Tool channels'),
 tr('将工具参数和返回内容纳入观察范围后，目标恢复率增加。','Observing tool arguments and returned observations increases target recovery.')],
'R-struct':[
 tr('结构化记录中的目标信息','Target information in structured records'),
 tr('lookup_record 等工具按字段返回记录，使目标值进入工具返回内容。','Field-keyed tools such as lookup_record return records containing the target value.'),
 tr('访问方式：结构化查询','Access: structured lookup'),
 tr('工具返回内容','Tool observations'),
 tr('StaR 降低了回答通道的泄露率，但工具返回内容仍可包含目标值。','StaR reduces answer-channel leakage while the target remains recoverable in tool observations.')]
};
function renderLane(key){document.querySelectorAll('[data-lane]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.lane===key)));['lane-title','lane-copy','lane-tag','lane-channel','lane-result'].forEach((id,i)=>$(id).textContent=laneCopy[key][i]);}
document.querySelectorAll('[data-lane]').forEach(b=>b.addEventListener('click',()=>renderLane(b.dataset.lane)));renderLane('P');
let budgetStep=4;
const budgetLabels=tr(['A₁ · 仅最终回答','A₂ · 加入事后摘要','A₃ · 加入推理文本','A₄ · 加入工具通道','A₅ · 全部六个通道'],['A₁ · Answer only','A₂ · Add summary','A₃ · Add reasoning','A₄ · Add tool channels','A₅ · All six channels']);
const budgetChannels=tr(['最终回答','回答与摘要','回答、摘要与推理文本','上述通道及工具参数与返回内容','上述通道及检索结果'],['Final answer','Answer and summary','Answer, summary, and reasoning','Above, plus tool arguments and observations','Above, plus retrieval results']);
function updateLanes(){const model=$('budget-model').value;const previous=$('budget-lane').value;$('budget-lane').replaceChildren(...Object.keys(KB_DATA.budget[model]).map(k=>new Option(laneNames[k],k)));if(KB_DATA.budget[model][previous])$('budget-lane').value=previous;renderBudget();}
function renderBudget(){const model=$('budget-model').value,lane=$('budget-lane').value,row=KB_DATA.budget[model][lane],rates=row.rates;const selected=rates[budgetStep];
 $('budget-name').textContent=budgetLabels[budgetStep];$('budget-rate').innerHTML=(selected*100).toFixed(1)+'<span>%</span>';
 const delta=((selected-rates[0])*100).toFixed(1);
 $('budget-delta').textContent=budgetStep===0?tr('仅观察最终回答时的基线恢复率','Baseline recovery with the final answer alone'):tr('较仅观察最终回答增加 '+delta+' 个百分点',delta+' percentage points above answer-only recovery');
 const insights=lane==='P'?
 [tr('摘要通道带来的增量','Additional recovery from summaries'),tr('从 A₁ 到 A₂ 的恢复率增量来自事后摘要。进一步纳入其他通道未增加恢复率。','The increase from A₁ to A₂ comes from elicited summaries. Additional channels do not increase recovery in these configurations.')]:lane==='C'?
 [tr('上下文中的直接恢复','Direct recovery from context'),model==='Qwen3.5-9B'?tr('该配置在所有观察范围下的恢复率均为 99.2%。','Recovery remains at 99.2% across all observation budgets.'):tr('纳入工具通道后，恢复率从 19.2% 增加至 22.3%。','Including tool channels raises recovery from 19.2% to 22.3%.')]:lane==='R-text'?
 [tr('工具通道带来的增量','Additional recovery from tool channels'),tr('A₄ 纳入工具参数及返回内容，使观察者能够恢复前三种观察范围未覆盖的目标值。','A₄ includes tool arguments and observations, exposing target values absent from the first three observation budgets.')]:
 [tr('结构化检索的模型间差异','Model differences on structured retrieval'),tr('Llama 和 Mistral 的答案通道基线恢复率较高，Qwen 的对应恢复率较低。各模型分别评估。','Answer-only baseline recovery is higher on Llama and Mistral than on Qwen. Each model is evaluated separately.')];
 $('budget-insight-title').textContent=insights[0];$('budget-insight').textContent=insights[1];
 const [changed,total]=row.flips.split('/');
 $('budget-flips').textContent=tr('观察范围从 A₁ 扩大至 A₅ 后，'+total+' 个可评分方法中有 '+changed+' 个的 K-class 判定发生变化。终末崩溃方法不计入分母。',changed+' of '+total+' scored method verdicts change from A₁ to A₅. Terminal-collapse cases are excluded from the denominator.');
 $('budget-scope').textContent=(model==='Mistral-7B'?tr('Mistral 的上下文基线恢复率低于 0.10，未通过可测性门槛。 ','Mistral context is excluded because baseline recovery is below the 0.10 measurability gate. '):'')+tr('观察范围逐级包含前一级通道。A₄ 同时加入两个工具通道，A₅ 加入检索结果；所有恢复率均按查询总体计算。','Budgets are nested. A₄ adds both tool channels; A₅ adds retrieval results. Recovery is computed over the query population.');
 $('budget-table-caption').textContent=model+' · '+laneNames[lane];$('budget-table').innerHTML=rates.map((v,i)=>`<tr><th scope="row">${budgetLabels[i]}</th><td>${budgetChannels[i]}</td><td>${v.toFixed(3)} (${(v*100).toFixed(1)}%)</td></tr>`).join('');
 const x=i=>54+i*144,y=v=>244-210*v;
 let svg=`<title>${model}, ${lane}: ${tr('目标恢复率','target recovery')}</title><desc>${rates.map((v,i)=>budgetLabels[i]+' '+(v*100).toFixed(1)+'%').join('; ')}</desc>`;
 [0,.25,.5,.75,1].forEach(v=>svg+=`<line x1="54" x2="630" y1="${y(v)}" y2="${y(v)}" stroke="#dce2d3" stroke-dasharray="3 5"/><text x="40" y="${y(v)+4}" text-anchor="end" font-size="11" fill="#75816b">${v*100}%</text>`);
 const pts=rates.map((v,i)=>`${x(i)},${y(v)}`).join(' ');
 svg+=`<polygon points="54,244 ${pts} 630,244" fill="#829868" fill-opacity=".08"/><polyline points="${pts}" fill="none" stroke="#59784d" stroke-width="3" stroke-linejoin="round"/><line x1="${x(budgetStep)}" x2="${x(budgetStep)}" y1="22" y2="244" stroke="#b77b52" stroke-dasharray="4 5" opacity=".7"/>`;
 rates.forEach((v,i)=>{svg+=`<circle cx="${x(i)}" cy="${y(v)}" r="${i===budgetStep?7:4}" fill="${i===budgetStep?'#b77b52':'#f7f5ef'}" stroke="${i===budgetStep?'#b77b52':'#59784d'}" stroke-width="2"/><text x="${x(i)+(i===0?12:i===4?-12:0)}" y="${y(v)-15}" text-anchor="${i===0?'start':i===4?'end':'middle'}" font-size="13" fill="#425b38">${(v*100).toFixed(1)}%</text><text x="${x(i)}" y="270" text-anchor="middle" font-size="12" fill="#6c7c5e">A${i+1}</text>`;});
 $('budget-chart').innerHTML=svg;document.querySelectorAll('[data-budget]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.budget)===budgetStep)));
}
$('budget-model').addEventListener('change',updateLanes);$('budget-lane').addEventListener('change',renderBudget);document.querySelectorAll('[data-budget]').forEach(b=>b.addEventListener('click',()=>{budgetStep=Number(b.dataset.budget);renderBudget();}));updateLanes();
const presets={selective:[10,0,0],retain:[10,60,0],collapse:[10,0,90]};
function renderScore(){const vals=['forget','retain','degen'].map(id=>Number($(id).value)/100);['forget','retain','degen'].forEach((id,i)=>$(id+'-out').textContent=Math.round(vals[i]*100)+(id==='degen'?tr(' 个百分点',' pp'):'%'));const factors=vals.map(v=>1-v);$('score-value').textContent=factors.reduce((a,b)=>a*b,1).toFixed(3);$('score-arithmetic').textContent=factors.map(v=>v.toFixed(2)).join(' × ');}
document.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-preset]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));presets[b.dataset.preset].forEach((v,i)=>$( ['forget','retain','degen'][i]).value=v);renderScore();}));
['forget','retain','degen'].forEach(id=>$(id).addEventListener('input',()=>{document.querySelectorAll('[data-preset]').forEach(b=>b.setAttribute('aria-pressed','false'));renderScore();}));renderScore();
function renderLeader(model){const rows=KB_DATA.leaderboard[model];$('eligible-count').textContent=rows.length;$('gate-dots').innerHTML=Array.from({length:20},(_,i)=>`<i class="${i<rows.length?'pass':''}"></i>`).join('');$('leader-caption').textContent=model+tr(' · 通过主要门槛的方法',' · Methods passing the primary gate');
 $('leader-insight').textContent=rows.length===1?tr('ELM 是该模型上唯一通过主要门槛的方法。','ELM is the only method passing the primary gate on this model.'):tr(rows.length+' 种方法通过门槛，其中 '+rows[0].method+' 的泄露抑制项最高；其余 '+(20-rows.length)+' 种未通过门槛。',rows.length+' methods pass the gate. '+rows[0].method+' has the highest suppression factor; '+(20-rows.length)+' methods are ineligible.');
 $('leader-table').innerHTML=rows.map(r=>`<tr><td>${r.method}</td><td><div class="meter-cell"><div class="meter-track" aria-hidden="true"><span style="width:${r.suppression*100}%"></span></div>${r.suppression.toFixed(3)}</div></td><td>${r.retain.toFixed(0)}%</td><td>+${r.degen.toFixed(1)} ${tr('个百分点','pp')}</td></tr>`).join('');document.querySelectorAll('[data-model]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.model===model)));}
document.querySelectorAll('[data-model]').forEach(b=>b.addEventListener('click',()=>renderLeader(b.dataset.model)));renderLeader('Llama-3.1-8B');
