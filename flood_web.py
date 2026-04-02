"""
flood_web.py — 水体变化监测可视化仪表盘生成器
生成单文件 index.html，含时序对比、滑动对比、变化统计、预警报告等模块。

使用普通字符串模板（非 f-string）彻底避免 CSS/JS 大括号转义问题，
仅在最后用 __SITE_DATA__ 占位符替换 JSON 数据。
"""
import json
import os

# ── HTML 模板（普通 r-string，CSS/JS 大括号无需任何转义）────────────────────
_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>水体变化监测系统</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:#080d12; --surface:#0e1620; --card:#131e2b;
  --border:#1a2e40; --border2:#223344;
  --cyan:#00c8e8; --cyan-dim:rgba(0,200,232,.12);
  --red:#ff4d4d; --green:#22e87a; --amber:#f5a623;
  --text:#c5d8e8; --muted:#4e6880; --white:#f0f6fb;
  --mono:'JetBrains Mono','Fira Code',Consolas,monospace;
  --radius:6px;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--text); font-family:var(--mono); font-size:13px; line-height:1.6; }

.topbar {
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 28px;
  background:linear-gradient(90deg,rgba(0,200,232,.06),transparent);
  border-bottom:1px solid var(--border);
}
.topbar h1 { font-size:18px; color:var(--white); letter-spacing:.5px; }
.topbar p  { color:var(--muted); font-size:11px; margin-top:2px; }
.badge { padding:4px 12px; border-radius:20px; font-size:11px; letter-spacing:1px; font-weight:700; border:1px solid currentColor; }
.badge-high { color:var(--red); }
.badge-mid  { color:var(--amber); }
.badge-low  { color:var(--green); }

.kpi-row { display:grid; grid-template-columns:repeat(5,1fr); border-bottom:1px solid var(--border); }
.kpi { padding:16px 20px; border-right:1px solid var(--border); background:var(--surface); }
.kpi:last-child { border-right:none; }
.kpi-label { color:var(--muted); font-size:10px; letter-spacing:1.5px; text-transform:uppercase; }
.kpi-value { font-size:24px; color:var(--white); margin:6px 0 2px; }
.kpi-sub   { font-size:11px; color:var(--muted); }
.kpi-value.up     { color:var(--red); }
.kpi-value.down   { color:var(--green); }
.kpi-value.stable { color:var(--cyan); }

.main { padding:24px 28px; display:flex; flex-direction:column; gap:24px; }
.section-title {
  font-size:10px; letter-spacing:2px; text-transform:uppercase;
  color:var(--muted); padding-bottom:10px;
  border-bottom:1px solid var(--border); margin-bottom:16px;
}
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:16px; }

.panel { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; }
.panel-header {
  padding:10px 14px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; justify-content:space-between;
  font-size:11px; color:var(--cyan);
}
.panel-body { padding:14px; }
.chip { background:var(--cyan-dim); color:var(--cyan); padding:2px 8px; border-radius:3px; font-size:10px; }

.timeline { display:flex; gap:12px; overflow-x:auto; padding-bottom:4px; }
.tl-item { flex:0 0 220px; background:var(--card); border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; position:relative; }
.tl-img  { width:100%; aspect-ratio:4/3; object-fit:cover; display:block; background:#0e1620; }
.tl-footer { padding:8px 10px; border-top:1px solid var(--border); }
.tl-label  { color:var(--cyan); font-size:11px; }
.tl-area   { color:var(--white); font-size:16px; margin:2px 0; }
.tl-delta  { font-size:11px; }
.tl-delta.up    { color:var(--red); }
.tl-delta.down  { color:var(--green); }
.tl-delta.first { color:var(--muted); }
.tl-index {
  position:absolute; top:8px; left:8px;
  background:rgba(8,13,18,.75); backdrop-filter:blur(4px);
  border:1px solid var(--border); border-radius:3px;
  font-size:10px; color:var(--cyan); padding:2px 6px;
}

.change-cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; }
.change-card  { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; }
.change-card img { width:100%; display:block; aspect-ratio:4/3; object-fit:cover; background:#0e1620; }
.change-footer { padding:10px 12px; }
.change-title  { color:var(--muted); font-size:10px; letter-spacing:1px; }
.change-stats  { display:flex; gap:14px; margin-top:8px; }
.cs-item { display:flex; flex-direction:column; align-items:center; }
.cs-dot  { width:8px; height:8px; border-radius:50%; margin-bottom:3px; }
.cs-val  { font-size:14px; color:var(--white); }
.cs-lbl  { font-size:10px; color:var(--muted); }
.dot-p { background:#1e78dc; }
.dot-r { background:#dc3c3c; }
.dot-n { background:#1ec850; }

.slider-wrap { border-radius:var(--radius); overflow:hidden; border:1px solid var(--border); }
.slider-tabs { display:flex; background:var(--surface); border-bottom:1px solid var(--border); flex-wrap:wrap; }
.slider-tab {
  padding:8px 16px; font-size:11px; color:var(--muted);
  cursor:pointer; border-right:1px solid var(--border); transition:color .2s,background .2s;
}
.slider-tab:hover  { color:var(--white); }
.slider-tab.active { color:var(--cyan); background:var(--cyan-dim); }
.slider-hint {
  padding:8px 14px; font-size:10px; color:var(--muted);
  background:var(--surface); border-bottom:1px solid var(--border);
  display:flex; justify-content:space-between; align-items:center;
}
.slider-container {
  position:relative; overflow:hidden; cursor:col-resize;
  user-select:none; background:#000; min-height:200px;
}
.slider-container img.bg { width:100%; display:block; pointer-events:none; }
.slider-overlay { position:absolute; top:0; left:0; width:50%; height:100%; overflow:hidden; }
.slider-overlay img { position:absolute; top:0; left:0; pointer-events:none; }
.slider-line {
  position:absolute; top:0; left:50%; width:2px; height:100%;
  background:var(--cyan); box-shadow:0 0 12px var(--cyan); transform:translateX(-50%);
}
.slider-handle {
  position:absolute; top:50%; left:50%;
  width:36px; height:36px; border-radius:50%;
  background:var(--cyan); transform:translate(-50%,-50%);
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 0 16px var(--cyan); font-size:9px; color:#000; letter-spacing:-1px;
}
.slider-label {
  position:absolute; top:10px; font-size:10px;
  background:rgba(8,13,18,.8); padding:3px 8px; border-radius:3px;
  pointer-events:none; border:1px solid var(--border);
}
.slider-label.left  { left:12px; }
.slider-label.right { right:12px; }

.chart-container { position:relative; height:260px; }

.matrix-table { width:100%; border-collapse:collapse; font-size:12px; }
.matrix-table th {
  padding:8px 10px; text-align:left;
  border-bottom:1px solid var(--border2);
  color:var(--muted); font-weight:normal; font-size:10px; letter-spacing:1px;
}
.matrix-table td { padding:8px 10px; border-bottom:1px solid var(--border); }
.matrix-table tr:last-child td { border-bottom:none; }
.matrix-table tr:hover td { background:rgba(0,200,232,.04); }
.val-pos { color:var(--red); }
.val-neg { color:var(--green); }
.val-0   { color:var(--muted); }

.metric-bars { display:flex; flex-direction:column; gap:10px; padding:4px 0; }
.mb-row  { display:flex; align-items:center; gap:10px; }
.mb-name { width:76px; color:var(--muted); font-size:11px; flex-shrink:0; }
.mb-track { flex:1; height:6px; background:var(--border); border-radius:3px; overflow:hidden; }
.mb-fill  { height:100%; border-radius:3px; transition:width .8s ease; }
.mb-val   { width:46px; text-align:right; color:var(--white); font-size:11px; flex-shrink:0; }

.advice-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.advice-item {
  background:var(--card); border:1px solid var(--border);
  border-left:3px solid var(--cyan);
  border-radius:0 var(--radius) var(--radius) 0; padding:14px;
}
.advice-label { color:var(--cyan); font-size:10px; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px; }
.advice-text  { color:var(--white); font-size:12px; line-height:1.8; white-space:pre-wrap; }
.advice-ol    { color:var(--white); font-size:12px; }

/* ── 图片可点击放大提示 ────────────────────────────── */
.tl-item { cursor:zoom-in; }
.tl-item:hover .tl-img { opacity:.88; }
.change-card img { cursor:zoom-in; }
.change-card img:hover { opacity:.88; }
.tl-img, .change-card img { transition:opacity .15s; }

/* ── Lightbox ────────────────────────────────────── */
#lightbox {
  display:none; position:fixed; inset:0; z-index:9999;
  background:rgba(4,8,14,.94); align-items:center; justify-content:center;
  flex-direction:column; gap:12px;
}
#lightbox.open { display:flex; }
#lb-img {
  max-width:92vw; max-height:82vh; object-fit:contain;
  border:1px solid var(--border); border-radius:var(--radius);
  user-select:none;
}
#lb-caption {
  color:var(--muted); font-size:11px; letter-spacing:.5px; text-align:center;
}
#lb-nav {
  display:flex; gap:12px; align-items:center;
}
.lb-btn {
  background:rgba(14,22,32,.9); border:1px solid var(--border);
  color:var(--cyan); padding:6px 14px; border-radius:var(--radius);
  cursor:pointer; font-size:13px; font-family:var(--mono);
  transition:background .15s, border-color .15s;
}
.lb-btn:hover { background:var(--cyan-dim); border-color:var(--cyan); }
#lb-close {
  position:fixed; top:18px; right:22px;
  background:transparent; border:1px solid var(--border);
  color:var(--muted); font-size:18px; width:36px; height:36px;
  border-radius:50%; cursor:pointer; display:flex;
  align-items:center; justify-content:center;
  transition:color .15s, border-color .15s;
}
#lb-close:hover { color:var(--white); border-color:var(--white); }
#lb-counter { color:var(--muted); font-size:11px; min-width:50px; text-align:center; }

/* ── 滑块键盘提示 ────────────────────────────────── */
.slider-kb-hint {
  font-size:10px; color:var(--muted);
  padding:5px 14px; background:var(--surface);
  border-top:1px solid var(--border);
  display:flex; gap:14px; align-items:center;
}
.kb-key {
  display:inline-block; padding:1px 6px;
  border:1px solid var(--border2); border-radius:3px;
  font-size:10px; color:var(--cyan); line-height:1.6;
}
.slider-delta-badge {
  position:absolute; bottom:10px; left:50%; transform:translateX(-50%);
  background:rgba(8,13,18,.85); border:1px solid var(--border);
  border-radius:var(--radius); padding:3px 10px;
  font-size:11px; color:var(--white); pointer-events:none;
  white-space:nowrap;
}

/* ── 移动端响应式（精细化） ────────────────────────── */
@media(max-width:1024px) {
  .main { padding:20px 20px; }
}
@media(max-width:900px) {
  .kpi-row { grid-template-columns:repeat(3,1fr); }
  .two-col { grid-template-columns:1fr; }
  .advice-grid { grid-template-columns:1fr; }
  .topbar { padding:12px 20px; }
  .main { padding:16px 20px; gap:20px; }
}
@media(max-width:680px) {
  .kpi-row { grid-template-columns:repeat(2,1fr); }
  .kpi:nth-child(3) { border-right:none; }
  .kpi { padding:12px 14px; }
  .kpi-value { font-size:20px; }
  .topbar h1 { font-size:15px; }
  .topbar { flex-direction:column; gap:8px; align-items:flex-start; }
  .tl-item { flex:0 0 160px; }
  .chart-container { height:200px; }
  .matrix-table { font-size:11px; }
  .matrix-table th, .matrix-table td { padding:6px 8px; }
  .section-title { font-size:9px; }
  #tl-layer-btns button { font-size:9px !important; padding:2px 7px !important; }
}
@media(max-width:420px) {
  .kpi-row { grid-template-columns:1fr 1fr; }
  .kpi:nth-child(5) { grid-column:1/-1; border-right:none; }
  .main { padding:12px 14px; gap:16px; }
  .topbar { padding:10px 14px; }
  .advice-item { padding:10px 12px; }
}
</style>
</head>
<body>

<!-- Lightbox -->
<div id="lightbox" role="dialog" aria-modal="true">
  <button id="lb-close" aria-label="关闭">✕</button>
  <img id="lb-img" src="" alt="">
  <div id="lb-caption"></div>
  <div id="lb-nav">
    <button class="lb-btn" id="lb-prev">‹ 上一张</button>
    <span id="lb-counter"></span>
    <button class="lb-btn" id="lb-next">下一张 ›</button>
  </div>
</div>

<header class="topbar">
  <div>
    <h1>水体变化监测系统</h1>
    <p>MULTI-TEMPORAL WATER BODY CHANGE DETECTION · SAR REMOTE SENSING</p>
  </div>
  <div id="warning-badge" class="badge"></div>
</header>

<div class="kpi-row">
  <div class="kpi"><div class="kpi-label">预警等级</div><div class="kpi-value" id="kpi-level">—</div><div class="kpi-sub" id="kpi-summary"></div></div>
  <div class="kpi"><div class="kpi-label">风险评分</div><div class="kpi-value" id="kpi-score">—</div><div class="kpi-sub">0=安全 / 1=极危</div></div>
  <div class="kpi"><div class="kpi-label">水域净变化</div><div class="kpi-value" id="kpi-delta">—</div><div class="kpi-sub">首末时相对比</div></div>
  <div class="kpi"><div class="kpi-label">累计新增水域</div><div class="kpi-value" id="kpi-new">—</div><div class="kpi-sub">各时相叠加</div></div>
  <div class="kpi"><div class="kpi-label">时相数</div><div class="kpi-value" id="kpi-count">—</div><div class="kpi-sub">已处理时相</div></div>
</div>

<main class="main">

  <section>
    <div class="section-title" style="display:flex;align-items:center;justify-content:space-between">
      <span>时序水域 · 分割结果缩略图</span>
      <div id="tl-layer-btns" style="display:flex;gap:4px"></div>
    </div>
    <div class="timeline" id="timeline"></div>
  </section>

  <section class="two-col">
    <div class="panel">
      <div class="panel-header"><span>水域面积趋势</span><span class="chip">km²</span></div>
      <div class="panel-body"><div class="chart-container"><canvas id="areaChart"></canvas></div></div>
    </div>
    <div class="panel">
      <div class="panel-header"><span>时相间变化量对比</span><span class="chip">km²</span></div>
      <div class="panel-body"><div class="chart-container"><canvas id="deltaChart"></canvas></div></div>
    </div>
  </section>

  <section id="change-section" style="display:none">
    <div class="section-title">
      变化检测图 ·
      <span style="color:#1e78dc">■ 持续</span>
      <span style="color:#dc3c3c">■ 消退</span>
      <span style="color:#1ec850">■ 新增</span>
    </div>
    <div class="change-cards" id="change-cards"></div>
  </section>

  <section id="slider-section" style="display:none">
    <div class="section-title">分割结果 · 滑动对比</div>
    <div class="slider-wrap">
      <div class="slider-tabs" id="slider-tabs"></div>
      <div class="slider-hint">
        <span id="slider-hint-text">← 左右拖动分隔线对比两期水域分割结果</span>
        <span id="slider-ratio"></span>
      </div>
      <div class="slider-container" id="sliderContainer">
        <img class="bg" id="sliderBg" src="" alt="">
        <div class="slider-overlay" id="sliderOverlay">
          <img id="sliderFg" src="" alt="">
        </div>
        <div class="slider-line" id="sliderLine">
          <div class="slider-handle">&#9667;&#9657;</div>
        </div>
        <div class="slider-label left"  id="lblLeft"></div>
        <div class="slider-label right" id="lblRight"></div>
        <div class="slider-delta-badge" id="sliderDelta" style="display:none"></div>
      </div>
      <div class="slider-kb-hint">
        <span><span class="kb-key">←</span><span class="kb-key">→</span> 键盘微调</span>
        <span><span class="kb-key">Home</span> 归左 · <span class="kb-key">End</span> 归右</span>
        <span><span class="kb-key">Space</span> 重置到50%</span>
        <span style="margin-left:auto"><span class="kb-key">点击图片</span> 全屏查看</span>
      </div>
    </div>
  </section>

  <section class="two-col">
    <div class="panel">
      <div class="panel-header"><span>时相变化统计矩阵</span></div>
      <div class="panel-body">
        <table class="matrix-table">
          <thead><tr>
            <th>时相对</th><th>持续 (km²)</th><th>消退 (km²)</th>
            <th>新增 (km²)</th><th>净变化 (km²)</th>
          </tr></thead>
          <tbody id="matrix-body"></tbody>
        </table>
      </div>
    </div>
    <div class="panel">
      <div class="panel-header"><span id="quality-panel-title">平均分割质量指标</span><span class="chip" id="quality-mode-chip"></span></div>
      <div class="panel-body">
        <div id="quality-note" style="display:none;font-size:11px;color:var(--amber);border:1px solid rgba(245,166,35,.25);background:rgba(245,166,35,.06);border-radius:var(--radius);padding:8px 10px;margin-bottom:12px;line-height:1.6;"></div>
        <div class="metric-bars" id="metric-bars"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-title">决策建议</div>
    <div class="advice-grid">
      <div class="advice-item" style="border-left-color:var(--cyan)">
        <div class="advice-label">&#9670; 专家态势研判</div>
        <div class="advice-text" id="adv-expert"></div>
      </div>
      <div class="advice-item" style="border-left-color:var(--amber)">
        <div class="advice-label">&#9650; 预警响应建议</div>
        <div class="advice-text" id="adv-warn"></div>
      </div>
      <div class="advice-item" style="border-left-color:var(--red)">
        <div class="advice-label">&#9654; 应急处置措施</div>
        <div class="advice-ol" id="adv-act"></div>
      </div>
    </div>
  </section>

</main>

<script>
var D = __SITE_DATA__;

var $ = function(id) { return document.getElementById(id); };
var fmt = function(v) { return (v >= 0 ? '+' : '') + v.toFixed(3); };
var clsCh = function(v) { return v > 0.01 ? 'val-pos' : v < -0.01 ? 'val-neg' : 'val-0'; };

// KPI
var ra    = (D.report && D.report.risk_assessment)  || {};
var ds    = (D.report && D.report.decision_support) || {};
var areas = D.areas || [];
var level = ra.warning_level || '-';
var score = Number(ra.risk_score || 0);
var netDelta = areas.length > 1 ? areas[areas.length - 1] - areas[0] : 0;
var totalNew = (D.change_stats || []).reduce(function(s, c) { return s + (c.new || 0); }, 0);

var levelBadge = { '高': 'high', '中': 'mid', '低': 'low' };
var levelCls   = { '高': 'up',   '中': '',    '低': 'down' };
$('warning-badge').textContent = '⚠ 预警等级：' + level;
$('warning-badge').className   = 'badge badge-' + (levelBadge[level] || 'low');
$('kpi-level').textContent     = level;
$('kpi-level').className       = 'kpi-value ' + (levelCls[level] || '');
$('kpi-summary').textContent   = ra.summary || '';
$('kpi-score').textContent     = score.toFixed(3);
$('kpi-score').className       = 'kpi-value ' + (score >= 0.7 ? 'up' : score >= 0.4 ? '' : 'down');
$('kpi-delta').textContent     = fmt(netDelta) + ' km²';
$('kpi-delta').className       = 'kpi-value ' + (netDelta > 0.05 ? 'up' : netDelta < -0.05 ? 'down' : 'stable');
$('kpi-new').textContent       = '+' + totalNew.toFixed(3) + ' km²';
$('kpi-new').className         = 'kpi-value ' + (totalNew > 0.1 ? 'up' : 'stable');
$('kpi-count').textContent     = D.labels.length;

// 时序缩略图 + 图层切换
var tlLayers = [
  { key: 'result_images', label: '叠加图',   icon: '⊞' },
  { key: 'seg_images',    label: '分割掩膜', icon: '◧' },
  { key: 'prob_images',   label: '概率图',   icon: '◈' }
];
var tlCurrentLayer = 'result_images';

function buildTimeline(layerKey) {
  var tl = $('timeline');
  tl.innerHTML = '';
  var imgs = D[layerKey] || D.result_images;
  D.labels.forEach(function(lbl, i) {
    var prev  = i > 0 ? areas[i - 1] : null;
    var delta = prev !== null ? areas[i] - prev : null;
    var dCls  = delta === null ? 'first' : delta > 0.01 ? 'up' : delta < -0.01 ? 'down' : 'first';
    var dTxt  = delta === null ? '基准时相' : fmt(delta) + ' km²';
    var el    = document.createElement('div');
    el.className = 'tl-item';
    el.innerHTML =
      '<div class="tl-index">T' + (i + 1) + '</div>' +
      '<img class="tl-img" src="' + (imgs[i] || '') + '" alt="' + lbl + '" loading="lazy">' +
      '<div class="tl-footer">' +
        '<div class="tl-label">' + lbl + '</div>' +
        '<div class="tl-area">' + (areas[i] || 0).toFixed(3) + ' km²</div>' +
        '<div class="tl-delta ' + dCls + '">' + dTxt + '</div>' +
      '</div>';
    tl.appendChild(el);
  });
}

// 构建图层切换按钮（只显示数据中存在的图层）
var btnContainer = $('tl-layer-btns');
tlLayers.forEach(function(layer) {
  if (!D[layer.key] || !D[layer.key].length) return;
  var btn = document.createElement('button');
  btn.textContent = layer.icon + ' ' + layer.label;
  btn.dataset.key = layer.key;
  btn.style.cssText = 'padding:3px 10px;font-size:10px;letter-spacing:.5px;cursor:pointer;border-radius:3px;border:1px solid var(--border);background:transparent;color:var(--muted);font-family:var(--mono);transition:all .15s';
  btn.onmouseenter = function() { if (tlCurrentLayer !== layer.key) this.style.color = 'var(--white)'; };
  btn.onmouseleave = function() { if (tlCurrentLayer !== layer.key) this.style.color = 'var(--muted)'; };
  btn.onclick = function() {
    tlCurrentLayer = layer.key;
    buildTimeline(layer.key);
    btnContainer.querySelectorAll('button').forEach(function(b) {
      var active = b.dataset.key === layer.key;
      b.style.background = active ? 'var(--cyan-dim)' : 'transparent';
      b.style.color      = active ? 'var(--cyan)'    : 'var(--muted)';
      b.style.borderColor= active ? 'var(--cyan)'    : 'var(--border)';
    });
  };
  if (layer.key === tlCurrentLayer) {
    btn.style.background  = 'var(--cyan-dim)';
    btn.style.color       = 'var(--cyan)';
    btn.style.borderColor = 'var(--cyan)';
  }
  btnContainer.appendChild(btn);
});
buildTimeline(tlCurrentLayer);

// Chart.js 公共配置 + 自定义深色 Tooltip
var tooltipPlugin = {
  backgroundColor: 'rgba(8,13,18,.96)',
  borderColor: '#1a2e40',
  borderWidth: 1,
  titleColor: '#f0f6fb',
  bodyColor: '#c5d8e8',
  padding: 10,
  cornerRadius: 4,
  titleFont: { family: 'Consolas,monospace', size: 12, weight: 'bold' },
  bodyFont:  { family: 'Consolas,monospace', size: 11 },
  displayColors: true,
  boxWidth: 10,
  boxHeight: 10,
  callbacks: {
    label: function(ctx) {
      var v = ctx.parsed.y !== undefined ? ctx.parsed.y : ctx.parsed;
      return ' ' + ctx.dataset.label + ': ' + (v >= 0 ? '+' : '') + v.toFixed(3) + ' km²';
    }
  }
};

var chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { labels: { color: '#c5d8e8', font: { family: 'Consolas,monospace', size: 11 }, boxWidth: 12 } },
    tooltip: tooltipPlugin
  },
  scales: {
    x: { ticks: { color: '#4e6880', font: { size: 11 } }, grid: { color: '#1a2e40' } },
    y: { ticks: { color: '#4e6880', font: { size: 11 },
           callback: function(v) { return v.toFixed(2) + ' km²'; }
         },
         grid: { color: '#1a2e40' }
    }
  }
};

// 面积趋势 tooltip 额外显示变化量
var areaTooltipCallbacks = Object.assign({}, tooltipPlugin.callbacks, {
  afterBody: function(ctxArr) {
    var i = ctxArr[0].dataIndex;
    if (i === 0) return ['  基准时相'];
    var delta = areas[i] - areas[i - 1];
    var sign  = delta >= 0 ? '+' : '';
    return ['  较上期: ' + sign + delta.toFixed(3) + ' km²'];
  }
});

// 面积趋势
new Chart($('areaChart'), {
  type: 'line',
  data: {
    labels: D.labels,
    datasets: [{
      label: '水域面积 (km²)',
      data: areas,
      borderColor: '#00c8e8',
      backgroundColor: 'rgba(0,200,232,.15)',
      fill: true, pointRadius: 5, pointBackgroundColor: '#00c8e8',
      pointHoverRadius: 7,
      tension: 0.3, borderWidth: 2
    }]
  },
  options: Object.assign({}, chartDefaults, {
    plugins: Object.assign({}, chartDefaults.plugins, {
      tooltip: Object.assign({}, tooltipPlugin, { callbacks: areaTooltipCallbacks })
    })
  })
});

// 变化量柱图
var dLabels = D.labels.slice(1).map(function(_, i) { return D.labels[i] + ' → ' + D.labels[i + 1]; });
var dVals   = D.labels.slice(1).map(function(_, i) { return areas[i + 1] - areas[i]; });
var newVals = (D.change_stats || []).map(function(c) { return c.new || 0; });
var recVals = (D.change_stats || []).map(function(c) { return -(c.receding || 0); });

new Chart($('deltaChart'), {
  type: 'bar',
  data: {
    labels: dLabels.length ? dLabels : ['（单时相，无对比）'],
    datasets: [
      {
        label: '净变化 (km²)',
        data: dLabels.length ? dVals : [0],
        backgroundColor: dVals.map(function(v) { return v > 0 ? 'rgba(255,77,77,.75)' : 'rgba(34,232,122,.75)'; }),
        borderColor:     dVals.map(function(v) { return v > 0 ? '#ff4d4d' : '#22e87a'; }),
        borderWidth: 1
      },
      {
        label: '新增 (km²)',
        data: newVals,
        backgroundColor: 'rgba(34,232,122,.4)',
        borderColor: '#22e87a', borderWidth: 1
      },
      {
        label: '消退（取反）(km²)',
        data: recVals,
        backgroundColor: 'rgba(255,77,77,.4)',
        borderColor: '#ff4d4d', borderWidth: 1
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#c5d8e8', font: { family: 'Consolas,monospace', size: 11 }, boxWidth: 12 } },
      tooltip: tooltipPlugin
    },
    scales: {
      x: { ticks: { color: '#4e6880', font: { size: 10 } }, grid: { color: '#1a2e40' } },
      y: { ticks: { color: '#4e6880', font: { size: 11 },
             callback: function(v) { return v.toFixed(2) + ' km²'; }
           },
           grid: { color: '#1a2e40' }
      }
    }
  }
});

// 变化图卡片
if (D.change_images && D.change_images.length) {
  $('change-section').style.display = 'block';
  var cc = $('change-cards');
  D.change_images.forEach(function(img, i) {
    var cs = (D.change_stats && D.change_stats[i]) || {};
    var el = document.createElement('div');
    el.className = 'change-card';
    el.innerHTML =
      '<img src="' + img + '" alt="change ' + (i + 1) + '">' +
      '<div class="change-footer">' +
        '<div class="change-title">T' + (i + 1) + ' ' + D.labels[i] + ' → T' + (i + 2) + ' ' + D.labels[i + 1] + '</div>' +
        '<div class="change-stats">' +
          '<div class="cs-item"><div class="cs-dot dot-p"></div><div class="cs-val">' + (cs.persistent || 0).toFixed(2) + '</div><div class="cs-lbl">持续</div></div>' +
          '<div class="cs-item"><div class="cs-dot dot-r"></div><div class="cs-val">' + (cs.receding  || 0).toFixed(2) + '</div><div class="cs-lbl">消退</div></div>' +
          '<div class="cs-item"><div class="cs-dot dot-n"></div><div class="cs-val">' + (cs.new       || 0).toFixed(2) + '</div><div class="cs-lbl">新增</div></div>' +
        '</div>' +
      '</div>';
    cc.appendChild(el);
  });
}

// 滑动对比
var segImgs = D.seg_images || D.result_images;
var pairs   = [];
for (var si = 0; si < segImgs.length - 1; si++) {
  pairs.push({ left: si, right: si + 1, key: si + '_' + (si + 1) });
}
if (segImgs.length >= 3) {
  pairs.push({ left: 0, right: segImgs.length - 1, key: '0_last' });
}

if (pairs.length) {
  $('slider-section').style.display = 'block';
  var tabsEl = $('slider-tabs');
  var currentPair = null;
  var sliderRatio = 0.5;

  var applyRatio = function(ratio) {
    sliderRatio = Math.max(0, Math.min(ratio, 1));
    var pct = (sliderRatio * 100).toFixed(1);
    $('sliderOverlay').style.width = pct + '%';
    $('sliderLine').style.left     = pct + '%';
    $('slider-ratio').textContent  = Math.round(sliderRatio * 100) + '% / ' + Math.round((1 - sliderRatio) * 100) + '%';
    // 实时面积差浮窗
    if (currentPair !== null) {
      var aL = areas[currentPair.left]  || 0;
      var aR = areas[currentPair.right] || 0;
      var d  = aR - aL;
      var badge = $('sliderDelta');
      badge.style.display = 'block';
      badge.textContent   = '净变化 ' + (d >= 0 ? '+' : '') + d.toFixed(3) + ' km²  ·  左 ' + aL.toFixed(3) + '  右 ' + aR.toFixed(3);
    }
  };

  var setSliderPair = function(pair) {
    currentPair = pair;
    $('sliderBg').src  = segImgs[pair.right];
    $('sliderFg').src  = segImgs[pair.left];
    $('sliderFg').style.width = $('sliderContainer').offsetWidth + 'px';
    $('lblLeft').textContent  = D.labels[pair.left];
    $('lblRight').textContent = D.labels[pair.right];
    var aL    = (areas[pair.left]  || 0).toFixed(3);
    var aR    = (areas[pair.right] || 0).toFixed(3);
    var delta = (areas[pair.right] || 0) - (areas[pair.left] || 0);
    $('slider-hint-text').textContent =
      '← 左: ' + D.labels[pair.left]  + ' (' + aL + ' km²)' +
      '  |  右: ' + D.labels[pair.right] + ' (' + aR + ' km²)' +
      '  |  变化: ' + fmt(delta) + ' km²';
    for (var ti = 0; ti < tabsEl.children.length; ti++) {
      tabsEl.children[ti].classList.remove('active');
    }
    var tab = document.getElementById('st_' + pair.key);
    if (tab) tab.classList.add('active');
    applyRatio(0.5);
  };

  pairs.forEach(function(pair) {
    var t = document.createElement('div');
    t.className   = 'slider-tab';
    t.id          = 'st_' + pair.key;
    t.textContent = D.labels[pair.left] + ' vs ' + D.labels[pair.right];
    t.onclick     = (function(p) { return function() { setSliderPair(p); }; })(pair);
    tabsEl.appendChild(t);
  });

  var moveSlider = function(clientX) {
    var rect  = $('sliderContainer').getBoundingClientRect();
    applyRatio((clientX - rect.left) / rect.width);
  };

  var drag = false;
  var sc   = $('sliderContainer');
  sc.addEventListener('mousedown',   function(e) { drag = true; moveSlider(e.clientX); sc.focus(); });
  window.addEventListener('mouseup', function()  { drag = false; });
  window.addEventListener('mousemove', function(e) { if (drag) moveSlider(e.clientX); });

  // 触摸：单指滑动，防止页面滚动时误触
  var touchStartX = null;
  sc.addEventListener('touchstart', function(e) {
    touchStartX = e.touches[0].clientX;
    moveSlider(e.touches[0].clientX);
  }, { passive: true });
  sc.addEventListener('touchmove', function(e) {
    if (Math.abs(e.touches[0].clientX - touchStartX) > 5) {
      moveSlider(e.touches[0].clientX);
    }
  }, { passive: true });

  // 键盘控制（需先点击获焦）
  sc.setAttribute('tabindex', '0');
  sc.setAttribute('aria-label', '滑动对比器，使用方向键调整');
  sc.addEventListener('keydown', function(e) {
    var step = e.shiftKey ? 0.1 : 0.02;
    if      (e.key === 'ArrowLeft')  { applyRatio(sliderRatio - step); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { applyRatio(sliderRatio + step); e.preventDefault(); }
    else if (e.key === 'Home')       { applyRatio(0);   e.preventDefault(); }
    else if (e.key === 'End')        { applyRatio(1);   e.preventDefault(); }
    else if (e.key === ' ')          { applyRatio(0.5); e.preventDefault(); }
  });

  $('sliderBg').onload = function() { $('sliderFg').style.width = $('sliderContainer').offsetWidth + 'px'; };
  window.addEventListener('resize',  function() { $('sliderFg').style.width = $('sliderContainer').offsetWidth + 'px'; });

  setSliderPair(pairs[0]);
}

// 变化矩阵
var mb = $('matrix-body');
(D.change_stats || []).forEach(function(cs, i) {
  var net = (cs.new || 0) - (cs.receding || 0);
  var tr  = document.createElement('tr');
  tr.innerHTML =
    '<td style="color:#c5d8e8">T' + (i + 1) + '→T' + (i + 2) +
      '<br><small style="color:#4e6880">' + D.labels[i] + ' → ' + D.labels[i + 1] + '</small></td>' +
    '<td><span style="color:#1e78dc">■</span> ' + (cs.persistent || 0).toFixed(3) + '</td>' +
    '<td><span style="color:#dc3c3c">■</span> ' + (cs.receding   || 0).toFixed(3) + '</td>' +
    '<td><span style="color:#1ec850">■</span> ' + (cs.new        || 0).toFixed(3) + '</td>' +
    '<td class="' + clsCh(net) + '">' + fmt(net) + '</td>';
  mb.appendChild(tr);
});
if (!(D.change_stats && D.change_stats.length)) {
  mb.innerHTML = '<tr><td colspan="5" style="color:#4e6880;text-align:center;padding:20px">' +
    '单时相模式，无变化统计</td></tr>';
}

// 质量指标进度条（自动识别自检模式 vs 标注对比模式）
var avgMetric = function(k) {
  return D.quality.reduce(function(s, m) { return s + (m[k] || 0); }, 0) / Math.max(D.quality.length, 1);
};
var isSelfVal = D.quality.length > 0 && D.quality[0].reference_source === 'self_validation';

// 根据模式配置不同的指标标签和说明
var metricsGt = [
  { name: 'IoU',    key: 'IoU',           color: '#00c8e8', tip: '交并比' },
  { name: '精确率',  key: 'Precision',     color: '#34d399', tip: '预测水体中正确的比例' },
  { name: '召回率',  key: 'Recall',        color: '#60a5fa', tip: '实际水体被检测到的比例' },
  { name: 'F1',     key: 'F1_Score',      color: '#a78bfa', tip: '精确率与召回率的调和均值' },
  { name: '面积准确', key: 'area_accuracy', color: '#f59e0b', tip: '预测面积与标注面积的接近程度' }
];
var metricsSv = [
  { name: '综合评分',  key: 'IoU',           color: '#00c8e8', tip: '覆盖合理性、概率一致性、连通性的加权综合' },
  { name: '概率一致性', key: 'Precision',     color: '#34d399', tip: '水体像素概率均值高于非水体的程度' },
  { name: '覆盖合理性', key: 'Recall',        color: '#60a5fa', tip: '水体占有效区域的比例是否合理' },
  { name: '综合F1',   key: 'F1_Score',      color: '#a78bfa', tip: '概率一致性与覆盖合理性的调和均值' },
  { name: '连通性',   key: 'area_accuracy', color: '#f59e0b', tip: '连通组件数量是否正常（组件过多说明噪点多）' }
];
var metrics = isSelfVal ? metricsSv : metricsGt;

if (isSelfVal) {
  $('quality-panel-title').textContent = '分割自检质量评估';
  $('quality-mode-chip').textContent   = '自检模式';
  $('quality-mode-chip').style.cssText = 'background:rgba(245,166,35,.15);color:var(--amber);padding:2px 8px;border-radius:3px;font-size:10px';
  var noteEl = $('quality-note');
  noteEl.style.display = 'block';
  noteEl.textContent   = '当前无人工标注图，指标基于模型自检（概率一致性、覆盖合理性、连通性），不代表与真实水体边界的对比精度。如需标准 IoU/F1 评估，请提供标注掩膜文件。';
} else {
  $('quality-panel-title').textContent = '平均分割质量指标';
  $('quality-mode-chip').textContent   = '标注对比';
  $('quality-mode-chip').style.cssText = 'background:var(--cyan-dim);color:var(--cyan);padding:2px 8px;border-radius:3px;font-size:10px';
}

var barEl = $('metric-bars');
metrics.forEach(function(m) {
  var v   = avgMetric(m.key);
  var pct = (v * 100).toFixed(1);
  var row = document.createElement('div');
  row.className = 'mb-row';
  row.title = m.tip;
  row.innerHTML =
    '<div class="mb-name" style="width:88px;cursor:help" title="' + m.tip + '">' + m.name + '</div>' +
    '<div class="mb-track"><div class="mb-fill" style="width:' + pct + '%;background:' + m.color + '"></div></div>' +
    '<div class="mb-val">' + pct + '%</div>';
  barEl.appendChild(row);
});

// 决策建议 — 三栏内容来自不同字段，用列表渲染
function renderAdviceText(el, text) {
  el.textContent = text || '暂无';
}
function renderAdviceList(el, items) {
  if (!items || !items.length) { el.textContent = '暂无'; return; }
  var ol = document.createElement('ol');
  ol.style.cssText = 'padding-left:16px;margin:0;color:var(--white);font-size:12px;line-height:1.9';
  items.forEach(function(item) {
    var li = document.createElement('li');
    li.textContent = item;
    li.style.marginBottom = '4px';
    ol.appendChild(li);
  });
  el.appendChild(ol);
}

// 专家态势研判 — 段落文字（来自 expert_opinion）
renderAdviceText($('adv-expert'), ds.expert_opinion);

// 预警响应建议 — 编号列表（来自 warning_recommendations）
var warnRecs = ds.warning_recommendations || [];
// 若与 response_actions 完全相同（旧版数据兼容），则降级展示提示
var actRecs  = ds.response_actions || [];
var warnIsSameAsAct = JSON.stringify(warnRecs) === JSON.stringify(actRecs);
if (warnIsSameAsAct && warnRecs.length) {
  // 旧版数据：两个字段相同，预警建议栏展示前半部分，措施栏展示后半部分
  renderAdviceList($('adv-warn'), warnRecs.slice(0, Math.ceil(warnRecs.length / 2)));
  renderAdviceList($('adv-act'),  actRecs.slice(Math.ceil(actRecs.length / 2)));
} else {
  renderAdviceList($('adv-warn'), warnRecs);
  renderAdviceList($('adv-act'),  actRecs);
}

// 如果专家意见为空但有 expert_model_output，尝试从中提取
if (!ds.expert_opinion && D.report && D.report.expert_model_output) {
  var emo = D.report.expert_model_output;
  var riskText = emo.risk || emo.trend || '';
  renderAdviceText($('adv-expert'), riskText);
}

// ── Lightbox ──────────────────────────────────────────────────────────────────
var lbImages = [];  // { src, caption }
var lbIndex  = 0;

function lbRegister(imgEl, caption) {
  var idx = lbImages.length;
  lbImages.push({ src: imgEl.src || imgEl.getAttribute('src'), caption: caption || '' });
  imgEl.addEventListener('click', function() { lbOpen(idx); });
}

function lbOpen(idx) {
  lbIndex = idx;
  lbRender();
  $('lightbox').classList.add('open');
  document.body.style.overflow = 'hidden';
  $('lb-img').focus();
}

function lbClose() {
  $('lightbox').classList.remove('open');
  document.body.style.overflow = '';
}

function lbRender() {
  var item = lbImages[lbIndex];
  $('lb-img').src        = item.src;
  $('lb-caption').textContent = item.caption;
  $('lb-counter').textContent = (lbIndex + 1) + ' / ' + lbImages.length;
  $('lb-prev').style.visibility = lbImages.length > 1 ? 'visible' : 'hidden';
  $('lb-next').style.visibility = lbImages.length > 1 ? 'visible' : 'hidden';
}

$('lb-close').onclick = lbClose;
$('lb-prev').onclick  = function() { lbIndex = (lbIndex - 1 + lbImages.length) % lbImages.length; lbRender(); };
$('lb-next').onclick  = function() { lbIndex = (lbIndex + 1) % lbImages.length; lbRender(); };
$('lightbox').addEventListener('click', function(e) { if (e.target === this) lbClose(); });
document.addEventListener('keydown', function(e) {
  if (!$('lightbox').classList.contains('open')) return;
  if (e.key === 'Escape')      lbClose();
  if (e.key === 'ArrowLeft')   { lbIndex = (lbIndex - 1 + lbImages.length) % lbImages.length; lbRender(); }
  if (e.key === 'ArrowRight')  { lbIndex = (lbIndex + 1) % lbImages.length; lbRender(); }
});

// 注册时序缩略图（动态生成，需要 MutationObserver 捕获）
function lbRegisterTimeline() {
  document.querySelectorAll('#timeline .tl-img').forEach(function(img, i) {
    var lbl = D.labels[i] || ('T' + (i + 1));
    // 避免重复注册
    if (img.dataset.lbReg) return;
    img.dataset.lbReg = '1';
    lbRegister(img, lbl + '  ·  ' + (areas[i] || 0).toFixed(3) + ' km²');
  });
}

// 监听时序图重建（切换图层时）
var tlObs = new MutationObserver(function() { lbImages = []; lbRegisterTimeline(); lbRegisterChangeCards(); });
tlObs.observe($('timeline'), { childList: true });

// 注册变化检测图
function lbRegisterChangeCards() {
  document.querySelectorAll('#change-cards .change-card img').forEach(function(img, i) {
    if (img.dataset.lbReg) return;
    img.dataset.lbReg = '1';
    var cs = (D.change_stats && D.change_stats[i]) || {};
    var cap = D.labels[i] + ' → ' + D.labels[i + 1] +
      '  持续 ' + (cs.persistent || 0).toFixed(2) +
      '  消退 ' + (cs.receding || 0).toFixed(2) +
      '  新增 ' + (cs.new || 0).toFixed(2) + ' km²';
    lbRegister(img, cap);
  });
}

// 注册滑块背景图点击（打开对应原图）
$('sliderBg').addEventListener('click', function() {
  if (this.src) { lbImages = [{ src: this.src, caption: $('lblRight').textContent }]; lbOpen(0); }
});
$('sliderFg').addEventListener('click', function() {
  if (this.src) { lbImages = [{ src: this.src, caption: $('lblLeft').textContent }]; lbOpen(0); }
});

// 初始注册
setTimeout(function() {
  lbRegisterTimeline();
  lbRegisterChangeCards();
}, 200);
</script>
</body>
</html>"""


def build_dashboard_html(site_data):
    data_json = json.dumps(site_data, ensure_ascii=False)
    return _TEMPLATE.replace("__SITE_DATA__", data_json)


def write_dashboard(out_dir, site_data):
    html = build_dashboard_html(site_data)
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
