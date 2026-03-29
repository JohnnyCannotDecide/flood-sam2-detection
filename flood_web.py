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

@media(max-width:900px) {
  .kpi-row { grid-template-columns:repeat(3,1fr); }
  .two-col, .advice-grid { grid-template-columns:1fr; }
}
@media(max-width:600px) {
  .kpi-row { grid-template-columns:1fr 1fr; }
  .topbar  { flex-direction:column; gap:10px; align-items:flex-start; }
}
</style>
</head>
<body>

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
    <div class="section-title">时序水域 · 分割结果缩略图</div>
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
      <div class="panel-header"><span>平均分割质量指标</span></div>
      <div class="panel-body"><div class="metric-bars" id="metric-bars"></div></div>
    </div>
  </section>

  <section>
    <div class="section-title">决策建议</div>
    <div class="advice-grid">
      <div class="advice-item"><div class="advice-label">专家分析</div><div class="advice-text" id="adv-expert"></div></div>
      <div class="advice-item"><div class="advice-label">预警建议</div><div class="advice-text" id="adv-warn"></div></div>
      <div class="advice-item"><div class="advice-label">应对措施</div><div class="advice-text" id="adv-act"></div></div>
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

var levelBadge = { '\u9ad8': 'high', '\u4e2d': 'mid', '\u4f4e': 'low' };
var levelCls   = { '\u9ad8': 'up',   '\u4e2d': '',    '\u4f4e': 'down' };
$('warning-badge').textContent = '\u26a0 \u9884\u8b66\u7b49\u7ea7\uff1a' + level;
$('warning-badge').className   = 'badge badge-' + (levelBadge[level] || 'low');
$('kpi-level').textContent     = level;
$('kpi-level').className       = 'kpi-value ' + (levelCls[level] || '');
$('kpi-summary').textContent   = ra.summary || '';
$('kpi-score').textContent     = score.toFixed(3);
$('kpi-score').className       = 'kpi-value ' + (score >= 0.7 ? 'up' : score >= 0.4 ? '' : 'down');
$('kpi-delta').textContent     = fmt(netDelta) + ' km\u00b2';
$('kpi-delta').className       = 'kpi-value ' + (netDelta > 0.05 ? 'up' : netDelta < -0.05 ? 'down' : 'stable');
$('kpi-new').textContent       = '+' + totalNew.toFixed(3) + ' km\u00b2';
$('kpi-new').className         = 'kpi-value ' + (totalNew > 0.1 ? 'up' : 'stable');
$('kpi-count').textContent     = D.labels.length;

// 时序缩略图
var tl = $('timeline');
D.labels.forEach(function(lbl, i) {
  var prev  = i > 0 ? areas[i - 1] : null;
  var delta = prev !== null ? areas[i] - prev : null;
  var dCls  = delta === null ? 'first' : delta > 0.01 ? 'up' : delta < -0.01 ? 'down' : 'first';
  var dTxt  = delta === null ? '\u57fa\u51c6\u65f6\u76f8' : fmt(delta) + ' km\u00b2';
  var el    = document.createElement('div');
  el.className = 'tl-item';
  el.innerHTML =
    '<div class="tl-index">T' + (i + 1) + '</div>' +
    '<img class="tl-img" src="' + D.result_images[i] + '" alt="' + lbl + '">' +
    '<div class="tl-footer">' +
      '<div class="tl-label">' + lbl + '</div>' +
      '<div class="tl-area">' + (areas[i] || 0).toFixed(3) + ' km\u00b2</div>' +
      '<div class="tl-delta ' + dCls + '">' + dTxt + '</div>' +
    '</div>';
  tl.appendChild(el);
});

// Chart.js 公共配置
var chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { labels: { color: '#c5d8e8', font: { family: 'Consolas,monospace', size: 11 } } } },
  scales: {
    x: { ticks: { color: '#4e6880', font: { size: 11 } }, grid: { color: '#1a2e40' } },
    y: { ticks: { color: '#4e6880', font: { size: 11 } }, grid: { color: '#1a2e40' } }
  }
};

// 面积趋势
new Chart($('areaChart'), {
  type: 'line',
  data: {
    labels: D.labels,
    datasets: [{
      label: '\u6c34\u57df\u9762\u79ef (km\u00b2)',
      data: areas,
      borderColor: '#00c8e8',
      backgroundColor: 'rgba(0,200,232,.15)',
      fill: true, pointRadius: 5, pointBackgroundColor: '#00c8e8',
      tension: 0.3, borderWidth: 2
    }]
  },
  options: chartDefaults
});

// 变化量柱图
var dLabels = D.labels.slice(1).map(function(_, i) { return D.labels[i] + ' \u2192 ' + D.labels[i + 1]; });
var dVals   = D.labels.slice(1).map(function(_, i) { return areas[i + 1] - areas[i]; });
var newVals = (D.change_stats || []).map(function(c) { return c.new || 0; });
var recVals = (D.change_stats || []).map(function(c) { return -(c.receding || 0); });

new Chart($('deltaChart'), {
  type: 'bar',
  data: {
    labels: dLabels.length ? dLabels : ['\uff08\u5355\u65f6\u76f8\uff0c\u65e0\u5bf9\u6bd4\uff09'],
    datasets: [
      {
        label: '\u51c0\u53d8\u5316 (km\u00b2)',
        data: dLabels.length ? dVals : [0],
        backgroundColor: dVals.map(function(v) { return v > 0 ? 'rgba(255,77,77,.75)' : 'rgba(34,232,122,.75)'; }),
        borderColor:     dVals.map(function(v) { return v > 0 ? '#ff4d4d' : '#22e87a'; }),
        borderWidth: 1
      },
      {
        label: '\u65b0\u589e (km\u00b2)',
        data: newVals,
        backgroundColor: 'rgba(34,232,122,.4)',
        borderColor: '#22e87a', borderWidth: 1
      },
      {
        label: '\u6d88\u9000\uff08\u53d6\u53cd\uff09(km\u00b2)',
        data: recVals,
        backgroundColor: 'rgba(255,77,77,.4)',
        borderColor: '#ff4d4d', borderWidth: 1
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#c5d8e8', font: { family: 'Consolas,monospace', size: 11 } } } },
    scales: {
      x: { ticks: { color: '#4e6880', font: { size: 10 } }, grid: { color: '#1a2e40' } },
      y: { ticks: { color: '#4e6880', font: { size: 11 } }, grid: { color: '#1a2e40' } }
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
        '<div class="change-title">T' + (i + 1) + ' ' + D.labels[i] + ' \u2192 T' + (i + 2) + ' ' + D.labels[i + 1] + '</div>' +
        '<div class="change-stats">' +
          '<div class="cs-item"><div class="cs-dot dot-p"></div><div class="cs-val">' + (cs.persistent || 0).toFixed(2) + '</div><div class="cs-lbl">\u6301\u7eed</div></div>' +
          '<div class="cs-item"><div class="cs-dot dot-r"></div><div class="cs-val">' + (cs.receding  || 0).toFixed(2) + '</div><div class="cs-lbl">\u6d88\u9000</div></div>' +
          '<div class="cs-item"><div class="cs-dot dot-n"></div><div class="cs-val">' + (cs.new       || 0).toFixed(2) + '</div><div class="cs-lbl">\u65b0\u589e</div></div>' +
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

  var setSliderPair = function(pair) {
    $('sliderBg').src  = segImgs[pair.left];
    $('sliderFg').src  = segImgs[pair.right];
    $('sliderFg').style.width = $('sliderContainer').offsetWidth + 'px';
    $('lblLeft').textContent  = D.labels[pair.left];
    $('lblRight').textContent = D.labels[pair.right];
    var aL    = (areas[pair.left]  || 0).toFixed(3);
    var aR    = (areas[pair.right] || 0).toFixed(3);
    var delta = (areas[pair.right] || 0) - (areas[pair.left] || 0);
    $('slider-hint-text').textContent =
      '\u2190 \u5de6: ' + D.labels[pair.left]  + ' (' + aL + ' km\u00b2)' +
      '  |  \u53f3: ' + D.labels[pair.right] + ' (' + aR + ' km\u00b2)' +
      '  |  \u53d8\u5316: ' + fmt(delta) + ' km\u00b2';
    for (var ti = 0; ti < tabsEl.children.length; ti++) {
      tabsEl.children[ti].classList.remove('active');
    }
    var tab = document.getElementById('st_' + pair.key);
    if (tab) tab.classList.add('active');
    $('sliderOverlay').style.width = '50%';
    $('sliderLine').style.left     = '50%';
  };

  pairs.forEach(function(pair) {
    var t = document.createElement('div');
    t.className  = 'slider-tab';
    t.id         = 'st_' + pair.key;
    t.textContent = D.labels[pair.left] + ' vs ' + D.labels[pair.right];
    t.onclick    = (function(p) { return function() { setSliderPair(p); }; })(pair);
    tabsEl.appendChild(t);
  });

  var moveSlider = function(clientX) {
    var rect  = $('sliderContainer').getBoundingClientRect();
    var ratio = Math.max(0, Math.min((clientX - rect.left) / rect.width, 1));
    $('sliderOverlay').style.width = (ratio * 100) + '%';
    $('sliderLine').style.left     = (ratio * 100) + '%';
    $('slider-ratio').textContent  = Math.round(ratio * 100) + '% / ' + Math.round((1 - ratio) * 100) + '%';
  };

  var drag = false;
  var sc   = $('sliderContainer');
  sc.addEventListener('mousedown',   function(e) { drag = true; moveSlider(e.clientX); });
  window.addEventListener('mouseup', function()  { drag = false; });
  window.addEventListener('mousemove', function(e) { if (drag) moveSlider(e.clientX); });
  sc.addEventListener('touchstart', function(e) { moveSlider(e.touches[0].clientX); }, { passive: true });
  sc.addEventListener('touchmove',  function(e) { moveSlider(e.touches[0].clientX); }, { passive: true });

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
    '<td style="color:#c5d8e8">T' + (i + 1) + '\u2192T' + (i + 2) +
      '<br><small style="color:#4e6880">' + D.labels[i] + ' \u2192 ' + D.labels[i + 1] + '</small></td>' +
    '<td><span style="color:#1e78dc">\u25a0</span> ' + (cs.persistent || 0).toFixed(3) + '</td>' +
    '<td><span style="color:#dc3c3c">\u25a0</span> ' + (cs.receding   || 0).toFixed(3) + '</td>' +
    '<td><span style="color:#1ec850">\u25a0</span> ' + (cs.new        || 0).toFixed(3) + '</td>' +
    '<td class="' + clsCh(net) + '">' + fmt(net) + '</td>';
  mb.appendChild(tr);
});
if (!(D.change_stats && D.change_stats.length)) {
  mb.innerHTML = '<tr><td colspan="5" style="color:#4e6880;text-align:center;padding:20px">' +
    '\u5355\u65f6\u76f8\u6a21\u5f0f\uff0c\u65e0\u53d8\u5316\u7edf\u8ba1</td></tr>';
}

// 质量指标进度条
var avgMetric = function(k) {
  return D.quality.reduce(function(s, m) { return s + (m[k] || 0); }, 0) / Math.max(D.quality.length, 1);
};
var metrics = [
  { name: 'IoU',      key: 'IoU',           color: '#00c8e8' },
  { name: '\u7cbe\u786e\u7387', key: 'Precision',     color: '#34d399' },
  { name: '\u53ec\u56de\u7387', key: 'Recall',        color: '#60a5fa' },
  { name: 'F1',       key: 'F1_Score',      color: '#a78bfa' },
  { name: '\u9762\u79ef\u51c6\u786e', key: 'area_accuracy', color: '#f59e0b' }
];
var barEl = $('metric-bars');
metrics.forEach(function(m) {
  var v   = avgMetric(m.key);
  var pct = (v * 100).toFixed(1);
  var row = document.createElement('div');
  row.className = 'mb-row';
  row.innerHTML =
    '<div class="mb-name">'  + m.name + '</div>' +
    '<div class="mb-track"><div class="mb-fill" style="width:' + pct + '%;background:' + m.color + '"></div></div>' +
    '<div class="mb-val">' + pct + '%</div>';
  barEl.appendChild(row);
});

// 决策建议
$('adv-expert').textContent = ds.expert_opinion || '\u6682\u65e0';
$('adv-warn').textContent   = (ds.warning_recommendations || []).join('\n') || '\u6682\u65e0';
$('adv-act').textContent    = (ds.response_actions        || []).join('\n') || '\u6682\u65e0';
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