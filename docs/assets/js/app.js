/*
 * AI/ML Interview Map: study status per page, flashcard drill, progress
 * dashboard and the "continue" widget on the home page.
 *
 * Everything is stored in this browser's localStorage under the "aiml-map:"
 * prefix. Use Export / Import on the My Progress page to move it between
 * browsers or devices.
 */
(function () {
  "use strict";

  var PREFIX = "aiml-map:";
  var STATUS = ["Not started", "Learning", "Confident"];
  var ACTS = {
    "0": "Foundations",
    "1": "Act I · Learn from features",
    "2": "Act II · Learn the features",
    "3": "Act III · Attention and scale",
    "4": "Act IV · Align, ground, act",
    "5": "Production"
  };

  function load(key, fallback) {
    try {
      var raw = localStorage.getItem(PREFIX + key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (e) {
      return fallback;
    }
  }
  function save(key, value) {
    try {
      localStorage.setItem(PREFIX + key, JSON.stringify(value));
    } catch (e) {
      /* storage unavailable: the page still works, it just won't remember */
    }
  }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // The site root, resolved once from the first page loaded. Instant navigation
  // keeps this script alive across page changes, so the value stays correct.
  var ROOT = (function () {
    try {
      var cfg = JSON.parse(document.getElementById("__config").textContent);
      return new URL((cfg.base || ".") + "/", location.href).href;
    } catch (e) {
      return new URL("./", location.href).href;
    }
  })();

  var cache = {};
  function getJSON(name) {
    if (!cache[name]) {
      cache[name] = fetch(ROOT + "assets/data/" + name).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
    }
    return cache[name];
  }

  function pageStatus(url) {
    var v = load("page:" + url, 0);
    return v === 1 || v === 2 ? v : 0;
  }
  function qaStatus(id) {
    var v = load("qa:" + id, 0);
    return v === 1 || v === 2 ? v : 0;
  }
  function statusButtons(url) {
    var s = pageStatus(url);
    return (
      '<span class="sb-seg" role="group" aria-label="Your status for this page">' +
      STATUS.map(function (label, i) {
        return (
          '<button type="button" data-st="' + i + '" aria-pressed="' + (i === s) + '">' +
          label + "</button>"
        );
      }).join("") +
      "</span>"
    );
  }
  function renderMath(el) {
    if (window.renderMathInElement && window.AIML_MATH_OPTIONS) {
      window.renderMathInElement(el, window.AIML_MATH_OPTIONS);
    }
  }

  /* ---------- study bar under each page title ---------- */
  function initStudyBars(root) {
    root.querySelectorAll(".study-bar").forEach(function (bar) {
      var url = bar.getAttribute("data-page");
      var mins = bar.getAttribute("data-minutes");
      var qn = +bar.getAttribute("data-questions");
      var act = bar.getAttribute("data-act");
      var prio = bar.getAttribute("data-priority");
      bar.innerHTML =
        '<span class="sb-meta">' +
        (ACTS[act] ? '<span class="act act-' + esc(act) + '">' + ACTS[act] + "</span> · " : "") +
        (prio ? "Interview priority: <strong>" + esc(prio) + "</strong> · " : "") +
        esc(mins) + " min read" +
        (qn ? " · " + qn + " interview question" + (qn === 1 ? "" : "s") : "") +
        "</span>" +
        '<span class="sb-status"><span class="sb-label">Your status</span>' +
        statusButtons(url) + "</span>";
    });
  }

  document.addEventListener("click", function (e) {
    var btn = e.target.closest && e.target.closest("[data-st]");
    if (!btn) return;
    var holder = btn.closest("[data-page]");
    if (!holder) return;
    var url = holder.getAttribute("data-page");
    var s = +btn.getAttribute("data-st");
    save("page:" + url, s);
    document.querySelectorAll('[data-page="' + CSS.escape(url) + '"] [data-st]').forEach(function (b) {
      b.setAttribute("aria-pressed", String(+b.getAttribute("data-st") === s));
    });
    var dash = document.getElementById("progress-dashboard");
    if (dash && dash.contains(btn)) refreshDashboardTotals();
  });

  /* ---------- "continue where you left off" on the home page ---------- */
  function initContinue(root) {
    var host = root.querySelector("#continue-widget");
    if (!host) return;
    getJSON("pages.json").then(function (pages) {
      var study = pages.filter(function (p) { return p.url; });
      var learning = study.filter(function (p) { return pageStatus(p.url) === 1; });
      var next = learning[0] || study.filter(function (p) { return pageStatus(p.url) === 0; })[0];
      var done = study.filter(function (p) { return pageStatus(p.url) === 2; }).length;
      if (!next) {
        host.innerHTML = '<p class="cw-title">You marked every study page Confident.</p>' +
          '<p>Keep it fresh with the <a href="' + ROOT + 'practice/flashcards/">Flashcard Drill</a>.</p>';
        return;
      }
      host.innerHTML =
        '<p class="cw-kicker">' + (done || learning.length ? "Continue your path" : "Start your path") + "</p>" +
        '<p class="cw-title"><a href="' + ROOT + esc(next.url) + '">' + esc(next.title) + "</a></p>" +
        '<p class="cw-sub">' + esc(next.section) + " · " + next.minutes + " min" + (next.questions ? " · " + next.questions + " questions" : "") +
        " · " + done + " of " + study.length + " pages Confident</p>";
    }).catch(function () { host.hidden = true; });
  }

  /* ---------- flashcard drill ---------- */
  function initDrill(root) {
    var host = root.querySelector("#drill");
    if (!host || host.getAttribute("data-ready")) return;
    host.setAttribute("data-ready", "1");
    getJSON("qa.json").then(function (qa) {
      buildDrill(host, qa);
    }).catch(function () {
      host.innerHTML = '<p class="drill-error">The question bank could not be loaded. ' +
        "If you are running the site locally, use <code>mkdocs serve</code> and reload.</p>";
    });
  }

  function buildDrill(host, qa) {
    var sections = [];
    qa.forEach(function (q) { if (sections.indexOf(q.section) < 0) sections.push(q.section); });
    host.innerHTML =
      '<div class="drill-controls">' +
        '<label for="drill-section">Section</label>' +
        '<select id="drill-section"><option value="all">All sections (' + qa.length + " questions)</option>" +
        sections.map(function (s) {
          var n = qa.filter(function (q) { return q.section === s; }).length;
          return '<option value="' + esc(s) + '">' + esc(s) + " (" + n + ")</option>";
        }).join("") + "</select>" +
        '<label for="drill-page">Page</label><select id="drill-page"></select>' +
        '<label class="drill-check" for="drill-weak"><input type="checkbox" id="drill-weak" checked> ' +
        "Ask unseen and “review again” questions more often</label>" +
      "</div>" +
      '<div class="drill-stats" id="drill-stats" aria-live="polite"></div>' +
      '<div class="drill-card">' +
        '<div class="drill-meta" id="drill-meta"></div>' +
        '<div class="drill-q" id="drill-q"></div>' +
        '<div class="drill-a" id="drill-a" hidden></div>' +
        '<div class="drill-actions">' +
          '<button type="button" class="md-button md-button--primary" id="drill-show">Show answer</button>' +
          '<button type="button" class="md-button drill-again" data-mark="1">Review again</button>' +
          '<button type="button" class="md-button drill-known" data-mark="2">I knew it</button>' +
          '<button type="button" class="md-button" id="drill-skip">Skip</button>' +
        "</div>" +
        '<p class="drill-keys">Keyboard: <kbd>Space</kbd> show answer · <kbd>1</kbd> review again · ' +
        "<kbd>2</kbd> I knew it · <kbd>3</kbd> skip</p>" +
      "</div>";

    var secSel = host.querySelector("#drill-section");
    var pageSel = host.querySelector("#drill-page");
    var weak = host.querySelector("#drill-weak");
    var qEl = host.querySelector("#drill-q");
    var aEl = host.querySelector("#drill-a");
    var metaEl = host.querySelector("#drill-meta");
    var statsEl = host.querySelector("#drill-stats");
    var showBtn = host.querySelector("#drill-show");
    var current = null;

    var saved = load("drill:filters", {});
    if (saved.section && sections.indexOf(saved.section) >= 0) secSel.value = saved.section;
    if (typeof saved.weak === "boolean") weak.checked = saved.weak;

    function fillPages() {
      var sec = secSel.value;
      var pages = [];
      qa.forEach(function (q) {
        if ((sec === "all" || q.section === sec) && !pages.some(function (p) { return p.url === q.url; })) {
          pages.push({ url: q.url, title: q.page });
        }
      });
      pageSel.innerHTML = '<option value="all">All pages</option>' + pages.map(function (p) {
        return '<option value="' + esc(p.url) + '">' + esc(p.title) + "</option>";
      }).join("");
      if (saved.page && pages.some(function (p) { return p.url === saved.page; })) pageSel.value = saved.page;
      saved.page = null;
    }
    function pool() {
      var sec = secSel.value, pg = pageSel.value;
      return qa.filter(function (q) {
        return (sec === "all" || q.section === sec) && (pg === "all" || q.url === pg);
      });
    }
    function stats() {
      var c = pool(), known = 0, again = 0;
      c.forEach(function (q) { var s = qaStatus(q.id); if (s === 2) known++; else if (s === 1) again++; });
      statsEl.innerHTML =
        '<span class="ds ds-known">' + known + " known</span>" +
        '<span class="ds ds-again">' + again + " to review</span>" +
        '<span class="ds ds-new">' + (c.length - known - again) + " unseen</span>" +
        '<span class="ds-total">of ' + c.length + " in this set</span>";
    }
    function pick() {
      var c = pool();
      if (!c.length) {
        current = null;
        qEl.innerHTML = "No questions in this selection.";
        aEl.hidden = true;
        metaEl.innerHTML = "";
        stats();
        return;
      }
      var choice = null;
      for (var tries = 0; tries < 8; tries++) {
        if (weak.checked) {
          var weights = c.map(function (q) { return [3, 6, 1][qaStatus(q.id)]; });
          var total = weights.reduce(function (a, b) { return a + b; }, 0);
          var r = Math.random() * total;
          choice = c[c.length - 1];
          for (var i = 0; i < c.length; i++) {
            r -= weights[i];
            if (r <= 0) { choice = c[i]; break; }
          }
        } else {
          choice = c[Math.floor(Math.random() * c.length)];
        }
        if (!current || choice !== current || c.length === 1) break;
      }
      current = choice;
      var s = qaStatus(current.id);
      metaEl.innerHTML =
        '<span class="dm-section">' + esc(current.section) + "</span>" +
        '<a href="' + ROOT + esc(current.url) + '">' + esc(current.page) + "</a>" +
        (s ? '<span class="dm-state dm-' + s + '">' + (s === 2 ? "Known" : "Review again") + "</span>" : "");
      qEl.innerHTML = current.q;
      aEl.innerHTML = current.a;
      aEl.hidden = true;
      showBtn.textContent = "Show answer";
      renderMath(qEl);
      stats();
    }
    function persistFilters() {
      save("drill:filters", { section: secSel.value, page: pageSel.value, weak: weak.checked });
    }

    showBtn.addEventListener("click", function () {
      if (!current) return;
      aEl.hidden = !aEl.hidden;
      showBtn.textContent = aEl.hidden ? "Show answer" : "Hide answer";
      if (!aEl.hidden) renderMath(aEl);
    });
    host.querySelectorAll("[data-mark]").forEach(function (b) {
      b.addEventListener("click", function () {
        if (!current) return;
        save("qa:" + current.id, +b.getAttribute("data-mark"));
        pick();
      });
    });
    host.querySelector("#drill-skip").addEventListener("click", pick);
    secSel.addEventListener("change", function () { fillPages(); persistFilters(); pick(); });
    pageSel.addEventListener("change", function () { persistFilters(); pick(); });
    weak.addEventListener("change", function () { persistFilters(); pick(); });

    fillPages();
    pick();
  }

  document.addEventListener("keydown", function (e) {
    var host = document.getElementById("drill");
    if (!host || !host.getAttribute("data-ready") || !host.querySelector("#drill-show")) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var tag = (e.target && e.target.tagName ? e.target.tagName : "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea" || (e.target && e.target.isContentEditable)) return;
    if (e.key === " " && tag !== "button" && tag !== "a") {
      e.preventDefault();
      host.querySelector("#drill-show").click();
    } else if (e.key === "1") {
      host.querySelector('[data-mark="1"]').click();
    } else if (e.key === "2") {
      host.querySelector('[data-mark="2"]').click();
    } else if (e.key === "3") {
      host.querySelector("#drill-skip").click();
    }
  });

  /* ---------- progress dashboard ---------- */
  var dashData = null;

  function initDashboard(root) {
    var host = root.querySelector("#progress-dashboard");
    if (!host || host.getAttribute("data-ready")) return;
    host.setAttribute("data-ready", "1");
    Promise.all([getJSON("pages.json"), getJSON("qa.json").catch(function () { return []; })]).then(function (res) {
      dashData = { pages: res[0].filter(function (p) { return p.url; }), qa: res[1] };
      renderDashboard(host);
    }).catch(function () {
      host.innerHTML = '<p class="drill-error">Progress data could not be loaded. Run <code>mkdocs serve</code> and reload.</p>';
    });
  }

  function qaCounts(list) {
    var known = 0, again = 0;
    list.forEach(function (q) { var s = qaStatus(q.id); if (s === 2) known++; else if (s === 1) again++; });
    return { known: known, again: again, total: list.length };
  }

  function refreshDashboardTotals() {
    var host = document.getElementById("progress-dashboard");
    if (!host || !dashData) return;
    var pages = dashData.pages;
    var conf = 0, learn = 0;
    pages.forEach(function (p) { var s = pageStatus(p.url); if (s === 2) conf++; else if (s === 1) learn++; });
    var qc = qaCounts(dashData.qa);
    var el = host.querySelector("#pd-summary");
    if (!el) return;
    el.innerHTML =
      '<div class="pd-tile"><span class="pd-num">' + conf + "<small>/" + pages.length + "</small></span>" +
      '<span class="pd-lbl">pages Confident</span>' + bar(conf, learn, pages.length) + "</div>" +
      '<div class="pd-tile"><span class="pd-num">' + learn + '</span><span class="pd-lbl">pages Learning</span></div>' +
      '<div class="pd-tile"><span class="pd-num">' + qc.known + "<small>/" + qc.total + "</small></span>" +
      '<span class="pd-lbl">flashcards known</span>' + bar(qc.known, qc.again, qc.total) + "</div>" +
      '<div class="pd-tile"><span class="pd-num">' + qc.again + '</span><span class="pd-lbl">flashcards to review</span></div>';
    host.querySelectorAll("[data-sec-count]").forEach(function (s) {
      var name = s.getAttribute("data-sec-count");
      var list = pages.filter(function (p) { return p.section === name; });
      var c = list.filter(function (p) { return pageStatus(p.url) === 2; }).length;
      s.textContent = c + " of " + list.length + " Confident";
    });
  }

  function bar(a, b, total) {
    total = total || 1;
    return '<span class="pd-bar"><span class="pd-bar-a" style="width:' + (a / total * 100) + '%"></span>' +
      '<span class="pd-bar-b" style="width:' + (b / total * 100) + '%"></span></span>';
  }

  function renderDashboard(host) {
    var pages = dashData.pages;
    var sections = [];
    pages.forEach(function (p) { if (sections.indexOf(p.section) < 0) sections.push(p.section); });
    var html = '<div class="pd-summary" id="pd-summary"></div>';
    sections.forEach(function (sec) {
      html += '<h3 class="pd-sec">' + esc(sec) + ' <span data-sec-count="' + esc(sec) + '"></span></h3>' +
        '<div class="pd-list">';
      pages.filter(function (p) { return p.section === sec; }).forEach(function (p) {
        var qc = qaCounts(dashData.qa.filter(function (q) { return q.url === p.url; }));
        html += '<div class="pd-row" data-page="' + esc(p.url) + '">' +
          '<a class="pd-title" href="' + ROOT + esc(p.url) + '">' + esc(p.title) + "</a>" +
          '<span class="pd-info">' + p.minutes + " min" +
          (qc.total ? " · flashcards " + qc.known + "/" + qc.total + " known" : "") + "</span>" +
          statusButtons(p.url) + "</div>";
      });
      html += "</div>";
    });
    html +=
      '<div class="pd-tools">' +
        '<button type="button" class="md-button" id="pd-export">Export progress</button>' +
        '<label class="md-button" for="pd-import">Import progress</label>' +
        '<input type="file" id="pd-import" accept="application/json" hidden>' +
        '<button type="button" class="md-button" id="pd-reset">Reset all progress</button>' +
        '<span id="pd-msg" role="status"></span>' +
      "</div>";
    host.innerHTML = html;
    refreshDashboardTotals();

    var msg = host.querySelector("#pd-msg");
    host.querySelector("#pd-export").addEventListener("click", function () {
      var data = {};
      try {
        for (var i = 0; i < localStorage.length; i++) {
          var k = localStorage.key(i);
          if (k && k.indexOf(PREFIX) === 0) data[k] = localStorage.getItem(k);
        }
      } catch (e) { /* ignore */ }
      var blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "aiml-map-progress.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      msg.textContent = "Exported " + Object.keys(data).length + " saved items.";
    });
    host.querySelector("#pd-import").addEventListener("change", function (e) {
      var file = e.target.files && e.target.files[0];
      if (!file) return;
      file.text().then(function (text) {
        var data = JSON.parse(text), n = 0;
        Object.keys(data).forEach(function (k) {
          if (k.indexOf(PREFIX) === 0) { try { localStorage.setItem(k, data[k]); n++; } catch (err) { /* ignore */ } }
        });
        msg.textContent = "Imported " + n + " saved items.";
        host.removeAttribute("data-ready");
        initDashboard(document);
      }).catch(function () { msg.textContent = "That file is not a progress export from this site."; });
    });
    var armed = false;
    host.querySelector("#pd-reset").addEventListener("click", function (e) {
      if (!armed) {
        armed = true;
        e.target.textContent = "Click again to erase everything";
        msg.textContent = "This clears page statuses and flashcard history in this browser.";
        setTimeout(function () { armed = false; e.target.textContent = "Reset all progress"; }, 5000);
        return;
      }
      try {
        Object.keys(localStorage).forEach(function (k) { if (k.indexOf(PREFIX) === 0) localStorage.removeItem(k); });
      } catch (err) { /* ignore */ }
      host.removeAttribute("data-ready");
      initDashboard(document);
    });
  }

  /* ---------- wire up on every (instant) page load ---------- */
  function onPage() {
    initStudyBars(document);
    initContinue(document);
    initDrill(document);
    initDashboard(document);
  }
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(onPage);
  } else {
    document.addEventListener("DOMContentLoaded", onPage);
  }
})();
