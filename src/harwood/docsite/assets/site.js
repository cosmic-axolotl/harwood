// The documentation's behaviour: example tabs, copy buttons, the look buttons,
// the search, and the filter on the migration table. Plain script, no
// libraries, so that it works from file:// in any browser and in the window.
(function () {
  "use strict";
  var root = document.body.getAttribute("data-root") || "./";

  function load(key, fallback) {
    try { var v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); }
    catch (e) { return fallback; }
  }
  function save(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) {}
  }

  // ---------------------------------------------------------------- tabs
  // Choosing Python in one example chooses it in all of them, here and on the
  // next page: someone who reads Python wants to keep reading Python.
  function chooseWay(way, focus) {
    document.querySelectorAll(".example").forEach(function (example) {
      example.querySelectorAll('[role="tab"]').forEach(function (tab) {
        var on = tab.getAttribute("data-way") === way;
        tab.setAttribute("aria-selected", on ? "true" : "false");
        tab.tabIndex = on ? 0 : -1;
        if (on && focus === tab) tab.focus();
      });
      example.querySelectorAll('[role="tabpanel"]').forEach(function (panel) {
        panel.hidden = panel.getAttribute("data-way") !== way;
      });
    });
    save("harwood-way", way);
  }
  document.querySelectorAll('.example [role="tab"]').forEach(function (tab) {
    tab.addEventListener("click", function () { chooseWay(tab.getAttribute("data-way")); });
    tab.addEventListener("keydown", function (event) {
      var tabs = Array.prototype.slice.call(tab.parentNode.querySelectorAll('[role="tab"]'));
      var at = tabs.indexOf(tab), next = null;
      if (event.key === "ArrowRight") next = tabs[(at + 1) % tabs.length];
      if (event.key === "ArrowLeft") next = tabs[(at - 1 + tabs.length) % tabs.length];
      if (next) { event.preventDefault(); chooseWay(next.getAttribute("data-way"), next); }
    });
  });
  chooseWay(load("harwood-way", "terminal"));

  // ---------------------------------------------------------------- copy
  document.querySelectorAll(".code .copy").forEach(function (button) {
    button.addEventListener("click", function () {
      var text = button.parentNode.querySelector("pre").innerText
        .split("\n").map(function (line) { return line.replace(/^\$ /, ""); }).join("\n");
      var done = function () {
        button.textContent = "Copied ✓";
        setTimeout(function () { button.textContent = "Copy"; }, 1500);
      };
      if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
      else done();
    });
  });

  // ---------------------------------------------------------------- the look
  var look = load("harwood-look", {});
  function syncLook() {
    document.querySelectorAll("[data-look]").forEach(function (button) {
      var what = button.getAttribute("data-look");
      if (what === "theme") {
        var now = document.documentElement.getAttribute("data-theme");
        button.querySelector("span").textContent = now === "light" ? "light" : "dark";
      } else {
        button.setAttribute("aria-pressed", look[what] ? "true" : "false");
      }
    });
  }
  document.querySelectorAll("[data-look]").forEach(function (button) {
    button.addEventListener("click", function () {
      var what = button.getAttribute("data-look");
      if (what === "theme") {
        look.theme = document.documentElement.getAttribute("data-theme") === "light"
          ? "dark" : "light";
      } else {
        look[what] = !look[what];
      }
      save("harwood-look", look);
      window.harwoodApplyLook(look);
      syncLook();
    });
  });
  syncLook();

  // ---------------------------------------------------------------- contents (narrow)
  var menu = document.querySelector(".menu");
  if (menu) menu.addEventListener("click", function () {
    var open = document.body.classList.toggle("nav-open");
    menu.setAttribute("aria-expanded", open ? "true" : "false");
  });

  // ---------------------------------------------------------------- search
  var input = document.getElementById("search");
  var list = document.getElementById("results");
  var index = window.HARWOOD_INDEX || [];
  function score(entry, words) {
    var total = 0;
    var title = entry.t.toLowerCase(), heads = entry.h.join(" ").toLowerCase();
    var text = entry.x.toLowerCase();
    for (var i = 0; i < words.length; i++) {
      var w = words[i], here = 0;
      if (title === w) here += 40;
      if (title.indexOf(w) >= 0) here += 12;
      if (heads.indexOf(w) >= 0) here += 5;
      if (text.indexOf(w) >= 0) here += 1;
      if (!here) return 0;                     // every word has to be somewhere
      total += here;
    }
    return total;
  }
  function snippet(entry, words) {
    var text = entry.x, at = text.toLowerCase().indexOf(words[0]);
    if (at < 0) return text.slice(0, 120);
    var start = Math.max(0, at - 50);
    return (start ? "…" : "") + text.slice(start, at + 90) + "…";
  }
  function search() {
    var words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    list.innerHTML = "";
    if (!words.length) { list.hidden = true; return; }
    var found = index.map(function (e) { return [score(e, words), e]; })
      .filter(function (p) { return p[0] > 0; })
      .sort(function (a, b) { return b[0] - a[0]; }).slice(0, 12);
    if (!found.length) {
      var none = document.createElement("li");
      none.className = "none";
      none.textContent = "Nothing found for “" + input.value + "”.";
      list.appendChild(none);
    }
    found.forEach(function (pair, n) {
      var entry = pair[1], item = document.createElement("li");
      item.setAttribute("role", "option");
      item.id = "result-" + n;
      var link = document.createElement("a");
      link.href = root + entry.u;
      link.innerHTML = "<b></b> <span class='where'></span><span class='snip'></span>";
      link.querySelector("b").textContent = entry.t;
      link.querySelector(".where").textContent = entry.s;
      link.querySelector(".snip").textContent = snippet(entry, words);
      item.appendChild(link);
      list.appendChild(item);
    });
    list.hidden = false;
  }
  if (input) {
    input.addEventListener("input", search);
    input.addEventListener("keydown", function (event) {
      var links = list.querySelectorAll("a");
      if (event.key === "Enter" && links.length) { location.href = links[0].href; }
      if (event.key === "ArrowDown" && links.length) { event.preventDefault(); links[0].focus(); }
      if (event.key === "Escape") { input.value = ""; search(); }
    });
    list.addEventListener("keydown", function (event) {
      var links = Array.prototype.slice.call(list.querySelectorAll("a"));
      var at = links.indexOf(document.activeElement);
      if (event.key === "ArrowDown" && at < links.length - 1) { event.preventDefault(); links[at + 1].focus(); }
      if (event.key === "ArrowUp") { event.preventDefault(); (at > 0 ? links[at - 1] : input).focus(); }
      if (event.key === "Escape") { input.focus(); input.value = ""; search(); }
    });
    document.addEventListener("keydown", function (event) {
      var typing = /INPUT|TEXTAREA/.test((document.activeElement || {}).tagName || "");
      if (event.key === "/" && !typing) { event.preventDefault(); input.focus(); }
    });
    document.addEventListener("click", function (event) {
      if (!event.target.closest || !event.target.closest(".search")) list.hidden = true;
    });
  }

  // ---------------------------------------------------------------- table filter
  document.querySelectorAll("input.filter").forEach(function (filter) {
    var table = document.getElementById(filter.getAttribute("data-filter"));
    filter.addEventListener("input", function () {
      var words = filter.value.toLowerCase().split(/\s+/).filter(Boolean);
      table.querySelectorAll("tbody tr").forEach(function (row) {
        var text = row.textContent.toLowerCase();
        row.hidden = !words.every(function (w) { return text.indexOf(w) >= 0; });
      });
    });
  });
})();
