// How the page looks, applied before it is drawn so that it never flashes the
// wrong theme. Chosen with the buttons at the top and kept in this browser;
// the window passes its own choice as ?look=light,colorblind,dyslexia&scale=1.2
// so that the documentation it shows looks like the window around it.
(function () {
  var store = {};
  try { store = JSON.parse(localStorage.getItem("harwood-look") || "{}"); } catch (e) {}
  var query = new URLSearchParams(location.search);
  if (query.has("look")) {
    var words = query.get("look").split(",");
    store = {
      theme: words.indexOf("light") >= 0 ? "light" : "dark",
      colorblind: words.indexOf("colorblind") >= 0,
      dyslexia: words.indexOf("dyslexia") >= 0,
      font: query.get("font") || "atkinson",
      scale: parseFloat(query.get("scale") || "1") || 1
    };
    try { localStorage.setItem("harwood-look", JSON.stringify(store)); } catch (e) {}
  }
  window.harwoodApplyLook = function (look) {
    var html = document.documentElement;
    var theme = look.theme || "system";
    if (theme === "system") {
      theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches
        ? "light" : "dark";
    }
    html.setAttribute("data-theme", theme);
    html.toggleAttribute("data-colorblind", !!look.colorblind);
    html.toggleAttribute("data-dyslexia", !!look.dyslexia);
    html.setAttribute("data-font", look.font || "atkinson");
    html.style.fontSize = (100 * (look.scale || 1)) + "%";
  };
  window.harwoodLook = function (look) {   // from the window, when its look changes
    try { localStorage.setItem("harwood-look", JSON.stringify(look)); } catch (e) {}
    window.harwoodApplyLook(look);
  };
  window.harwoodApplyLook(store);
})();
