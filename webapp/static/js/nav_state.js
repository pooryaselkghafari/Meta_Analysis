// Shared across every page: greys out nav links whose prerequisite pipeline
// stage hasn't completed yet, and blocks navigating to them. Included on
// every template (including locked.html) so the nav bar always reflects
// real progress, not just the page you happen to be on.
(function () {
  const NAV_REQUIRES = {
    "/": null,
    "/results": "corpus",
    "/cheap-ai": "corpus",
    "/detection-ai": "cheap_ai",
    "/main-ai": "detection_ai",
    "/regression": "main_ai",
    "/dashboard": "main_ai",
    "/settings": null,
  };
  const STAGE_LABELS = {
    corpus: "Corpus analysis",
    cheap_ai: "Cheap AI — Classification",
    detection_ai: "Detection AI — Detection",
    main_ai: "Main AI — Extraction",
    regression: "Regression — Meta-analysis",
    dashboard: "Dashboard — Key findings",
  };

  async function applyNavState() {
    const nav = document.getElementById("app-nav") || document.querySelector(".app-nav");
    if (!nav) return;
    let state;
    try {
      const res = await fetch("/api/pipeline/state");
      state = await res.json();
    } catch (e) {
      return; // if the state endpoint is unreachable, leave the nav as-is
    }
    nav.querySelectorAll("a[href]").forEach(a => {
      const href = a.getAttribute("href");
      const required = NAV_REQUIRES[href];
      const ready = !required || state[`${required}_done`];
      a.classList.toggle("nav-disabled", !ready);
      if (!ready) {
        a.title = `Run ${STAGE_LABELS[required]} first`;
        a.addEventListener("click", (e) => e.preventDefault());
      } else {
        a.removeAttribute("title");
      }
    });
  }

  const NEW_PROJECT_VALUE = "__new__";

  async function loadProjectSwitcher() {
    const el = document.getElementById("project-switcher");
    if (!el) return;
    let data;
    try {
      const res = await fetch("/api/projects");
      data = await res.json();
    } catch (e) {
      return; // leave the switcher blank if the endpoint is unreachable
    }
    const select = document.createElement("select");
    select.className = "project-select mono";
    select.title = "Switch project — each project has its own papers, targets, and results";
    (data.projects || []).forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name;
      opt.selected = p.id === data.active_project_id;
      select.appendChild(opt);
    });
    const newOpt = document.createElement("option");
    newOpt.value = NEW_PROJECT_VALUE;
    newOpt.textContent = "+ New project…";
    select.appendChild(newOpt);

    select.addEventListener("change", async () => {
      const chosen = select.value;
      if (chosen === NEW_PROJECT_VALUE) {
        const name = window.prompt("Name the new project (its own separate papers, targets, and results):");
        select.value = data.active_project_id; // reset selection while we decide what to do
        if (!name || !name.trim()) return;
        await fetch("/api/projects", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name.trim() }),
        });
        window.location.href = "/";
        return;
      }
      if (chosen === data.active_project_id) return;
      await fetch(`/api/projects/${chosen}/activate`, { method: "POST" });
      window.location.href = "/";
    });

    el.innerHTML = "";
    el.appendChild(select);
  }

  applyNavState();
  loadProjectSwitcher();
})();
