const book = document.getElementById("book");
const player = document.getElementById("player");
const nowPlaying = document.getElementById("now-playing");
const thumbs = document.getElementById("thumbs");

const DEFAULT_LAYOUT = "paged";

let currentAudio = null;
let pagesData = [];
let layoutMode = DEFAULT_LAYOUT;
let currentIndex = 0;
let pagedControls = null;
const observers = new Map();
let pageObserver = null;
const pageVisibility = new Map();

let editState = null;

function startEditRect(page, startX, startY) {
  const overlay = page.querySelector(".overlay");
  const rectEl = document.createElement("div");
  rectEl.className = "editor-rect";
  overlay.appendChild(rectEl);
  editState = { page, rectEl, startX, startY };
}

function updateEditRect(currentX, currentY) {
  if (!editState) return;
  const { rectEl, startX, startY } = editState;
  const x = Math.min(startX, currentX);
  const y = Math.min(startY, currentY);
  const w = Math.abs(currentX - startX);
  const h = Math.abs(currentY - startY);
  rectEl.style.left = `${x * 100}%`;
  rectEl.style.top = `${y * 100}%`;
  rectEl.style.width = `${w * 100}%`;
  rectEl.style.height = `${h * 100}%`;
  editState.current = { x, y, w, h };
}

function finishEditRect() {
  if (!editState) return;
  const { rectEl, page, current } = editState;
  rectEl.remove();
  if (!current || current.w < 0.005 || current.h < 0.005) {
    editState = null;
    return;
  }
  const pageNum = page.dataset.page;
  const chapter = window.prompt(`Chapter for page ${pageNum}? (e.g. 3.4)`);
  if (!chapter) {
    editState = null;
    return;
  }
  const entry = {
    chapter: chapter.trim(),
    x: Number(current.x.toFixed(5)),
    y: Number(current.y.toFixed(5)),
    w: Number(current.w.toFixed(5)),
    h: Number(current.h.toFixed(5)),
  };
  const snippet = JSON.stringify(entry);
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(snippet).catch(() => {});
  }
  window.alert(`Hotspot JSON copied to clipboard.
Page ${pageNum}: ${snippet}`);
  editState = null;
}

function handleEditPointerDown(event) {
  return;
}

function handleEditPointerMove(event) {
  return;
}

function handleEditPointerUp() {
  return;
}

function playAudio(hotspot) {
  const url = new URL(hotspot.audio, window.location.href);
  if (currentAudio && currentAudio === url.href) {
    player.currentTime = 0;
  } else {
    player.pause();
    player.src = url.href;
    currentAudio = url.href;
  }
  const rawName = hotspot.audio.split("/").pop() || "";
  const cleanName = decodeURIComponent(rawName).replace(/\.mp3$/i, "");
  const pageLabel = hotspot.page ? ` (Page ${hotspot.page})` : "";
  if (hotspot.chapter) {
    nowPlaying.textContent = `Playing Kap ${hotspot.chapter} — ${cleanName}${pageLabel}`;
  } else {
    nowPlaying.textContent = `Playing ${cleanName}${pageLabel}`;
  }
  player.play();
}

function makeHotspot(hotspot) {
  const btn = document.createElement("button");
  btn.className = "hotspot";
  btn.style.left = `${hotspot.x * 100}%`;
  btn.style.top = `${hotspot.y * 100}%`;
  btn.style.width = `${hotspot.w * 100}%`;
  btn.style.height = `${hotspot.h * 100}%`;
  btn.setAttribute("aria-label", `Play chapter ${hotspot.chapter}`);
  btn.addEventListener("click", () => playAudio(hotspot));
  return btn;
}

function makeLazyImage(src, alt, width, height) {
  const img = document.createElement("img");
  img.alt = alt;
  img.dataset.src = src;
  img.loading = "lazy";
  img.decoding = "async";
  if (width && height) {
    img.width = width;
    img.height = height;
  }
  return img;
}

function makePage(pageData) {
  const wrapper = document.createElement("section");
  wrapper.className = "page";
  wrapper.dataset.page = String(pageData.page);

  const img = makeLazyImage(pageData.image, `Page ${pageData.page}`, pageData.width, pageData.height);

  const overlay = document.createElement("div");
  overlay.className = "overlay";

  pageData.hotspots.forEach((hotspot) => {
    overlay.appendChild(makeHotspot({ ...hotspot, page: pageData.page }));
  });

  wrapper.appendChild(img);
  wrapper.appendChild(overlay);
  return wrapper;
}

function getObserver(root) {
  const key = root ? "root" : "window";
  if (observers.has(key)) {
    return observers.get(key);
  }
  if (!("IntersectionObserver" in window)) {
    observers.set(key, null);
    return null;
  }
  const options = root
    ? { root, rootMargin: "100px 0px" }
    : { rootMargin: "300px 0px" };

  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) {
        return;
      }
      const img = entry.target;
      const src = img.dataset.src;
      if (src) {
        img.src = src;
        delete img.dataset.src;
      }
      obs.unobserve(img);
    });
  }, options);

  observers.set(key, observer);
  return observer;
}

function observeImages(container, root = null) {
  const images = Array.from(container.querySelectorAll("img[data-src]"));
  const observer = getObserver(root);
  images.forEach((img) => {
    if (!observer) {
      img.src = img.dataset.src;
      delete img.dataset.src;
      return;
    }
    observer.observe(img);
  });
}

function renderScroll(pages) {
  book.classList.add("layout-scroll");
  pages.forEach((pageData) => {
    book.appendChild(makePage(pageData));
  });
  observeImages(book);
  setupPageObserver();
}

function renderPaged(pages) {
  const nav = document.createElement("div");
  nav.style.display = "flex";
  nav.style.justifyContent = "space-between";
  nav.style.alignItems = "center";
  nav.style.gap = "12px";
  nav.style.margin = "0 auto 16px";
  nav.style.width = "min(920px, 100%)";

  const prev = document.createElement("button");
  const next = document.createElement("button");
  const label = document.createElement("div");

  prev.textContent = "Previous";
  next.textContent = "Next";

  [prev, next].forEach((btn) => {
    btn.style.padding = "10px 16px";
    btn.style.borderRadius = "10px";
    btn.style.border = "1px solid rgba(0,0,0,0.12)";
    btn.style.background = "#fff";
    btn.style.cursor = "pointer";
  });

  nav.appendChild(prev);
  nav.appendChild(label);
  nav.appendChild(next);
  book.appendChild(nav);

  const pageHost = document.createElement("div");
  book.appendChild(pageHost);

  function update() {
    pageHost.innerHTML = "";
    pageHost.appendChild(makePage(pages[currentIndex]));
    observeImages(pageHost);
    label.textContent = `Page ${pages[currentIndex].page} of ${pages.length}`;
    prev.disabled = currentIndex === 0;
    next.disabled = currentIndex === pages.length - 1;
    setActivePage(pages[currentIndex].page);
  }

  prev.addEventListener("click", () => {
    if (currentIndex > 0) {
      currentIndex -= 1;
      update();
    }
  });

  next.addEventListener("click", () => {
    if (currentIndex < pages.length - 1) {
      currentIndex += 1;
      update();
    }
  });

  pagedControls = { update };
  update();
}

function setLayout(layout) {
  layoutMode = "paged";
  localStorage.setItem("layoutMode", layoutMode);
  if (!pagesData.length) {
    return;
  }
  book.innerHTML = "";
  book.classList.remove("layout-scroll");
  disconnectPageObserver();
  renderPaged(pagesData);
}

function setActivePage(pageNumber) {
  const pageStr = String(pageNumber);
  const items = Array.from(thumbs.querySelectorAll(".thumb"));
  items.forEach((item) => {
    item.classList.toggle("active", item.dataset.page === pageStr);
  });
}

function jumpToPage(pageNumber) {
  const target = Math.min(Math.max(pageNumber, 1), pagesData.length);
  const index = pagesData.findIndex((p) => p.page === target);
  if (index === -1) {
    return;
  }
  currentIndex = index;
  if (layoutMode === "paged") {
    if (pagedControls) {
      pagedControls.update();
    } else {
      setLayout("paged");
    }
  } else {
    const node = book.querySelector(`.page[data-page="${target}"]`);
    if (node) {
      node.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    setActivePage(target);
  }
}

function setupPageObserver() {
  disconnectPageObserver();
  if (!("IntersectionObserver" in window)) {
    return;
  }
  pageVisibility.clear();
  const pages = Array.from(book.querySelectorAll(".page"));
  pageObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const pageNum = entry.target.dataset.page;
        pageVisibility.set(pageNum, entry.intersectionRatio);
      });
      let bestPage = null;
      let bestRatio = 0;
      pageVisibility.forEach((ratio, pageNum) => {
        if (ratio > bestRatio) {
          bestRatio = ratio;
          bestPage = pageNum;
        }
      });
      if (bestPage) {
        setActivePage(bestPage);
      }
    },
    { threshold: [0.25, 0.5, 0.75] }
  );

  pages.forEach((page) => pageObserver.observe(page));
}

function disconnectPageObserver() {
  if (pageObserver) {
    pageObserver.disconnect();
    pageObserver = null;
  }
}

function renderThumbs(pages) {
  thumbs.innerHTML = "";
  pages.forEach((pageData) => {
    const btn = document.createElement("button");
    btn.className = "thumb";
    btn.dataset.page = String(pageData.page);
    btn.style.aspectRatio = `${pageData.width} / ${pageData.height}`;
    const img = document.createElement("img");
    img.alt = `Page ${pageData.page}`;
    img.src = pageData.thumb || pageData.image;
    img.loading = "eager";
    img.decoding = "async";
    btn.appendChild(img);
    const label = document.createElement("span");
    label.className = "thumb-label";
    label.textContent = String(pageData.page);
    btn.appendChild(label);
    btn.addEventListener("click", () => jumpToPage(pageData.page));
    thumbs.appendChild(btn);
  });
}


fetch("data.json")
  .then((res) => res.json())
  .then((data) => {
    pagesData = data.pages;
    renderThumbs(pagesData);
    setLayout(DEFAULT_LAYOUT);
  })
  .catch((err) => {
    nowPlaying.textContent = `Failed to load data.json: ${err}`;
  });
