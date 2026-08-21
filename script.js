// Theme toggle with localStorage persistence
const themeToggle = document.querySelector('.theme-toggle');
const themeText = document.querySelector('.theme-text');
const rootEl = document.documentElement;

const setTheme = theme => {
  rootEl.dataset.theme = theme;
  try {
    localStorage.setItem('portfolio-theme', theme);
  } catch {
    // Theme still changes for the current page even if storage is blocked.
  }
  if (themeToggle) {
    const isLight = theme === 'light';
    themeToggle.setAttribute('aria-pressed', String(isLight));
    themeToggle.setAttribute('aria-label', isLight ? 'Switch to dark mode' : 'Switch to light mode');
  }
  if (themeText) themeText.textContent = theme === 'light' ? 'Light' : 'Dark';
};

let savedTheme = null;
try {
  savedTheme = localStorage.getItem('portfolio-theme');
} catch {
  savedTheme = null;
}
const preferredTheme = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
setTheme(savedTheme || preferredTheme);

themeToggle?.addEventListener('click', () => {
  setTheme(rootEl.dataset.theme === 'light' ? 'dark' : 'light');
});

const menuToggle = document.querySelector('.menu-toggle');
const navLinks = document.querySelector('.nav-links');
const navAnchors = Array.from(document.querySelectorAll('.nav-links a'));

if (menuToggle && navLinks) {
  menuToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });

  navAnchors.forEach(link => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('open');
    });
  });
}

// Smooth scroll for in-page navigation and hero buttons
Array.from(document.querySelectorAll('a[href^="#"]')).forEach(link => {
  link.addEventListener('click', event => {
    const target = link.getAttribute('href');
    if (!target || target === '#') return;

    const el = document.querySelector(target);
    if (!el) return;

    event.preventDefault();
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

const yearEl = document.getElementById('year');
if (yearEl) {
  yearEl.textContent = new Date().getFullYear();
}

const revealTargets = document.querySelectorAll('.reveal, .slide-up');
const revealObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.16 }
);
revealTargets.forEach(el => revealObserver.observe(el));

// Sticky-nav active section highlighting
const sectionIds = navAnchors
  .map(link => link.getAttribute('href'))
  .filter(href => href && href.startsWith('#') && href.length > 1)
  .map(href => href.slice(1));
const observedSections = sectionIds
  .map(id => document.getElementById(id))
  .filter(Boolean);

if (observedSections.length) {
  const sectionObserver = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        const id = entry.target.id;
        navAnchors.forEach(link => {
          link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
        });
      });
    },
    { threshold: 0.45 }
  );

  observedSections.forEach(section => sectionObserver.observe(section));
}

// Image preview modal (gallery/CAD renders/diagrams/prototypes)
const modal = document.querySelector('[data-modal]');
const modalImage = modal?.querySelector('.modal-image');
const modalCaption = modal?.querySelector('.modal-caption');
const modalClose = modal?.querySelector('.modal-close');

if (modal && modalImage && modalCaption && modalClose) {
  const closeModal = () => {
    modal.hidden = true;
    modalImage.src = '';
    modalCaption.textContent = '';
    document.body.style.overflow = '';
  };

  document.addEventListener('click', event => {
    const trigger = event.target.closest('[data-modal-img]');
    if (!trigger) return;

    const src = trigger.getAttribute('src');
    const caption = trigger.getAttribute('data-caption') || trigger.getAttribute('alt') || '';
    if (!src) return;

    modalImage.src = src;
    modalCaption.textContent = caption;
    modal.hidden = false;
    document.body.style.overflow = 'hidden';
  });

  modalClose.addEventListener('click', closeModal);
  modal.addEventListener('click', event => {
    if (event.target === modal) closeModal();
  });

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !modal.hidden) closeModal();
  });
}

const copyEmailBtn = document.getElementById('copy-email-btn');
if (copyEmailBtn) {
  copyEmailBtn.addEventListener('click', async () => {
    const email = copyEmailBtn.dataset.email;
    const original = copyEmailBtn.textContent;

    try {
      await navigator.clipboard.writeText(email);
      copyEmailBtn.textContent = 'Copied!';
      copyEmailBtn.classList.add('copied');
      setTimeout(() => {
        copyEmailBtn.textContent = original;
        copyEmailBtn.classList.remove('copied');
      }, 1400);
    } catch {
      copyEmailBtn.textContent = 'Copy failed';
      setTimeout(() => {
        copyEmailBtn.textContent = original;
      }, 1400);
    }
  });
}


// Latest GitHub repositories (lightweight client-side fetch)
(async function loadLatestRepos() {
  const grid = document.getElementById('repo-grid');
  if (!grid) return;

  try {
    const response = await fetch('https://api.github.com/users/azizmoh1/repos?sort=updated&per_page=6');
    if (!response.ok) throw new Error('GitHub fetch failed');
    const repos = await response.json();

    const filtered = repos.filter(repo => !repo.fork).slice(0, 6);
    if (!filtered.length) {
      grid.innerHTML = '<article class="card repo-card slide-up visible"><p>No public repositories found yet.</p></article>';
      return;
    }

    grid.innerHTML = filtered
      .map(
        repo => `
        <article class="card repo-card slide-up visible">
          <h3>${repo.name}</h3>
          <p>${repo.description ? repo.description : 'No description provided yet.'}</p>
          <div class="repo-meta">
            <span class="repo-star">Stars: ${repo.stargazers_count}</span>
            <a class="text-link" href="${repo.html_url}" target="_blank" rel="noopener noreferrer">View Repository</a>
          </div>
        </article>
      `
      )
      .join('');
  } catch (error) {
    grid.innerHTML = '<article class="card repo-card slide-up visible"><p>Unable to load repositories right now.</p></article>';
  }
})();


// Project filtering for recruiter-specific scanning
const filterButtons = Array.from(document.querySelectorAll('[data-filter]'));
const projectTiles = Array.from(document.querySelectorAll('.project-tile[data-tags]'));

filterButtons.forEach(button => {
  button.addEventListener('click', () => {
    const filter = button.dataset.filter;
    filterButtons.forEach(btn => btn.classList.toggle('active', btn === button));

    projectTiles.forEach(tile => {
      const tags = (tile.dataset.tags || '').split(/\s+/);
      const show = filter === 'all' || tags.includes(filter);
      tile.classList.toggle('is-filtered-out', !show);
      tile.setAttribute('aria-hidden', String(!show));
      tile.tabIndex = show ? 0 : -1;
    });
  });
});

// Animated metric counters
const counterEls = Array.from(document.querySelectorAll('[data-count]'));
const formatCounter = (value, decimals) => Number(value).toLocaleString(undefined, {
  minimumFractionDigits: decimals,
  maximumFractionDigits: decimals
});

const counterObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const el = entry.target;
    const target = Number(el.dataset.count || 0);
    const decimals = Number(el.dataset.decimals || 0);
    const prefix = el.dataset.prefix || '';
    const suffix = el.dataset.suffix || '';
    const duration = 900;
    const start = performance.now();

    const tick = now => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = target * eased;
      el.textContent = `${prefix}${formatCounter(current, decimals)}${suffix}`;
      if (progress < 1) requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
    counterObserver.unobserve(el);
  });
}, { threshold: 0.45 });

counterEls.forEach(el => counterObserver.observe(el));


// Image comparison sliders for CAD/render vs. documentation views
const compareBlocks = Array.from(document.querySelectorAll('[data-compare]'));
compareBlocks.forEach(block => {
  const input = block.querySelector('.compare-range');
  if (!input) return;
  const update = () => block.style.setProperty('--position', `${input.value}%`);
  input.addEventListener('input', update);
  update();
});
