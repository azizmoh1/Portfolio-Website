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
