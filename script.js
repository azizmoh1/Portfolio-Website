const menuToggle = document.querySelector('.menu-toggle');
const navLinks = document.querySelector('.nav-links');
const navAnchors = document.querySelectorAll('.nav-links a');
const sections = document.querySelectorAll('main section[id]');

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

const yearEl = document.getElementById('year');
if (yearEl) {
  yearEl.textContent = new Date().getFullYear();
}

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

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

const sectionObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;

      const id = entry.target.getAttribute('id');
      navAnchors.forEach(link => {
        const target = link.getAttribute('href');
        link.classList.toggle('active', target === `#${id}`);
      });
    });
  },
  { threshold: 0.5 }
);

sections.forEach(section => sectionObserver.observe(section));

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
    } catch (err) {
      copyEmailBtn.textContent = 'Copy failed';
      setTimeout(() => {
        copyEmailBtn.textContent = original;
      }, 1400);
    }
  });
}
