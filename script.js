const menuToggle = document.querySelector('.menu-toggle');
const navLinks = document.querySelector('.nav-links');

if (menuToggle && navLinks) {
  menuToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });
}

const yearEl = document.getElementById('year');
if (yearEl) {
  yearEl.textContent = new Date().getFullYear();
}

const reveals = document.querySelectorAll('.reveal');
const observer = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.15 }
);

reveals.forEach(el => observer.observe(el));


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
      }, 1500);
    } catch (err) {
      copyEmailBtn.textContent = 'Copy failed';
      setTimeout(() => {
        copyEmailBtn.textContent = original;
      }, 1500);
    }
  });
}
