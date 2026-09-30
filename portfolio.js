(() => {
  const search=document.querySelector('#project-search');
  const type=document.querySelector('#project-type');
  const cards=[...document.querySelectorAll('.work-card')];
  const buttons=[...document.querySelectorAll('.work-filter')];
  let discipline='all';
  const update=()=>{
    const query=search.value.trim().toLowerCase();
    let visible=0;
    cards.forEach(card=>{
      const match=(!query||card.textContent.toLowerCase().includes(query))&&(type.value==='all'||card.dataset.kind===type.value)&&(discipline==='all'||card.dataset.disciplines.split(' ').includes(discipline));
      card.hidden=!match;if(match)visible++;
    });
    document.querySelector('#project-count').textContent=`${visible} of ${cards.length} projects`;
    document.querySelector('#no-projects').hidden=visible!==0;
    buttons.forEach(button=>{const active=button.dataset.discipline===discipline;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
  };
  search.addEventListener('input',update);type.addEventListener('change',update);
  buttons.forEach(button=>button.addEventListener('click',()=>{discipline=button.dataset.discipline;update();}));
  document.querySelector('#reset-projects').addEventListener('click',()=>{search.value='';type.value='all';discipline='all';update();search.focus();});
  const preview=document.querySelector('.resume-preview');
  preview.addEventListener('toggle',()=>{if(preview.open){const frame=preview.querySelector('iframe');if(!frame.getAttribute('src'))frame.src=frame.dataset.resumeSrc;}});
  const menu=document.querySelector('.menu-toggle');
  menu.addEventListener('click',()=>menu.setAttribute('aria-expanded',String(document.querySelector('.nav-links').classList.contains('open'))));
  document.querySelectorAll('.nav-links a').forEach(a=>a.addEventListener('click',()=>menu.setAttribute('aria-expanded','false')));
  update();
})();
