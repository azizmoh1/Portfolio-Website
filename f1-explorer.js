// Retrospective 2025 test predictions; this page never presents them as live forecasts.
(() => {
  const select = document.getElementById('f1-race-select');
  if (!select) return;
  const body = document.getElementById('f1-race-body');
  const summary = document.getElementById('f1-race-summary');
  const caption = document.getElementById('f1-race-caption');

  const render = race => {
    body.replaceChildren();
    let totalError = 0;
    race.drivers.forEach(driver => {
      const row = document.createElement('tr');
      [driver.predicted, driver.driver, driver.team, driver.actual,
        `${(driver.risk * 100).toFixed(1)}%`, driver.status].forEach(value => {
        const cell = document.createElement('td');
        cell.textContent = String(value);
        row.appendChild(cell);
      });
      totalError += Math.abs(driver.predicted - driver.actual);
      body.appendChild(row);
    });
    caption.textContent = `${race.name} 2025 - ordered by predicted position`;
    summary.textContent = `${race.drivers.length} drivers | Mean absolute rank error: ${(totalError / race.drivers.length).toFixed(2)} positions. Historical evaluation, not a live forecast.`;
  };

  fetch('f1-model/final_outputs/race_forecasts.json')
    .then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(races => {
      if (!Array.isArray(races) || races.length === 0) throw new Error('No race data');
      select.replaceChildren();
      races.forEach(race => {
        const option = document.createElement('option');
        option.value = String(race.round);
        option.textContent = `${race.round}. ${race.name}`;
        select.appendChild(option);
      });
      select.value = String(races[races.length - 1].round);
      select.disabled = false;
      select.addEventListener('change', () => render(races.find(race => String(race.round) === select.value)));
      render(races[races.length - 1]);
    })
    .catch(() => {
      summary.textContent = 'The race explorer could not load. The complete predictions are available in the CSV download below.';
      select.replaceChildren(new Option('Race data unavailable', ''));
    });
})();
