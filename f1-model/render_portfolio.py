"""Regenerate the public case study from the measured experiment artifacts."""
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
metrics = json.loads((OUT / "metrics.json").read_text())
test, baseline = metrics["test"], metrics["baseline"]
improvement = 100 * (1 - test["MAE"] / baseline["MAE"])
escape = html.escape


def table(caption, headings, rows):
    header = ''.join(f'<th scope="col">{escape(h)}</th>' for h in headings)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(cell))}</td>' for cell in row) + '</tr>' for row in rows)
    return f'<div class="f1-table-wrap"><table class="f1-table"><caption>{escape(caption)}</caption><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


model_names = {"random_forest": "Random Forest (primary)", "gradient_boosting": "Gradient Boosting", "linear_regression": "Linear Regression"}
validation = table("Validation: rounds 17-21", ["Model", "MAE", "MSE", "R²"],
                   [(model_names[name], f"{v['MAE']:.2f}", f"{v['MSE']:.2f}", f"{v['R2']:.3f}")
                    for name, v in metrics['validation'].items()])
teams = table("Per-team test error", ["Team", "MAE (positions)", "Drivers evaluated"],
              [(name, f"{v['MAE']:.2f}", v['n']) for name, v in test['TeamMAE'].items()])
with (OUT / "test_predictions.csv").open() as handle:
    predictions = list(csv.DictReader(handle))
race_tables = ''
for race in dict.fromkeys(row['RaceName'] for row in predictions):
    rows = [r for r in predictions if r['RaceName'] == race]
    rows.sort(key=lambda r: float(r['FinalPosition']))
    body = table(f"{race}: finishers only", ["Driver", "Team", "Actual", "Predicted", "Error"],
                 [(r['Driver'], r['TeamName'], int(float(r['FinalPosition'])), f"{float(r['Prediction']):.2f}",
                   f"{float(r['AbsoluteError']):.2f}") for r in rows])
    race_tables += f'<details><summary>{escape(race)} · {len(rows)} finishers</summary>{body}</details>'

page = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="Aziz Mohammad's completed F1 prediction pipeline: 24 Grands Prix, chronological evaluation, Python, FastF1 and Random Forest. Explore measured results and source code." />
  <title>Formula 1 Race Prediction | Aziz Mohammad</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&amp;family=Rajdhani:wght@500;600;700&amp;display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="styles.css" />
  <link rel="stylesheet" href="f1-case.css" />
</head>
<body>
  <header class="site-header"><nav class="nav container" aria-label="Main navigation">
    <a class="logo" href="index.html#projects">AM <span>/ Projects</span></a>
    <button class="menu-toggle" aria-label="Toggle menu">Menu</button>
    <ul class="nav-links"><li><a href="#overview">Overview</a></li><li><a href="#process">Method</a></li><li><a href="#results">Results</a></li><li><a href="#source">Source</a></li></ul>
  </nav></header>
  <main class="container f1-page">
    <section class="f1-hero" aria-labelledby="project-title">
      <div><span class="f1-status">Completed · Phase 1</span>
        <h1 id="project-title">Predicting the<br />finishing order.</h1>
        <p class="lead">Formula 1 race prediction, from session data to a tested machine-learning pipeline.</p>
        <p>Built by Aziz Mohammad, UIC Mechanical Engineering. Connecting SAE Formula drivetrain experience with data engineering and predictive analytics.</p>
        <div class="f1-actions"><a class="btn btn-primary" href="#results">Explore Results</a><a class="btn btn-secondary" href="https://github.com/azizmoh1/Portfolio-Website/tree/main/f1-model">View Python Source</a></div>
      </div>
      <figure><img src="F1pic.jpg" alt="Formula 1 race car" width="1000" height="750" /><figcaption>2025 season study · 24 Grands Prix · pre-race prediction</figcaption></figure>
    </section>
    <div class="f1-metrics" aria-label="Measured Random Forest test results">
      <div class="f1-metric"><strong>{test['MAE']:.2f}</strong><span>Mean absolute error</span><small>Finishing positions, test set</small></div>
      <div class="f1-metric"><strong>{test['R2']:.3f}</strong><span>Test R²</span><small>{test['n']} finishers across 3 later races</small></div>
      <div class="f1-metric"><strong>{improvement:.1f}%</strong><span>Lower MAE than grid baseline</span><small>{baseline['MAE']:.2f} → {test['MAE']:.2f} positions</small></div>
      <div class="f1-metric"><strong>{metrics['finisher_rows']}</strong><span>Eligible driver-race entries</span><small>Across the full 2025 season</small></div>
    </div>
    <section id="overview" class="f1-section">
      <p class="eyebrow">01 / Engineering question</p><h2>What can we know before lights out?</h2>
      <p>Starting position matters, but it does not determine the result. This project asks whether qualifying conditions, team finish rate and recent car performance can improve on simply predicting that every driver finishes where they start.</p>
      <div class="f1-grid"><div class="card"><h3>Completed scope</h3><p>Cached session ingestion, historical features, data cleaning, three regression models, chronological validation, test metrics, plots, model export and CSV-based inference.</p><p>The data pipeline uses FastF1 with a Jolpica results fallback. Reproduction and automated tests are included with the source.</p></div>
      <div class="card"><h3>Prediction boundary</h3><p>Features are available after qualifying and grid publication, before race start. Race outcomes supply training labels and prior-round history only.</p><p>Results are conditional on a driver finishing. Live-race strategy, retirement prediction and detailed tire physics are future extensions.</p></div></div>
    </section>
    <section id="process" class="f1-section">
      <p class="eyebrow">02 / Data to prediction</p><h2>A pipeline with a clear time boundary.</h2>
      <div class="f1-grid"><div class="card"><ol class="f1-steps">
        <li><strong>Load and cache.</strong> Qualifying and race classification for all 24 rounds. {metrics['raw_rows']} driver-race records; {metrics['excluded_rows']} retirement, DNS or disqualification records excluded from supervised modeling.</li>
        <li><strong>Build historical features.</strong> Completed races by driver, team finish percentage, and team-average classified position over the last five earlier Grands Prix.</li>
        <li><strong>Add qualifying context.</strong> Published starting grid, fastest valid qualifying-lap tire compound, surface/air temperature, and a qualifying rainfall proxy.</li>
        <li><strong>Fit training transformations.</strong> Missing-value imputation, numerical scaling and categorical one-hot encoding. Temperature outlier limits use training data only.</li>
        <li><strong>Evaluate later races.</strong> Train on rounds 1-16, compare models on 17-21, and test the prespecified Random Forest on 22-24.</li>
      </ol></div><div class="card"><h3>Why whole races?</h3><p>A random driver split would mix the same event across training and test. Holding out complete later races gives a more realistic check of performance on an upcoming Grand Prix.</p>
      <p>The split contains {metrics['splits']['train']['rows']} training, {metrics['splits']['validation']['rows']} validation and {test['n']} test entries. All ten teams appear in each partition.</p>
      <p>Evaluation proceeds one race at a time: results from an earlier completed test race may become historical features for the next. Model weights remain fixed.</p>
      <p class="f1-note">Track temperature measures the surface; air temperature measures the ambient air. The rainfall feature is a weather proxy, not a measurement of track wetness.</p></div></div>
    </section>
    <section id="results" class="f1-section">
      <p class="eyebrow">03 / Measured results</p><h2>Tested on the final three Grands Prix.</h2>
      <p>Las Vegas, Qatar and Abu Dhabi 2025. Random Forest achieved MAE {test['MAE']:.2f}, MSE {test['MSE']:.2f}, and R² {test['R2']:.3f}. Average absolute error is not a confidence interval or a guarantee for any individual driver.</p>
      <div class="f1-grid">
        <figure class="card"><a href="f1-model/outputs/pred_vs_actual.png"><img class="f1-plot" src="f1-model/outputs/pred_vs_actual.png" alt="Scatter plot of actual and predicted finishing positions for the held-out 2025 races" width="1440" height="1080" loading="lazy" /></a><figcaption>Each point is one finisher. The dashed line shows a perfect prediction. Click to view the full-resolution plot.</figcaption></figure>
        <figure class="card"><a href="f1-model/outputs/feature_importances.png"><img class="f1-plot" src="f1-model/outputs/feature_importances.png" alt="Top ten Random Forest feature importances, led by starting grid position" width="1620" height="1080" loading="lazy" /></a><figcaption>Starting position is the strongest model signal. Impurity-based importance describes this fitted model; it does not establish cause and effect.</figcaption></figure>
      </div>
      <div class="f1-grid" style="margin-top:1.25rem"><div class="card"><h3>How the alternatives compared</h3>{validation}<p class="f1-note">Linear Regression had the lowest validation MAE. Random Forest remains the prespecified primary experiment; the test set was not used to tune or select the model.</p></div>
      <div class="card"><h3>Error by team</h3>{teams}<p>Small sample counts make team-level comparisons uncertain.</p></div></div>
      <details><summary>Inspect individual test predictions</summary><p>Continuous predicted positions can tie and do not enforce a unique race ranking.</p>{race_tables}</details>
      <details><summary>View the error distribution</summary><img class="f1-plot" src="f1-model/outputs/residuals.png" alt="Histogram of predicted minus actual finishing position" width="1440" height="720" loading="lazy" /></details>
    </section>
    <section id="lessons" class="f1-section">
      <p class="eyebrow">04 / What this establishes</p><h2>A working baseline, with room to improve.</h2>
      <div class="f1-grid"><div class="card"><h3>Engineering lessons</h3><p>Reliable feature definitions matter: five races means five events, not five individual car results. Lapped finishers belong in the dataset. Team finish percentage combines mechanical failures with crashes and other causes.</p><p>The grid baseline is competitive, and the simpler linear model performed best on validation. Greater model complexity does not automatically produce better predictions.</p></div>
      <div class="card"><h3>Limits and next experiments</h3><p>This is a single-season study with only three test events and excludes drivers who did not finish. It does not yet demonstrate performance across seasons or predict a complete field including retirements.</p><p>Next experiments: multi-season evaluation, retirement probability, tire degradation, DRS speed delta and circuit-specific pit-lane loss. Stationary service time and total pit-stop time loss must be modeled separately.</p></div></div>
    </section>
    <section id="source" class="f1-section">
      <p class="eyebrow">05 / Reproduce the work</p><h2>Code, data and results you can inspect.</h2>
      <div class="card"><p>The repository includes modular Python functions, automated tests, dependency versions, the derived feature dataset, individual predictions and all figures. No synthetic examples contribute to these reported metrics.</p>
      <div class="f1-actions"><a class="btn btn-primary" href="https://github.com/azizmoh1/Portfolio-Website/tree/main/f1-model">Source &amp; Setup Guide</a><a class="btn btn-secondary" href="f1-model/outputs/summary.txt">Measured Report</a><a class="btn btn-secondary" href="f1-model/outputs/dataset.csv" download>Download Dataset</a><a class="btn btn-secondary" href="f1-model/outputs/test_predictions.csv" download>Test Predictions</a></div>
      <pre><code>python -m pip install -r requirements.txt
python f1_pipeline.py train --dataset outputs/dataset.csv \\
    --output-dir outputs-reproduced
python -m pytest -q</code></pre>
      <p>Data sources: <a href="https://docs.fastf1.dev/">FastF1</a> and <a href="https://github.com/jolpica/jolpica-f1">Jolpica</a>. <a href="f1-model/outputs/metrics.json">Full metrics and provenance</a>. Independent educational project; not affiliated with Formula 1.</p></div>
    </section>
  </main>
  <footer class="site-footer"><p>Aziz Mohammad · Mechanical Engineering · <a href="index.html#projects">Back to projects</a></p></footer>
  <script src="script.js"></script>
</body>
</html>
'''
(ROOT.parent / "project-f1.html").write_text(page)
print("Updated project-f1.html from measured results")
