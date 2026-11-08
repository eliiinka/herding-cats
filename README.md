# 🐱 Herding Quantum Cats — Qiskit Fall Fest 2026 Stuttgart

Build the cleanest and the biggest GHZ ("cat") state you can on an IBM Heron
processor: first on a realistic simulator, then on the real **ibm_kingston**.

## Files
| File | What |
|---|---|
| `herding_cats_challenge.ipynb` | The challenge: Missions 0–8, start here |
| `herding_cats.py` | Toolkit used by the notebook (keep it in the same folder) |
| `PITCH.md` | What to put in your 5-minute pitch deck |

## Before the event (please!)
1. Python 3.10–3.13. We recommend a fresh virtual environment, so these versions don't clash with
   other projects on your laptop:
   ```bash
   python -m venv fallfest
   source fallfest/bin/activate        # Windows: fallfest\Scripts\activate
   pip install "qiskit[visualization]==2.6.1" "qiskit-ibm-runtime==0.51.0" "qiskit-aer==0.18.0" jupyterlab
   ```
   (conda users: `conda create -n fallfest python=3.12 && conda activate fallfest`, then the same `pip install`.)
2. Create a free IBM Quantum account (Open Plan) at <https://quantum.cloud.ibm.com>,
   create an API key, and save it once on your laptop:
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(token="<your API key>", instance="<your instance CRN>",
                                     set_as_default=True)
   QiskitRuntimeService().backend("ibm_kingston")      # should print a backend, no error
   ```
   Never paste your API key into a notebook you upload.
3. Open the notebook and run the **Setup** cell. If it prints the versions, you're ready.

## Rules in one breath
Develop on the simulator; use real hardware only for the final runs (10 QPU-minutes
per person per 28 days, and queues can be long). Final hardware job submitted by **18:45**.
By **20:00**, one person per team posts the executed notebook (`team_<name>.ipynb`),
`submission_<team>.json` and `pitch_<team>.pdf` or `.pptx` (a 5-minute deck, see `PITCH.md`) in the organizers'
Discord channel. The deck is not scored, but the winning team presents it at the award ceremony.
