# Software prior-art review

Review date: 2026-10-03. The review is scoped to public documentation and
source descriptions, not a legal opinion. Existing components should be
reused or interoperated with under their licenses.

| System | Existing strength | Boundary in PhysioWorld | Interoperation decision |
|---|---|---|---|
| WFDB / PhysioNet tools | Waveform and annotation formats, record I/O, processing, public data distribution | Does not provide the cross-layer task/resource/policy execution contract | Use `WFDBAdapter`; preserve WFDB metadata and annotations |
| NeuroKit2 | Modality processing, simulation, analysis, examples | Preprocessing and downstream capability evaluation have distinct contracts | Explicit external preprocessing with preserved lineage; no implicit dependency |
| Pulse Physiology Engine | Mechanistic human physiology simulation and training integrations | Coupling requires explicit state, variable, unit, time and evidence contracts | External source/coupling must declare implemented fidelity and model provenance |
| MATLAB/Simulink biomedical tooling | Mature signal processing, block simulation, optimization ecosystem | Proprietary/runtime-dependent; no common open provenance and task-law contract here | Import/export through stream and manifest boundaries |
| Scilab biomedical toolboxes | Open numerical environment and signal processing | Different execution/runtime model and no unified clinical workload registry | Interoperate through files and declared metadata |
| Wearable/IoT simulators | Device/network/topology and sensor workload simulation | Usually packet/energy-centric rather than downstream conditional laws | Reuse channel ideas; retain PhysioWorld task/capability semantics |
| Semantic/task-oriented communication simulators | Semantic compression, task utility, channel experiments | Often not physiology- or clinical-provenance-aware | Use compatible channel metrics without transmitting ground-truth labels |
| Digital-twin health systems | Patient/system state and intervention modeling | Varying auditability, data access, and task-law definitions | Keep causal interfaces explicit and avoid causal claims from replay |

Primary public references consulted:

- WFDB documentation: <https://wfdb.readthedocs.io/en/latest/>;
- PhysioNet MIT-BIH Arrhythmia Database: <https://physionet.org/content/mitdb/1.0.0/>;
- NeuroKit2 documentation: <https://neuropsychology.github.io/NeuroKit/>;
- Pulse Physiology Engine: <https://pulse.kitware.com/>.

The intended distinction is architectural and scientific: an executable,
auditable cross-layer environment with explicit history, task-law,
capability, resource, communication, policy, causal, and provenance
semantics. It is not a claim that any individual component is novel.
